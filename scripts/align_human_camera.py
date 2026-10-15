#!/usr/bin/env python3
"""Relate WHAM's world frame to the MoGe-2 camera frame for one Track 1 episode.

Runs inside containers/wham.sif (WHAM's pickles hold torch tensors) with the workspace at
/track1. The Track 1 exo cameras are static (WHAM's DPVO recovers an identity trajectory), so
one rigid transform maps WHAM world metres into the camera. It is solved by a single PnP over
every observed frame: WHAM world-frame COCO joints (J_regressor_coco on the exported SMPL
mesh) against WHAM's own ViTPose 2D keypoints, using MoGe-2's intrinsics. WHAM itself ran with
its default focal length, so its camera-frame translation is not used.

WHAM's world root trajectory is integrated from predicted velocities and drifts against the
static camera (reprojection error grows through the clip). A per-frame, translation-only PnP
against the same keypoints measures that drift; it is median- and Gaussian-filtered so only its
low-frequency part is removed, which leaves WHAM's smooth local motion (ACC-H) intact.

MoGe-2 metric depth and the SMPL body imply slightly different metric scales. depth_scale is
the median ratio of MoGe depth to the posed, drift-corrected SMPL surface depth, from a coarse
z-buffer of the body. Object poses tracked in MoGe units are divided by it to share the human's
scale.

Writes outputs/alignment/episode_XXXXXX/:
  alignment.json  R_cw, t_cw: p_camera = R_cw @ (p_wham_world + drift_world[t]) + t_cw (metres);
                  depth_scale: MoGe depth / human-consistent depth; intrinsics; diagnostics
  drift.npz       drift_world [T, 3]: smoothed world-frame offset added to the WHAM human root
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares

ROOT = Path(os.environ.get("TRACK1_ROOT", "/track1"))
COCO_REGRESSOR = ROOT / "weights" / "wham" / "dataset" / "body_models" / "J_regressor_coco.npy"
MIN_KEYPOINT_CONFIDENCE = 0.5
MIN_FRAME_KEYPOINTS = 8
ZBUFFER_CELL = 4  # px; coarse cells let front-surface vertices cover the silhouette densely
DRIFT_MEDIAN_FRAMES = 15
DRIFT_SIGMA_FRAMES = 20.0  # ~0.67 s at 30 fps: removes drift, keeps WHAM's local dynamics


def median_intrinsics(episode_dir: Path) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted((episode_dir / "intrinsics").glob("*.json"))]
    return {k: float(np.median([r[k] for r in rows])) for k in ("fx", "fy", "cx", "cy", "width", "height")}


def depth_m(path: Path) -> np.ndarray:
    raw = np.array(Image.open(path)).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(raw > 0, 65535.0 / raw - 1.0, np.inf)


def project(points_c: np.ndarray, k: dict) -> np.ndarray:
    return np.stack([k["fx"] * points_c[..., 0] / points_c[..., 2] + k["cx"],
                     k["fy"] * points_c[..., 1] / points_c[..., 2] + k["cy"]], axis=-1)


def frame_offsets(joints_c: np.ndarray, keypoints: np.ndarray, confident: np.ndarray, k: dict) -> np.ndarray:
    """Per-frame camera-frame translation that best reprojects the joints (NaN if too few keypoints)."""
    offsets = np.full((len(joints_c), 3), np.nan)
    for i, (joints, observed, mask) in enumerate(zip(joints_c, keypoints, confident)):
        if mask.sum() < MIN_FRAME_KEYPOINTS:
            continue
        weights = np.sqrt(observed[mask, 2])[:, None]

        def residual(offset):
            return ((project(joints[mask] + offset, k) - observed[mask, :2]) * weights).ravel()

        fit = least_squares(residual, np.zeros(3), loss="huber", f_scale=8.0, max_nfev=60)
        if fit.success and (joints[mask, 2] + fit.x[2] > 0.3).all():
            offsets[i] = fit.x
    return offsets


def smooth_dense(frames: np.ndarray, values: np.ndarray, total: int) -> np.ndarray:
    """Interpolate sparse per-frame values onto every video frame, then median + Gaussian filter."""
    valid = np.isfinite(values).all(axis=1)
    frames, values = frames[valid], values[valid]
    dense = np.stack([np.interp(np.arange(total), frames, values[:, c]) for c in range(3)], axis=1)
    dense = median_filter(dense, size=(DRIFT_MEDIAN_FRAMES, 1), mode="nearest")
    return gaussian_filter1d(dense, DRIFT_SIGMA_FRAMES, axis=0, mode="nearest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--depth_stride", type=int, default=5)
    args = parser.parse_args()

    tag = f"episode_{args.episode:06d}"
    wham_dir = ROOT / "outputs" / "wham" / tag
    object_dir = ROOT / "outputs" / "object" / tag
    export = np.load(wham_dir / "wham_smpl_world.npz")
    subject = json.loads(str(export["report"]))["subject"]
    vertices_w = export["vertices"].astype(np.float64)  # [T, 6890, 3], dense over video frames
    observed = np.unique(export["source_frame_ids"].astype(np.int64))

    tracking = joblib.load(wham_dir / "tracking_results.pth")
    track = tracking[int(subject)] if int(subject) in tracking else tracking[subject]
    track_frames = np.asarray(track["frame_id"], dtype=np.int64)
    keypoints = np.asarray(track["keypoints"], dtype=np.float64)  # [N, 17, 3] pixel x, y, confidence
    keep = np.isin(track_frames, observed)
    track_frames, keypoints = track_frames[keep], keypoints[keep]

    regressor = np.load(COCO_REGRESSOR).astype(np.float64)  # [17, 6890]
    joints_w = np.einsum("jv,fvc->fjc", regressor, vertices_w[track_frames])  # [N, 17, 3]

    k = median_intrinsics(object_dir)
    camera = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=np.float64)
    confident = keypoints[..., 2] >= MIN_KEYPOINT_CONFIDENCE
    object_points = joints_w[confident]
    image_points = keypoints[..., :2][confident]
    rng = np.random.default_rng(0)
    if len(object_points) > 40000:
        pick = rng.choice(len(object_points), 40000, replace=False)
        object_points, image_points = object_points[pick], image_points[pick]

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera, None, iterationsCount=3000,
        reprojectionError=12.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 50:
        raise SystemExit(f"{tag}: PnP failed (inliers={0 if inliers is None else len(inliers)})")
    inliers = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers], image_points[inliers], camera, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec[:, 0]

    def reprojection(joints_c: np.ndarray) -> dict:
        error = np.linalg.norm(project(joints_c, k) - keypoints[..., :2], axis=-1)
        per_frame = np.array([np.median(e[c]) if c.any() else np.nan for e, c in zip(error, confident)])
        return {"median_px": float(np.nanmedian(error[confident])),
                "p90_px": float(np.nanpercentile(error[confident], 90)),
                "per_third_median_px": [float(np.nanmedian(t)) for t in np.array_split(per_frame, 3)]}

    joints_c = joints_w @ rotation.T + translation
    before = reprojection(joints_c)
    offsets = frame_offsets(joints_c, keypoints, confident, k)
    drift_camera = smooth_dense(track_frames, offsets, len(vertices_w))  # [T, 3], camera frame
    drift_world = drift_camera @ rotation  # row-wise R^T @ offset
    after = reprojection(joints_c + drift_camera[track_frames][:, None])

    # MoGe / SMPL depth ratio on the body's front surface.
    ratios = []
    height, width = int(k["height"]), int(k["width"])
    for frame in observed[:: args.depth_stride]:
        points_c = vertices_w[frame] @ rotation.T + translation + drift_camera[frame]
        if (points_c[:, 2] <= 0.1).any():
            continue
        uv = project(points_c, k)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv, z = uv[inside], points_c[inside, 2]
        cells = (uv[:, 1] // ZBUFFER_CELL).astype(np.int64) * (width // ZBUFFER_CELL + 1) + (uv[:, 0] // ZBUFFER_CELL).astype(np.int64)
        order = np.lexsort((z, cells))
        first = order[np.r_[True, cells[order][1:] != cells[order][:-1]]]
        depth = depth_m(object_dir / "depth" / f"{frame:06d}.png")
        observed_depth = depth[uv[first, 1].astype(int), uv[first, 0].astype(int)]
        ratio = observed_depth / z[first]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.2) & (ratio < 5.0)]
        if len(ratio) >= 20:
            ratios.append(float(np.median(ratio)))
    if not ratios:
        raise SystemExit(f"{tag}: no frame produced a usable MoGe/SMPL depth ratio")
    depth_scale = float(np.median(ratios))

    report = {
        "schema": "v2d.track1.wham_camera_alignment.v1",
        "episode": args.episode,
        "subject": str(subject),
        "R_cw": rotation.tolist(),
        "t_cw": translation.tolist(),
        "intrinsics": k,
        "depth_scale": depth_scale,
        "depth_scale_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
        "depth_scale_frames": len(ratios),
        "pnp": {
            "correspondences": int(len(object_points)),
            "global_inliers": int(len(inliers)),
            "camera_distance_to_mean_hip_m": float(np.linalg.norm(joints_c[:, 11:13].mean(axis=(0, 1)))),
            "reprojection_static": before,
            "reprojection_drift_corrected": after,
        },
        "drift": {
            "frames_with_offsets": int(np.isfinite(offsets).all(axis=1).sum()),
            "median_filter_frames": DRIFT_MEDIAN_FRAMES,
            "gaussian_sigma_frames": DRIFT_SIGMA_FRAMES,
            "max_norm_m": float(np.linalg.norm(drift_world, axis=1).max()),
            "end_minus_start_m": (drift_world[-1] - drift_world[0]).tolist(),
        },
        "frames": {"video": int(len(vertices_w)), "observed": int(len(observed)), "with_keypoints": int(len(track_frames))},
    }
    out = ROOT / "outputs" / "alignment" / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "drift.npz", drift_world=drift_world.astype(np.float64), drift_camera=drift_camera.astype(np.float64))
    (out / "alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("episode", "depth_scale", "depth_scale_iqr", "pnp", "drift")}, indent=2))
    print(f"PASS: {tag} WHAM-world to MoGe-camera alignment")


if __name__ == "__main__":
    main()
