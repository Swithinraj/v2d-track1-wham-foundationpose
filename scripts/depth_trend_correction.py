#!/usr/bin/env python3
"""Correct the slow depth drift of the human along the camera ray with MoGe-2's depth trend (wham.sif).

The scorer anchors every episode with one similarity transform at its first scored frame, so a constant
depth error costs nothing but a depth error that changes over time does. WHAM's trajectory drifts in depth;
MoGe-2's per-frame depth, normalised by the static background, does not.

  series EPISODE...   per frame (stride 3): r_t = median(MoGe depth / drift-corrected SMPL front-surface
                      depth) over the body's pixels; b_t = median(MoGe depth_t / per-pixel temporal median)
                      over static background pixels (no body, no object), which cancels MoGe's own
                      per-frame scale wobble; qn_t = (r_t / b_t) / median. -> outputs/depth_trend/<episode>.npz
  apply               qn is interpolated to every frame, median-filtered (15 frames) and Gaussian-smoothed
                      (sigma 30 frames) to keep only the trend; the human root moves along the camera ray by
                      alpha * (trend_t - 1) * centroid_t (body size unchanged). Objects are untouched.
                      Reads assembled episode files, writes a corrected copy + depth_trend_report.json.

Kaggle A/B against the same pipeline without this step (alpha 0.75): CD-H 23.14 -> 21.48 cm,
CD-O 31.23 -> 29.91 cm, PEN 0.0106 -> 0.0091, ACC-H 0.2209 -> 0.2208, ACC-O unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parents[1]
STRIDE, CELL = 3, 4
MEDIAN_FRAMES, SIGMA_FRAMES = 15, 30.0
FLIP = np.array([1.0, -1.0, -1.0])   # MHR root translation axes vs WHAM world (pose[:, :3] is in decimetres)


def depth_m(path: Path) -> np.ndarray:
    raw = np.array(Image.open(path)).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(raw > 0, 65535.0 / raw - 1.0, np.nan)


def series(episode: int) -> dict:
    tag = f"episode_{episode:06d}"
    z = np.load(ROOT / "outputs/wham" / tag / "wham_smpl_world.npz")
    V = z["vertices"].astype(np.float64)
    a = json.loads((ROOT / "outputs/alignment" / tag / "alignment.json").read_text())
    drift = np.load(ROOT / "outputs/alignment" / tag / "drift.npz")["drift_camera"]
    R, t, k = np.asarray(a["R_cw"]), np.asarray(a["t_cw"]), a["intrinsics"]
    H, W = int(k["height"]), int(k["width"])
    frames = np.arange(0, len(V), STRIDE)
    gy, gx = np.mgrid[8:H:16, 8:W:16]
    depth_stack, r, bg_masks, zroot = [], [], [], []
    for f in frames:
        pc = V[f] @ R.T + t + drift[f]
        uv = np.stack([k["fx"] * pc[:, 0] / pc[:, 2] + k["cx"], k["fy"] * pc[:, 1] / pc[:, 2] + k["cy"]], 1)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pc[:, 2] > 0.1)
        uvv, zz = uv[ok], pc[ok, 2]
        cells = (uvv[:, 1] // CELL).astype(int) * (W // CELL + 1) + (uvv[:, 0] // CELL).astype(int)
        order = np.lexsort((zz, cells))
        first = order[np.r_[True, cells[order][1:] != cells[order][:-1]]]   # front-most vertex per cell
        D = depth_m(ROOT / "outputs/object" / tag / "depth" / f"{f:06d}.png")
        ratio = D[uvv[first, 1].astype(int), uvv[first, 0].astype(int)] / zz[first]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.3) & (ratio < 3)]
        r.append(np.median(ratio) if len(ratio) > 30 else np.nan)
        zroot.append(np.median(zz))
        body = np.zeros((H, W), np.uint8)
        body[uvv[:, 1].astype(int), uvv[:, 0].astype(int)] = 1
        body = cv2.dilate(body, np.ones((61, 61), np.uint8)) > 0
        mask = np.array(Image.open(ROOT / "outputs/object" / tag / "sam2_masks/1" / f"{f:06d}.png")) > 127
        objm = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        bg_masks.append(~(body | objm)[gy, gx])
        depth_stack.append(D[gy, gx])
    Ds, Bm = np.stack(depth_stack), np.stack(bg_masks)
    ref = np.nanmedian(np.where(Bm, Ds, np.nan), axis=0)
    b = np.array([np.nanmedian((Ds[i] / ref)[Bm[i] & np.isfinite(ref)]) for i in range(len(frames))])
    r, zroot = np.array(r), np.array(zroot)
    q = r / b
    qn = q / np.nanmedian(q)
    centroid = np.stack([(V[f] @ R.T + t + drift[f]).mean(0) for f in frames])
    out = ROOT / "outputs/depth_trend"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", frames=frames, qn=qn, b=b, r=r, zroot=zroot, centroid_cam=centroid)
    err_cm = (qn - 1) * zroot * 100
    summary = {"episode": episode, "depth_err_median_abs_cm": float(np.nanmedian(np.abs(err_cm))),
               "trend_first_vs_last_third_cm": float(np.nanmedian(err_cm[-len(err_cm) // 3:]) - np.nanmedian(err_cm[:len(err_cm) // 3])),
               "moge_background_wobble_pct": float((np.nanpercentile(b, 90) - np.nanpercentile(b, 10)) / 2 * 100)}
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


def apply(input_dir: Path, out_dir: Path, alpha: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"alpha": alpha, "median_frames": MEDIAN_FRAMES, "sigma_frames": SIGMA_FRAMES, "episodes": {}}
    for path in sorted(input_dir.glob("episode_*.npz")):
        tag = path.stem
        data = dict(np.load(path))
        T = len(data["pose"])
        s = np.load(ROOT / "outputs/depth_trend" / f"{tag}.npz")
        ok = np.isfinite(s["qn"])
        q = np.interp(np.arange(T), s["frames"][ok], s["qn"][ok])
        q = median_filter(q, size=MEDIAN_FRAMES, mode="nearest")
        trend = gaussian_filter1d(q, SIGMA_FRAMES, mode="nearest")
        cent = np.stack([np.interp(np.arange(T), s["frames"][ok], s["centroid_cam"][ok][:, c]) for c in range(3)], 1)
        delta_cam = alpha * ((trend - 1.0)[:, None] * cent)
        R = np.asarray(json.loads((ROOT / "outputs/alignment" / tag / "alignment.json").read_text())["R_cw"])
        delta_world = delta_cam @ R          # row-wise R^T @ delta_cam
        if alpha:
            pose = data["pose"].astype(np.float64)
            pose[:, :3] += FLIP * delta_world * 10.0
            data["pose"] = pose.astype(np.float32)
        np.savez(out_dir / f"{tag}.npz", **data)
        for mesh in input_dir.glob(f"{tag}_object.*"):
            shutil.copy(mesh, out_dir / mesh.name)
        size = np.linalg.norm(delta_world, axis=1) * 100
        report["episodes"][tag] = {"correction_cm_median": float(np.median(size)), "correction_cm_max": float(size.max())}
    (out_dir / "depth_trend_report.json").write_text(json.dumps(report, indent=2))
    sizes = [v["correction_cm_median"] for v in report["episodes"].values()]
    print(f"depth trend: {len(sizes)} episodes, alpha {alpha}, correction median {np.median(sizes):.1f} cm -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("series")
    p.add_argument("episodes", type=int, nargs="+")
    p = sub.add_parser("apply")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.75)
    args = parser.parse_args()
    if args.cmd == "series":
        for episode in args.episodes:
            print(json.dumps(series(episode)), flush=True)
    else:
        apply(args.input, args.out, args.alpha)


if __name__ == "__main__":
    main()
