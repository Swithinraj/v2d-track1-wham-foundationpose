#!/usr/bin/env python3
"""Physics for the object track: rest and gravity (numpy + PIL + trimesh python, CPU).

A monocular tracker is worst along the camera ray; physics pins exactly that direction:
  REST     with a static camera an object can only move without changing its image along the camera ray. In our
           FoundationPose tracks 45-98 % of the motion during frames whose object mask is unchanged is along the depth
           axis (0.3-10 cm drift per still period, up to 25 cm). Rest periods: the object mask (SAM 2) is unchanged frame
           to frame (IoU > IOU_STEP at 1/4 resolution) and has not drifted (IoU with 5 frames earlier > IOU_DRIFT),
           for >= MIN_RUN frames (a hand passing over the object changes the mask, so occluded frames are never rest).
           During a rest period the translation is held constant (hard zero-velocity constraint) and the rotation is
           the period's medoid tracked rotation (never an average: averaging across a symmetry flip of a round object
           gives a pose no frame had). There the true acceleration is zero, so ACC-O's error there becomes zero.
  GRAVITY  nothing floats and nothing sinks: an object whose lowest point is near the floor (|height| < FLOOR_NEAR,
           fading out by FLOOR_FAR) is on the floor. It is moved along its camera ray (the image position stays)
           until its lowest point touches the floor plane (MoGe background RANSAC plane, --floor). Lifted objects are
           left alone.
  OCCLUSION (--occlusion, a Kalman/RTS-equivalent smoother): each frame's tracked position is a measurement whose
           weight is how much of the object the camera actually sees: visible = |SAM 2 mask n projected object| /
           |projected object| (the scorer-budgeted mesh at the tracked pose, 1/4 resolution); weight = visible^2
           (>= W_MIN). Where a hand hides the object the smoother leans on the frames around it (prediction) instead
           of the tracker's occluded estimate (correction). Unoccluded frames keep the plain LAM smoothing.
The translation is re-solved by weighted least squares (data = gravity-corrected track, second-difference smoothing
LAM, the rest constraint), so corrections blend into the moving parts. Rotations outside rest periods are untouched
(ACC-O uses the translation only). Only object_translation / object_rotation change.
Writes <out>/episode_*.npz (+ meshes, shared files) and physics_objects_report.json: rest and floor shares, the
object's acceleration on rest frames (= its exact ACC-O error there) and on moving frames, before and after.
Usage: physics_objects.py --input DIR --out DIR --floor FILE [--masks DIR] [--lam 10] [--no_rest] [--no_gravity]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
IOU_STEP, IOU_DRIFT, MIN_RUN, MAX_GAP, HOLD, LAM_DRIFT = 0.97, 0.93, 10, 60, 1e6, 1e4
FLOOR_NEAR_M, FLOOR_FAR_M, MAX_SNAP_M, LAM_GRAVITY = 0.06, 0.12, 0.25, 1e4
W_MIN = 0.05
ARGS = None


def rest_clusters(masks_dir: Path, T: int) -> np.ndarray:
    """Cluster label per frame (-1 = moving). A cluster is one resting placement: still runs (>= MIN_RUN frames) whose
    masks match across the gap between them (IoU > IOU_DRIFT, gap <= MAX_GAP frames) are one cluster, gap included
    (an object that reappears where it was did not move while it was occluded or its mask flickered)."""
    m = [np.array(Image.open(masks_dir / f"{f:06d}.png"))[::4, ::4] > 127 for f in range(T)]

    def iou(a, b):
        u = (a | b).sum()
        return (a & b).sum() / u if u else 1.0

    still = np.array([f >= 5 and iou(m[f], m[f - 1]) > IOU_STEP and iou(m[f], m[f - 5]) > IOU_DRIFT for f in range(T)])
    spans = [(max(a - 1, 0), b) for a, b in runs(still) if b - a >= MIN_RUN]   # frame a-1 starts the unchanged pair
    labels = np.full(T, -1)
    k = -1
    for i, (a, b) in enumerate(spans):
        if i and a - spans[i - 1][1] <= MAX_GAP and iou(m[spans[i - 1][1] - 1], m[a]) > IOU_DRIFT:
            labels[spans[i - 1][1]:b] = k          # same placement: the gap joins the cluster
        else:
            k += 1
            labels[a:b] = k
    return labels


def runs(mask: np.ndarray):
    out, f = [], 0
    while f < len(mask):
        if mask[f]:
            g = f
            while g < len(mask) and mask[g]:
                g += 1
            out.append((f, g))
            f = g
        else:
            f += 1
    return out


def solve(y: np.ndarray, labels: np.ndarray, lam: float, w: np.ndarray | None = None) -> np.ndarray:
    T = len(y)
    D2 = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    w = np.ones(T) if w is None else w
    A = np.diag(w) + lam * D2.T @ D2
    pairs = np.flatnonzero((labels[1:] == labels[:-1]) & (labels[1:] >= 0))
    if pairs.size:
        D1 = np.zeros((len(pairs), T))
        D1[np.arange(len(pairs)), pairs], D1[np.arange(len(pairs)), pairs + 1] = -1.0, 1.0
        A += HOLD * D1.T @ D1
    return np.linalg.solve(A, (w * y.T).T)


def drift_correction(x: np.ndarray, labels: np.ndarray, lam: float = LAM_DRIFT) -> np.ndarray:
    """Smooth correction D (shape of x): x + D is constant over every rest cluster, D has small second differences
    (LAM_DRIFT, a ~10-frame blend) and decays to zero away from rest periods. The tracker's error drifts slowly, so the
    moving frames next to a rest period carry the same drift: correcting the drift (not the frames) keeps the motion's
    own shape and adds no jolt where the object starts or stops."""
    T = len(x)
    D2 = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    A = np.eye(T) + lam * D2.T @ D2
    pairs = np.flatnonzero((labels[1:] == labels[:-1]) & (labels[1:] >= 0))
    if not pairs.size:
        return np.zeros_like(x)
    D1 = np.zeros((len(pairs), T))
    D1[np.arange(len(pairs)), pairs], D1[np.arange(len(pairs)), pairs + 1] = -1.0, 1.0
    A += HOLD * D1.T @ D1
    return np.linalg.solve(A, -HOLD * D1.T @ (D1 @ x))


def medoid_rotation(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.einsum("aij,bij->ab", R, R) - 1) / 2, -1, 1)
    return R[np.arccos(cos).sum(1).argmin()]


def visible_fraction(d, tag, T):
    """Per frame |mask n projected object| / |projected object| at 1/4 resolution (1 = fully seen)."""
    K = json.loads((ROOT / "outputs/camera" / tag / "intrinsics.json").read_text())
    W, H = int(K["width"]) // 4, int(K["height"]) // 4
    Km = np.array([[K["fx"] / 4, 0, K["cx"] / 4], [0, K["fy"] / 4, K["cy"] / 4], [0, 0, 1]])
    b = np.load(ARGS.meshes / f"{tag}_budget.npz")
    V, F = b["vertices"].astype(float), b["faces"]
    R, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(np.asarray(d["object_scale"]).reshape(()))
    vis = np.ones(T)
    for f in range(T):
        X = s * V @ R[f].T + t[f]
        if (X[:, 2] <= 0.05).any():
            continue
        uv = (X @ Km.T)[:, :2] / X[:, 2:3]
        proj = np.zeros((H, W), np.uint8)
        cv2.fillPoly(proj, [np.round(uv[tri]).astype(np.int32) for tri in F], 1)
        proj = proj.astype(bool)
        if proj.sum() < 20:
            continue
        m = np.array(Image.open(ARGS.masks / tag / "sam2_masks/1" / f"{f:06d}.png"))[::4, ::4] > 127
        vis[f] = (proj & m).sum() / proj.sum()
    return vis


def episode(path: Path) -> dict:
    tag = path.stem
    d = dict(np.load(path))
    t, R = d["object_translation"].astype(np.float64), d["object_rotation"].astype(np.float64)
    s_obj = float(np.asarray(d["object_scale"]).reshape(()))
    T = len(t)
    labels = np.full(T, -1) if ARGS.no_rest else rest_clusters(ARGS.masks / tag / "sam2_masks/1", T)
    rest = labels >= 0
    R1 = R.copy()
    for k in range(labels.max() + 1):
        R1[labels == k] = medoid_rotation(R[labels == k])

    # gravity: lowest point on the floor plane n.X = c / depth_scale (MoGe metres -> this frame), as a slowly varying
    # offset along the camera ray (per-frame snapping would turn rotation wobble into depth jitter)
    y, w, height = t.copy(), np.zeros(T), np.zeros(T)
    if not ARGS.no_gravity:
        fl = ARGS.floor_planes[tag]
        n = np.asarray(fl["normal"], float)
        c = fl["c"] / json.loads((ARGS.alignment / tag / "alignment.json").read_text())["depth_scale"]
        V = np.asarray(trimesh.load(next(path.parent.glob(f"{tag}_object*")), force="mesh").convex_hull.vertices, float)
        low = (s_obj * np.einsum("tij,vj->tvi", R1, V) @ n).min(axis=1)
        height = low + t @ n - c
        w = np.clip((FLOOR_FAR_M - np.abs(height)) / (FLOOR_FAR_M - FLOOR_NEAR_M), 0.0, 1.0)
        w = np.array([np.median(w[max(0, f - 7):f + 8]) for f in range(T)])
        ray = t / np.linalg.norm(t, axis=1, keepdims=True)
        nr = ray @ n
        step = np.where(np.abs(nr) > 0.05, -height / np.where(np.abs(nr) > 0.05, nr, 1.0), 0.0)
        step = np.clip(step, -MAX_SNAP_M, MAX_SNAP_M)
        if (w > 0).any():
            g = solve(step, np.full(T, -1), LAM_GRAVITY, w + 1e-6)
            fade = solve(w, np.full(T, -1), 100.0)
            y = t + (np.clip(fade, 0.0, 1.0) * g)[:, None] * ray

    t1 = y + drift_correction(y, labels)
    vis = np.ones(T)
    if ARGS.occlusion:
        vis = visible_fraction(d, tag, T)
        t1 = solve(t1, labels, ARGS.lam, np.clip(vis, W_MIN ** 0.5, 1.0) ** 2)
    elif ARGS.lam > 0:
        t1 = solve(t1, labels, ARGS.lam)
    d["object_translation"], d["object_rotation"] = t1.astype(np.float32), R1.astype(np.float32)
    np.savez(ARGS.out / f"{tag}.npz", **d)
    for extra in path.parent.glob(f"{tag}_object*"):
        shutil.copy2(extra, ARGS.out / extra.name)

    def acc(x, sel):
        a = np.r_[0.0, np.linalg.norm(np.diff(x, n=2, axis=0), axis=1), 0.0]
        return float(a[sel].mean() * 100) if sel.any() else 0.0

    inner = np.zeros(T, bool)
    inner[1:-1] = True
    return {"episode": tag, "rest_share": float(rest.mean()), "floor_share": float((w > 0.5).mean()),
            "occluded_share": float((vis < 0.8).mean()), "visible_mean": float(vis.mean()),
            "floor_gap_cm_median": float(np.median(np.abs(height[w > 0.5])) * 100) if (w > 0.5).any() else 0.0,
            "acc_rest_cm": [acc(t, rest & inner), acc(t1, rest & inner)],
            "acc_moving_cm": [acc(t, ~rest & inner), acc(t1, ~rest & inner)],
            "acc_all_cm": [acc(t, inner), acc(t1, inner)],
            "change_mean_cm": float(np.linalg.norm(t1 - t, axis=1).mean() * 100),
            "change_max_cm": float(np.linalg.norm(t1 - t, axis=1).max() * 100)}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True, help="JSON {episode: {normal, c}}: floor plane n.X = c, MoGe metres")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs/alignment_sam")
    ap.add_argument("--masks", type=Path, default=ROOT / "outputs/object")
    ap.add_argument("--lam", type=float, default=0.0, help="extra smoothing (0 = none)")
    ap.add_argument("--no_rest", action="store_true")
    ap.add_argument("--no_gravity", action="store_true")
    ap.add_argument("--occlusion", action="store_true", help="occlusion-weighted smoothing (needs --meshes, --lam)")
    ap.add_argument("--meshes", type=Path, default=ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes")
    ap.add_argument("--workers", type=int, default=8)
    ARGS = ap.parse_args()
    ARGS.floor_planes = json.loads(ARGS.floor.read_text())
    ARGS.out.mkdir(parents=True, exist_ok=True)
    for extra in ARGS.input.iterdir():
        if extra.is_file() and not extra.name.startswith("episode_"):
            shutil.copy2(extra, ARGS.out / extra.name)
    with ProcessPoolExecutor(ARGS.workers) as pool:
        report = list(pool.map(episode, sorted(ARGS.input.glob("episode_??????.npz"))))
    for r in report:
        print(f"{r['episode']}: rest {r['rest_share']:.2f} floor {r['floor_share']:.2f} (gap {r['floor_gap_cm_median']:.1f} cm) | "
              f"acc rest {r['acc_rest_cm'][0]:.4f} -> {r['acc_rest_cm'][1]:.4f}, moving {r['acc_moving_cm'][0]:.4f} -> "
              f"{r['acc_moving_cm'][1]:.4f} | change mean {r['change_mean_cm']:.2f} / max {r['change_max_cm']:.2f} cm", flush=True)
    mean = {k: [float(np.mean([r[k][i] for r in report])) for i in (0, 1)] for k in ("acc_rest_cm", "acc_moving_cm", "acc_all_cm")}
    mean.update({k: float(np.mean([r[k] for r in report])) for k in ("rest_share", "floor_share", "change_mean_cm", "occluded_share", "visible_mean")})
    (ARGS.out / "physics_objects_report.json").write_text(json.dumps(
        {"lam": ARGS.lam, "rest": not ARGS.no_rest, "gravity": not ARGS.no_gravity, "mean": mean, "episodes": report}, indent=1))
    print(json.dumps(mean))


if __name__ == "__main__":
    main()
