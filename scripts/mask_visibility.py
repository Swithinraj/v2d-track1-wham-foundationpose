#!/usr/bin/env python3
"""Per-frame, per-joint mask visibility of the actor's joints (v2d-host venv, CPU).

A joint counts as visible in frame t when its projection lies inside the actor's SAM 2 person mask (dilated by one cell
at 1/4 resolution) and outside the object's SAM 2 mask (outputs/object/<episode>/sam2_masks/1). This is the
mask-visibility idea of the multi-view GT optimiser (occluded keypoints get low weight), used here as a data-quality
weight that needs neither MediaPipe nor ground truth.
Writes <out>/<episode>.npz: vis [T, J] (0/1) for the joints of --bodies (mhr_bodies.py output, camera frame).
Usage: mask_visibility.py --bodies DIR --out DIR [--episodes ...]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); k = np.ones((3, 3), np.uint8)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        J = np.load(args.bodies / f"{tag}.npz")["joints"].astype(float)
        K = json.loads((ROOT / f"outputs/alignment_sam/{tag}/alignment.json").read_text())["intrinsics"]
        uv = np.stack([K["fx"] * J[..., 0] / J[..., 2] + K["cx"], K["fy"] * J[..., 1] / J[..., 2] + K["cy"]], -1)
        vis = np.zeros(uv.shape[:2])
        for f in range(len(uv)):
            P = cv2.imread(str(ROOT / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            O = cv2.imread(str(ROOT / f"outputs/object/{tag}/sam2_masks/1/{f:06d}.png"), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if P is None:
                continue
            P = cv2.dilate((P > 127).astype(np.uint8), k) > 0; O = (O > 127) if O is not None else np.zeros_like(P)
            with np.errstate(invalid="ignore"):
                u = np.clip(np.round(uv[f, :, 0] / 4).astype(int), 0, P.shape[1] - 1)
                v = np.clip(np.round(uv[f, :, 1] / 4).astype(int), 0, P.shape[0] - 1)
            vis[f] = np.isfinite(uv[f]).all(-1) & P[v, u] & ~O[v, u]
        np.savez(args.out / f"{tag}.npz", vis=vis)
        print(tag, "frames", len(uv), "| visible share (upper body)", round(float(vis[:, [75, 39, 76, 40, 78, 42, 2, 18]].mean()), 3), flush=True)


if __name__ == "__main__":
    main()
