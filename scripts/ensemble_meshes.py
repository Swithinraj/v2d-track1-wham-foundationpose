#!/usr/bin/env python3
"""MHR vertices of the scored frames of one episode directory (mhr_converter.sif, GPU), for comparing human estimates.

Every EVERY-th vertex of the 18439 is kept, which is plenty for trajectories and much smaller. Comparing estimates needs
no ground truth: with at least three estimators with independent errors, pairwise differences reveal each one's error.
Usage: ensemble_meshes.py --episodes_dir DIR --out DIR [--every 8]
Writes OUT/episode_*.npy: float32 metres [scored frames, vertices, 3].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"
SCORED = ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scored = json.loads(SCORED.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), device, chunk=128, precision="float64")
    for path in sorted(args.episodes_dir.glob("episode_??????.npz")):
        frames = np.asarray(scored[str(int(path.stem.split("_")[1]))])
        with np.load(path) as archive:
            pose = archive["pose"][frames]
            identity = np.concatenate([archive["scales"], archive["shape"]])
        identity = torch.as_tensor(identity[None], dtype=torch.float64, device=device)
        kept = []
        for start in range(0, len(pose), 128):
            vertices, _ = mhr.run(torch.as_tensor(pose[start:start + 128], dtype=torch.float64, device=device), identity)
            kept.append((vertices[:, ::args.every] / 1000.0).cpu().numpy().astype(np.float32))
        np.save(args.out / f"{path.stem}.npy", np.concatenate(kept))
    print(f"PASS: {args.episodes_dir} -> {args.out}")


if __name__ == "__main__":
    main()
