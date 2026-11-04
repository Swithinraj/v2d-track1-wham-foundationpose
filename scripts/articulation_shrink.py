#!/usr/bin/env python3
"""Scale the human's articulation acceleration by k, keep the root's motion, then the per-frame MHR refit (GPU).

The same shrink on object translation lowered ACC-O (0.07051 -> 0.06971, k 0.6). On the human root it raised ACC-H
(0.13616 -> 0.13643): the root's acceleration is mostly real motion. This tests the remainder, the body relative to its
root: vertex targets X = V + (1 - k) [(V_s - V) - (r_s - r)], where V_s is the Whittaker smooth (lam 1e4) of every vertex
over the video and r the root translation, smoothed the same way. So the root keeps its track and the articulation's
acceleration is k times as large. Each frame's 136 pose parameters are then refitted (kit lm_pose); objects and identity
stay byte-identical. Writes <out>/episode_*.npz (+ meshes) and articulation_shrink_report.json.
Usage: articulation_shrink.py --input DIR --out DIR --k K [--lam 1e4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

FLIP = np.array([1.0, -1.0, -1.0])     # MHR root translation axes vs the Track 1 frame; pose[:, :3] is in decimetres


def targets(V, root_mm, k, smoother):
    """V [T, Nv, 3] mm, root_mm [T, 3]; smoother maps [T, D] -> smoothed [T, D]. Articulation acceleration scaled by k."""
    T = len(V)
    Vs = smoother(V.reshape(T, -1)).reshape(V.shape)
    rs = smoother(root_mm)
    return V + (1 - k) * ((Vs - V) - (rs - root_mm)[:, None])


def main():
    import torch
    from human_refit import SCORED_J, smoother
    from mesh_to_mhr_params import MHR, lm_pose
    from mhr_bodies import MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--lam", type=float, default=1e4)
    args = parser.parse_args()
    if not 0 <= args.k <= 1:
        raise SystemExit("k must lie in [0, 1]")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    args.out.mkdir(parents=True)
    for f in args.input.iterdir():
        if f.is_file() and not (f.name.startswith("episode_") and f.suffix == ".npz"):
            shutil.copy2(f, args.out / f.name)
    device = "cuda"
    mhr = MHR(str(MODEL), device, chunk=128, precision="float64")
    columns = list(SCORED_J)
    report = {}
    for path in sorted(args.input.glob("episode_??????.npz")):
        with np.load(path) as archive:
            data = dict(archive)
        T = len(data["pose"])
        identity = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None], dtype=torch.float64, device=device)
        pose0 = torch.as_tensor(data["pose"], dtype=torch.float64, device=device)
        V, J0 = mhr.run(pose0, identity)                                       # mm, m
        A = torch.linalg.inv(smoother(T, args.lam, device))
        smooth = lambda x: (A @ torch.as_tensor(x, dtype=torch.float64, device=device)).cpu().numpy()
        root = data["pose"][:, :3].astype(np.float64) * FLIP / 10 * 1000
        X = targets(V.cpu().numpy(), root, args.k, smooth)
        goal = torch.as_tensor(X.reshape(T, -1), dtype=torch.float64, device=device)
        pose1, error = lm_pose(mhr, goal, pose0.clone(), identity, iters=6)
        _, J1 = mhr.run(pose1, identity)
        data["pose"] = pose1.cpu().numpy().astype(data["pose"].dtype)
        if not np.isfinite(data["pose"]).all():
            raise SystemExit(f"{path.stem}: non-finite pose after the refit")
        np.savez(args.out / path.name, **data)
        acc = lambda J: float((torch.linalg.norm(J[:-2] - 2 * J[1:-1] + J[2:], dim=-1)[:, columns].mean() * 100).item())
        report[path.stem] = dict(fit_error_mm=float(error.mean()), joint_acc_cm=[acc(J0), acc(J1)],
                                 root_change_mm=float(np.abs(data["pose"][:, :3].astype(np.float64) * FLIP / 10 * 1000 - root).max()))
        print(path.stem, json.dumps(report[path.stem]), flush=True)
    (args.out / "articulation_shrink_report.json").write_text(json.dumps(dict(k=args.k, lam=args.lam, episodes=report), indent=1) + "\n")
    r = list(report.values())
    print(f"PASS: k {args.k}: joint acc {np.mean([x['joint_acc_cm'][0] for x in r]):.4f} -> {np.mean([x['joint_acc_cm'][1] for x in r]):.4f} "
          f"cm/frame^2, fit err {np.mean([x['fit_error_mm'] for x in r]):.2f} mm -> {args.out}")


if __name__ == "__main__":
    import os
    import sys
    root = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
    sys.path.insert(0, str(root / "official/v2d_submission_kit/tools/track1"))
    main()
