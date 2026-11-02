#!/usr/bin/env python3
"""Hold planted feet still, then the usual per-frame MHR refit (mhr_converter.sif, GPU). No learned parameters.

Per joint, P2's acceleration is highest at the feet and ankles (0.075-0.081 cm/frame^2, as high as the wrists), although
the actors stand at a table in the scored frames: the table hides their legs and per-frame estimates guess them. A
planted foot does not move, so its true acceleration is close to zero and ours is error. ACC-H scores 22 body joints,
four of them ankles and feet.

Per foot (the vertices skinned to the ankle subtree), a run of at least RUN frames whose smoothed (lam 100, about 1.5 Hz)
centre speed stays under STILL is a stance. There the foot's vertex targets become their run mean, blended in and out
over RAMP frames. No floor is used, so a wrong floor estimate cannot move a foot. Every other vertex keeps its position,
and each frame's 136 pose parameters are refitted to the targets (kit lm_pose), so knees and hips absorb the pelvis
motion. Objects and identity stay byte-identical. Writes <out>/episode_*.npz (+ meshes) and feet_hold_report.json.
Usage: feet_hold.py --input DIR --out DIR [--still 4] [--run 10] [--ramp 4] [--lam 100]   (STILL in mm per frame)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def smooth(values, lam):
    """argmin |x - y|^2 + lam |D2 x|^2 along axis 0 (dense; episodes are under 1000 frames)."""
    T = len(values)
    if T < 3 or lam <= 0:
        return np.array(values, float)
    D = np.diff(np.eye(T), 2, axis=0)
    flat = np.asarray(values, float).reshape(T, -1)
    return np.linalg.solve(np.eye(T) + lam * D.T @ D, flat).reshape(np.shape(values))


def stance_runs(centre, still, run, lam):
    """[(start, stop)] runs of >= run frames whose smoothed speed (units of centre per frame) is below still."""
    speed = np.r_[0., np.linalg.norm(np.diff(smooth(centre, lam), axis=0), axis=1)]
    slow = speed < still
    runs, f, T = [], 0, len(slow)
    while f < T:
        if not slow[f]:
            f += 1
            continue
        g = f
        while g < T and slow[g]:
            g += 1
        if g - f >= run:
            runs.append((f, g))
        f = g
    return runs


def hold_targets(foot, still, run, ramp, lam):
    """foot [T, Nv, 3] -> (targets, held frames): stance runs pinned to their mean, cosine ramps at both ends."""
    targets = np.array(foot, float)
    held = 0
    for f, g in stance_runs(targets.mean(1), still, run, lam):
        mean = targets[f:g].mean(0)
        m = np.arange(g - f)
        beta = np.clip(np.minimum(m + 1, g - f - m) / ramp, 0, 1)
        beta = 0.5 - 0.5 * np.cos(np.pi * beta)
        targets[f:g] = (1 - beta[:, None, None]) * targets[f:g] + beta[:, None, None] * mean[None]
        held += g - f
    return targets, held


def main():
    import torch
    from human_refit import FOOT_JOINTS, SCORED_J, vertex_parts
    from mesh_to_mhr_params import MHR, lm_pose
    from mhr_bodies import MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--still", type=float, default=4.0)
    parser.add_argument("--run", type=int, default=10)
    parser.add_argument("--ramp", type=int, default=4)
    parser.add_argument("--lam", type=float, default=100.)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    args.out.mkdir(parents=True)
    for f in args.input.iterdir():
        if f.is_file() and not (f.name.startswith("episode_") and f.suffix == ".npz"):
            shutil.copy2(f, args.out / f.name)
    device = "cuda"
    mhr = MHR(str(MODEL), device, chunk=128, precision="float64")
    parts = vertex_parts(MODEL, np.array(FOOT_JOINTS))
    feet = [np.flatnonzero(parts == k) for k in range(len(FOOT_JOINTS))]
    columns = list(SCORED_J)
    report = {}
    for path in sorted(args.input.glob("episode_??????.npz")):
        with np.load(path) as archive:
            data = dict(archive)
        identity = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None], dtype=torch.float64, device=device)
        pose0 = torch.as_tensor(data["pose"], dtype=torch.float64, device=device)
        V, J0 = mhr.run(pose0, identity)                                       # mm, m
        X = V.cpu().numpy().copy()
        held = []
        for idx in feet:
            X[:, idx], frames = hold_targets(X[:, idx], args.still, args.run, args.ramp, args.lam)
            held.append(frames)
        goal = torch.as_tensor(X.reshape(len(X), -1), dtype=torch.float64, device=device)
        pose1, error = lm_pose(mhr, goal, pose0.clone(), identity, iters=6)
        _, J1 = mhr.run(pose1, identity)
        data["pose"] = pose1.cpu().numpy().astype(data["pose"].dtype)
        if not np.isfinite(data["pose"]).all():
            raise SystemExit(f"{path.stem}: non-finite pose after the refit")
        np.savez(args.out / path.name, **data)
        acc = lambda J, cols: float((torch.linalg.norm(J[:-2] - 2 * J[1:-1] + J[2:], dim=-1)[:, cols].mean() * 100).item())
        report[path.stem] = dict(held_frames=held, frames=len(X), fit_error_mm=float(error.mean()),
                                 joint_acc_cm=[acc(J0, columns), acc(J1, columns)],
                                 feet_acc_cm=[acc(J0, [4, 20, 8, 24]), acc(J1, [4, 20, 8, 24])])
        print(path.stem, json.dumps(report[path.stem]), flush=True)
    (args.out / "feet_hold_report.json").write_text(json.dumps(dict(still=args.still, run=args.run, ramp=args.ramp, lam=args.lam,
                                                                    episodes=report), indent=1) + "\n")
    r = list(report.values())
    print(f"PASS: feet held on {np.mean([sum(x['held_frames']) / (2 * x['frames']) for x in r]):.0%} of foot-frames; "
          f"joint acc {np.mean([x['joint_acc_cm'][0] for x in r]):.4f} -> {np.mean([x['joint_acc_cm'][1] for x in r]):.4f}, "
          f"feet {np.mean([x['feet_acc_cm'][0] for x in r]):.4f} -> {np.mean([x['feet_acc_cm'][1] for x in r]):.4f} cm/frame^2 -> {args.out}")


if __name__ == "__main__":
    import os
    import sys
    root = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
    sys.path.insert(0, str(root / "official/v2d_submission_kit/tools/track1"))
    main()
