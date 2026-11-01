#!/usr/bin/env python3
"""Average differently built human estimates on the mesh, then refit the pose (mhr_converter.sif, GPU).

Every member is first placed by the same similarity fit at the first scored frame that the scorer uses, so a difference in
overall placement or scale does not count as motion. The vertices are then averaged with fixed weights and each frame's pose
is refitted to the mean with the base member's identity (kit lm_pose). Euler parameters wrap, so they are never averaged.

Why this cannot be much worse: a weighted mean's distance to any target is at most the weighted mean of the members'
distances, and a second difference is linear in position, so the same bound holds for joint acceleration error. The gain
depends on how independent the members' errors are. Candidates built on SAM 3D Body share most of theirs
(scripts/analysis/hat.py), so a few tenths of CD-H is the realistic size.

Usage: human_ensemble.py --out DIR --member NAME=DIR:WEIGHT [--member ...]    (the first member is the base)
Writes OUT/episode_*.npz (+ the base's other files; objects from the base) and human_ensemble_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def similarity(source, target):
    """target = s * source @ R + t, without reflection; the scorer's fit."""
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = -1. if np.linalg.det(u @ vt) < 0 else 1.
    rotation = (u * sign) @ vt
    scale = (singular * sign).sum() / (x * x).sum()
    return scale, rotation, b - scale * (a @ rotation)


def aligned_mean(members, weights):
    """Weighted mean of [F,V,3] vertex arrays, each aligned to the first member at frame 0."""
    weights = np.asarray(weights, float)
    if len(members) != len(weights) or len(members) == 0 or not np.isclose(weights.sum(), 1.) or (weights < 0).any():
        raise ValueError("weights must be non-negative, sum to 1 and match the members")
    shape = np.shape(members[0])
    if len(shape) != 3 or shape[2] != 3 or any(np.shape(m) != shape for m in members):
        raise ValueError("members must be matching [F,V,3] arrays")
    base = np.asarray(members[0], float)
    out = weights[0] * base
    for member, weight in zip(members[1:], weights[1:]):
        member = np.asarray(member, float)
        scale, rotation, translation = similarity(member[0], base[0])
        out = out + weight * (scale * (member @ rotation) + translation)
    return out


def main():
    import torch
    sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
    from mesh_to_mhr_params import MHR, lm_pose

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--member", action="append", required=True, help="NAME=DIR:WEIGHT")
    parser.add_argument("--scored", type=Path, default=ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json")
    parser.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = parser.parse_args()
    members = []
    for item in args.member:
        name, rest = item.split("=", 1)
        path, weight = rest.rsplit(":", 1)
        members.append((name, Path(path), float(weight)))
    args.out.mkdir(parents=True, exist_ok=True)
    for f in members[0][1].iterdir():
        if f.is_file() and not (f.name.startswith("episode_") and f.suffix == ".npz"):
            shutil.copy2(f, args.out / f.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(ROOT / "weights/mhr/assets/mhr_model.pt"), device, chunk=128, precision="float64")
    scored = json.loads(args.scored.read_text())
    weights = [m[2] for m in members]
    report = {}
    for episode in args.episodes:
        tag = f"episode_{episode:06d}"
        frames = np.asarray(scored[str(episode)])
        vertices = []
        for _, path, _ in members:
            with np.load(path / f"{tag}.npz") as archive:
                pose = torch.as_tensor(archive["pose"][frames], dtype=torch.float64, device=device)
                identity = torch.as_tensor(np.concatenate([archive["scales"], archive["shape"]])[None], dtype=torch.float64, device=device)
            vertices.append(mhr.run(pose, identity)[0].cpu().numpy())
        target = aligned_mean(vertices, weights)
        with np.load(members[0][1] / f"{tag}.npz") as archive:
            data = dict(archive)
        identity = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None], dtype=torch.float64, device=device)
        start = torch.as_tensor(data["pose"][frames], dtype=torch.float64, device=device)
        goal = torch.as_tensor(target.reshape(len(frames), -1), dtype=torch.float64, device=device)
        fitted, error = lm_pose(mhr, goal, start, identity, iters=8)
        fitted = fitted.cpu().numpy()
        # Unscored frames are bridged between scored ones. Euler columns stay continuous across a small correction.
        delta = fitted - data["pose"][frames].astype(np.float64)
        pose = data["pose"].astype(np.float64)
        for column in range(pose.shape[1]):
            pose[:, column] += np.interp(np.arange(len(pose)), frames, delta[:, column])
        pose[frames] = fitted
        data["pose"] = pose.astype(data["pose"].dtype)
        np.savez(args.out / f"{tag}.npz", **data)
        report[tag] = dict(fit_error_mm=float(error.mean()), fit_error_max_mm=float(error.max()),
                           move_mean_cm=float(np.linalg.norm(target - vertices[0], axis=-1).mean() / 10))
        print(tag, json.dumps(report[tag]), flush=True)
    (args.out / f"human_ensemble_report_{'_'.join(map(str, args.episodes))}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PASS: ensemble of {[m[0] for m in members]} -> {args.out}")


if __name__ == "__main__":
    main()
