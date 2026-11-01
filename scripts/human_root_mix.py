#!/usr/bin/env python3
"""Mix the person's trajectory with an independently built estimate (host, CPU). No learned parameters.

Candidates built on SAM 3D Body differ by about 1.6 cm in body position, so they share most of their error and averaging
them gains little. WHAM-based estimates differ from them by 6-8 cm. Moving part of the way toward such an estimate can
only help if its error is largely independent, which only Kaggle can tell, so the weight is an explicit experiment.

Only the root translation changes, so articulation stays the base's. Both trajectories are compared after the same
similarity fit at the first scored frame the scorer uses. The first scored frame stays exactly in place, so the alignment
and CD-O do not change. The difference is smoothed, so the other estimate's jitter adds no acceleration.

Usage: human_root_mix.py --base DIR --base_meshes DIR --other_meshes DIR --out DIR --weight W [--lam 3000]
  meshes: ensemble_meshes.py output [scored frames, V, 3]. Writes OUT/episode_*.npz (+ other files) and root_mix_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from floor_root import FLIP, whittaker


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


def mix_shift(base, other, weight, *, lam=3000.):
    """Displacement [F,3] of the base over its frames [F,V,3]: toward other by weight, zero at frame 0."""
    base, other = np.asarray(base, float), np.asarray(other, float)
    if base.shape != other.shape or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base and other must be matching [F,V,3] arrays")
    if not np.isfinite(base).all() or not np.isfinite(other).all():
        raise ValueError("trajectories must be finite")
    scale, rotation, translation = similarity(other[0], base[0])
    aligned = scale * (other @ rotation) + translation
    difference = (aligned - base).mean(1)
    smooth = whittaker(difference, np.ones(len(difference)), lam)
    shift = weight * (smooth - smooth[0])
    shift[0] = 0.
    return shift


def full_shift(shift, frames, T):
    """[len(frames),3] over scored frames -> [T,3] over video frames, linear between them and held beyond."""
    return np.stack([np.interp(np.arange(T), frames, shift[:, k]) for k in range(3)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base_meshes", type=Path, required=True)
    parser.add_argument("--other_meshes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--lam", type=float, default=3000.)
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    scored = json.loads(args.scored.read_text())
    shutil.copytree(args.base, args.out)
    report = {}
    for path in sorted(args.base.glob("episode_??????.npz")):
        frames = np.asarray(scored[str(int(path.stem.split("_")[1]))])
        shift = mix_shift(np.load(args.base_meshes / f"{path.stem}.npy"), np.load(args.other_meshes / f"{path.stem}.npy"),
                          args.weight, lam=args.lam)
        with np.load(path) as archive:
            data = dict(archive)
        pose = data["pose"].astype(np.float64)
        pose[:, :3] += FLIP * full_shift(shift, frames, len(pose)) * 10.0
        data["pose"] = pose.astype(data["pose"].dtype)
        np.savez(args.out / path.name, **data)
        report[path.stem] = dict(shift_rms_m=float(np.sqrt((shift ** 2).sum(1).mean())), shift_max_m=float(np.linalg.norm(shift, axis=1).max()))
    (args.out / "root_mix_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PASS: root mix weight {args.weight}, median rms {100 * np.median([r['shift_rms_m'] for r in report.values()]):.1f} cm -> {args.out}")


if __name__ == "__main__":
    main()
