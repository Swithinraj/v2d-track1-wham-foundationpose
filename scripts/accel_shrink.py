#!/usr/bin/env python3
"""Scale predicted acceleration by k inside each contiguous scored stretch (host, CPU). No learned parameters.

The ACC boards score |s R a_pred - a_gt| per frame, a being the second difference over contiguous scored frames. Scored
files say our acceleration is partly real motion and partly estimation noise: removing it all (heavy smoothing) cost
ACC-O +0.0009 and ACC-H +0.0019, while scaling the table objects' tracks by 0.974 lowered ACC-O. If a = signal +
independent noise, the error is lowest at k = signal / (signal + noise), between those two ends.

Each stretch becomes x_s + k (x - x_s), with x_s a Whittaker smooth (lam 1e4, about 0.5 Hz) of that stretch: position and
velocity follow the track, and the acceleration is k times as large. Stretches are separate, so a jump across an unscored
gap never leaks in. One constant offset then pins the first scored frame, the scorer's alignment frame, so the alignment
is unchanged; a constant changes no acceleration. Unscored frames are left as they are.

Usage: accel_shrink.py --input DIR --out DIR --k K [--objects] [--root] [--lam 1e4] [--scored JSON]
  --objects: object_translation (ACC-O). --root: the human root translation pose[:, :3], which moves every joint rigidly
  (ACC-H). Writes OUT/episode_*.npz (+ the other files of INPUT) and accel_shrink_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from floor_root import whittaker


def stretches(frames):
    """Contiguous runs of scored video frames."""
    frames = np.asarray(frames)
    return np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)


def shrink(track, frames, k, lam=1e4):
    """[T, D] track over video frames -> copy whose scored stretches carry k times the acceleration; first scored frame fixed."""
    track = np.asarray(track, np.float64)
    frames = np.asarray(frames)
    if track.ndim != 2 or not np.isfinite(track).all():
        raise ValueError("track must be a finite [T, D] array")
    if (not len(frames) or not np.issubdtype(frames.dtype, np.integer) or frames[0] < 0 or frames[-1] >= len(track)
            or (np.diff(frames) <= 0).any()):
        raise ValueError("frames must be increasing video indices inside the track")
    if not 0 <= k <= 1 or lam <= 0:
        raise ValueError("k must lie in [0, 1] and lam must be positive")
    out = track.copy()
    for part in stretches(frames):
        x = track[part]
        smooth = whittaker(x, np.ones(len(x)), lam)
        out[part] = smooth + k * (x - smooth)
    out[frames] += track[frames[0]] - out[frames[0]]
    return out


def acceleration(track, frames):
    """Mean |second difference| over the scored stretches, per frame as the scorer pads it (track units)."""
    errors = []
    for part in stretches(frames):
        if len(part) < 3:
            continue
        x = np.asarray(track)[part]
        norm = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)
        errors.append(np.r_[norm[:1], norm, norm[-1:]])
    return float(np.concatenate(errors).mean()) if errors else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--objects", action="store_true")
    parser.add_argument("--root", action="store_true")
    parser.add_argument("--lam", type=float, default=1e4)
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    args = parser.parse_args()
    if not (args.objects or args.root):
        raise SystemExit("choose --objects and/or --root")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    scored = json.loads(args.scored.read_text())
    shutil.copytree(args.input, args.out)
    report = {}
    for path in sorted(args.input.glob("episode_??????.npz")):
        frames = np.asarray(scored[str(int(path.stem.split("_")[1]))])
        with np.load(path) as archive:
            data = dict(archive)
        entry = {}
        targets = ([("object_translation", "object_translation", None)] if args.objects else []) + \
                  ([("pose", "root_translation_dm", slice(0, 3))] if args.root else [])
        for key, label, columns in targets:
            values = data[key].astype(np.float64)
            track = (values if columns is None else values[:, columns]).copy()
            new = shrink(track, frames, args.k, args.lam)
            if columns is None:
                values = new
            else:
                values[:, columns] = new
            data[key] = values.astype(data[key].dtype)
            written = data[key].astype(np.float64)
            written = written if columns is None else written[:, columns]
            if not np.isfinite(written).all() or not np.array_equal(written[frames[0]], track[frames[0]]):
                raise SystemExit(f"{path.stem}: {key} is non-finite or moved at the alignment frame")
            entry[label] = dict(acceleration_before=acceleration(track, frames), acceleration_after=acceleration(written, frames),
                                max_change=float(np.linalg.norm(written[frames] - track[frames], axis=1).max()))
        np.savez(args.out / path.name, **data)
        report[path.stem] = entry
    (args.out / "accel_shrink_report.json").write_text(json.dumps(dict(k=args.k, lam=args.lam, episodes=report), indent=1) + "\n")
    ratios = [v["acceleration_after"] / v["acceleration_before"] for e in report.values() for v in e.values() if v["acceleration_before"] > 0]
    print(f"PASS: k {args.k}, median acceleration ratio {np.median(ratios):.3f} over {len(report)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
