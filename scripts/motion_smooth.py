#!/usr/bin/env python3
"""Extra temporal smoothing of the object's translation (host, CPU). No learned parameters.

ACC-O scores the second difference against the reference. Our object's own acceleration is about 0.03 cm/frame^2 against an
error of about 0.07, which implies the reference carries most of the error by itself. Acceleration of ours that is
uncorrelated with the reference only adds error, so smoothing can help until it removes real motion. How far is a Kaggle
question, so the strength is an explicit experiment. Unscored frames were only bridged (scripts/parquet_episodes.py), so
they carry almost no weight here.

Do not smooth the person this way: MHR's Euler parameters wrap, and a probe here made joint acceleration 16 times worse.
The person goes through scripts/human_refit.py, which smooths the mesh and refits the pose.

Kaggle measured smoothing here as worse for ACC-O, so --sharpen tests the opposite direction.

Usage: motion_smooth.py --input DIR --out DIR --lam_object L [--sharpen S]
Writes OUT/episode_*.npz (+ the other files of --input).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from floor_root import whittaker

UNSCORED_WEIGHT = 1e-3


def smooth_object(translation, frames, lam):
    """[T,3] -> the same, smoothed over time."""
    translation = np.asarray(translation, float)
    if lam <= 0:
        return translation.copy()
    weights = np.full(len(translation), UNSCORED_WEIGHT)
    weights[frames] = 1.
    return whittaker(translation, weights, lam)


def sharpen_object(translation, frames, lam, gain):
    """t + gain * (t - smooth t): the opposite experiment, raising motion that earlier smoothing damped."""
    translation = np.asarray(translation, float)
    return translation + gain * (translation - smooth_object(translation, frames, lam))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lam_object", type=float, required=True)
    parser.add_argument("--sharpen", type=float, default=0., help="use t + s (t - smooth t) instead of smooth t")
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    scored = json.loads(args.scored.read_text())
    shutil.copytree(args.input, args.out)
    for path in sorted(args.input.glob("episode_??????.npz")):
        frames = np.asarray(scored[str(int(path.stem.split("_")[1]))])
        with np.load(path) as archive:
            data = dict(archive)
        change = sharpen_object(data["object_translation"], frames, args.lam_object, args.sharpen) if args.sharpen else smooth_object(data["object_translation"], frames, args.lam_object)
        data["object_translation"] = change.astype(data["object_translation"].dtype)
        np.savez(args.out / path.name, **data)
    print(f"PASS: object smoothing {args.lam_object} -> {args.out}")


if __name__ == "__main__":
    main()
