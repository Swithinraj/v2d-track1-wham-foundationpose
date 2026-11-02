#!/usr/bin/env python3
"""Scale the objects of chosen episodes about the camera by one factor (host, CPU): a depth dose that keeps every pixel.

Scaling an object's translation and scale together about the camera centre moves it along its viewing rays and changes
its size in proportion, so its image is unchanged; only its depth relative to the person (and so CD-O) moves. Floor
objects already take their depth from floor contact; objects on tables have no such evidence and keep MoGe's. A global
0.97 cost about +4 CD-O, which put the best global scale near 1.01, and 0.974 on table objects alone cost +5.6 per such
episode, so the dose is tested above 1. Rotation, mesh and the human are untouched.

Usage: object_camera_scale.py --input DIR --out DIR --scale S --episodes N [N ...]
Writes OUT/episode_*.npz (+ the other files of INPUT) and object_camera_scale_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def scale_objects(data, factor):
    """Copy of an episode dict with object translation and scale multiplied by factor."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be a positive number")
    out = dict(data)
    out["object_translation"] = (data["object_translation"].astype(np.float64) * factor).astype(data["object_translation"].dtype)
    out["object_scale"] = np.asarray(float(data["object_scale"]) * factor, dtype=np.asarray(data["object_scale"]).dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    shutil.copytree(args.input, args.out)
    report = {}
    for episode in sorted(set(args.episodes)):
        path = args.input / f"episode_{episode:06d}.npz"
        with np.load(path) as archive:
            data = dict(archive)
        np.savez(args.out / path.name, **scale_objects(data, args.scale))
        report[path.stem] = dict(scale=args.scale, depth_change_m=float(np.linalg.norm(data["object_translation"], axis=1).mean() * (args.scale - 1)))
    (args.out / "object_camera_scale_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PASS: objects of {len(report)} episodes scaled by {args.scale} about the camera -> {args.out}")


if __name__ == "__main__":
    main()
