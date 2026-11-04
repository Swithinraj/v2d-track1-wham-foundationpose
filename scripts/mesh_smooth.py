#!/usr/bin/env python3
"""Temporal smoothing of MHR-topology vertices over scored frames (host, CPU), before converting them to parameters.

CARI4D's output moves far more than a person does: about 0.9-2.2 cm/frame^2 per joint, against 0.06 for the scored candidates.
Smoothing on the mesh avoids MHR's wrapping Euler parameters. Scored stretches separated by a gap are smoothed on the video's
frame axis with no data in the gap, so a gap is bridged smoothly and never forces neighbouring frames together.

Usage: mesh_smooth.py --verts IN.npy --frames JSON_OR_EPISODE --lam L --out OUT.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floor_root import whittaker


def smooth_scored(vertices, frames, lam):
    """[F,V,3] over scored frames -> the same, smoothed on the video's frame axis."""
    vertices = np.asarray(vertices)
    frames = np.asarray(frames)
    if len(vertices) != len(frames) or vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices must be [F,V,3] over the scored frames")
    if (np.diff(frames) <= 0).any():
        raise ValueError("scored frames must increase")
    if lam <= 0:
        return vertices.astype(np.float32)
    first = int(frames[0])
    T = int(frames[-1]) - first + 1
    index = frames - first
    weights = np.zeros(T)
    weights[index] = 1.
    grid = np.zeros((T, vertices.shape[1] * 3))
    grid[index] = np.asarray(vertices, float).reshape(len(frames), -1)
    return whittaker(grid, weights, lam)[index].reshape(vertices.shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verts", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=Path("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    args = parser.parse_args()
    frames = np.asarray(json.loads(args.scored.read_text())[str(args.episode)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, smooth_scored(np.load(args.verts), frames, args.lam))


if __name__ == "__main__":
    main()
