#!/usr/bin/env python3
"""Exactly round replacement mesh for a rotationally symmetric object (config/symmetric_objects.json).

kind=torus: a hula hoop is round, but the generated mesh is not quite. This writes a torus around the y axis
with the generated mesh's median centreline radius and tube radius (measured about its own ring axis, the
direction of least spread), so the object's spin about its axis is truly unobservable and invisible to CD-O.
Runs in v2d_sam3d.sif (trimesh). Output: inputs/object_meshes/<object>.symmetric.glb (+ .json with R, r).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / "config" / "symmetric_objects.json").read_text())[args.object]
    if spec["kind"] != "torus" or spec["axis"] != [0.0, 1.0, 0.0]:
        raise SystemExit(f"unsupported symmetry spec {spec}")

    src = trimesh.load(ROOT / "inputs" / "object_meshes" / f"{args.object}.glb", force="mesh")
    v = src.vertices - src.bounds.mean(axis=0)
    # The ring axis is the direction of least spread (the generator may return the ring in any orientation);
    # express the vertices with that axis as y. Only the ring and tube radii are used downstream.
    _, _, axes = np.linalg.svd(v - v.mean(axis=0), full_matrices=False)
    v = v @ np.stack([axes[0], axes[2], axes[1]]).T
    ring_radius = float(np.median(np.linalg.norm(v[:, [0, 2]], axis=1)))
    tube_radius = float((np.percentile(v[:, 1], 98) - np.percentile(v[:, 1], 2)) / 2)
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 360, endpoint=False), np.linspace(0, 2 * np.pi, 24, endpoint=False),
                       indexing="ij")
    vertices = np.stack([(ring_radius + tube_radius * np.cos(w)) * np.cos(u), tube_radius * np.sin(w),
                         (ring_radius + tube_radius * np.cos(w)) * np.sin(u)], -1).reshape(-1, 3)
    index = np.arange(u.size).reshape(u.shape)
    a, b = index, np.roll(index, -1, 0)
    c, d = np.roll(b, -1, 1), np.roll(index, -1, 1)
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    color = np.r_[np.asarray(src.visual.vertex_colors)[:, :3].mean(axis=0), 255].astype(np.uint8)
    torus = trimesh.Trimesh(vertices, faces, vertex_colors=np.tile(color, (len(vertices), 1)), process=True)
    out = ROOT / "inputs" / "object_meshes" / f"{args.object}.symmetric.glb"
    torus.export(out)
    out.with_suffix(".json").write_text(json.dumps({
        "object": args.object, "kind": "torus", "axis": spec["axis"], "ring_radius_m": ring_radius,
        "tube_radius_m": tube_radius, "source_extents_m": [float(x) for x in src.extents], "ground_truth": False}, indent=2))
    print(f"PASS: {out.name}: ring radius {ring_radius:.4f} m, tube radius {tube_radius:.4f} m, "
          f"extents {np.round(torus.extents, 3).tolist()} (generated mesh {np.round(src.extents, 3).tolist()})")


if __name__ == "__main__":
    main()
