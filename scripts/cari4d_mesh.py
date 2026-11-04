#!/usr/bin/env python3
"""The object mesh for NVIDIA's CARI4D pipeline, scaled to one size per physical object (submission venv, CPU).

CARI4D keeps the mesh's scale, so the mesh decides how far away FoundationPose places the object: a mesh that is too large
gets placed too far. The size comes from scripts/object_true_size.py. The generated input mesh has vertex colors for
FoundationPose's renders. Where its shape does not match the scored file's mesh (the paint roller's input mesh is only the
roller head), the scored mesh is used instead.

Usage: cari4d_mesh.py --sizes JSON --object NAME --scored_mesh GLB --scored_scale S --out GLB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHAPE_TOLERANCE = 0.25      # largest relative difference between the two meshes' sorted proportions


def proportions(extents):
    ordered = np.sort(np.asarray(extents, float))
    return ordered / ordered[-1]


def shapes_match(a, b, tolerance=SHAPE_TOLERANCE):
    return bool((np.abs(proportions(a) - proportions(b)) <= tolerance * np.maximum(proportions(b), 0.05)).all())


def main():
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--inputs", type=Path, default=Path("inputs/object_meshes"))
    parser.add_argument("--scored_mesh", type=Path, required=True)
    parser.add_argument("--scored_scale", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    size = json.loads(args.sizes.read_text())[args.object]
    scored = trimesh.load(args.scored_mesh, force="mesh", process=False)
    scored.apply_scale(args.scored_scale)
    candidate = trimesh.load(args.inputs / f"{args.object}.glb", force="mesh", process=False)
    use_input = shapes_match(candidate.extents, scored.extents)
    mesh = candidate if use_input else scored
    mesh.apply_scale(size / float(np.linalg.norm(mesh.extents)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    print(json.dumps(dict(object=args.object, source="input" if use_input else "scored", size_m=size, faces=len(mesh.faces),
                          extents_m=np.round(np.sort(mesh.extents), 3).tolist())))


if __name__ == "__main__":
    main()
