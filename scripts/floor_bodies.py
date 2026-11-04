#!/usr/bin/env python3
"""Per-frame foot and body landmarks for floor analysis (mhr_converter.sif, GPU).

Runs the scorer's MHR model on an episode directory and keeps only what floor geometry needs, so later steps run on the
host without the model: [T,V,3] for every foot vertex (dominant skinning joint in a foot subtree), the scored joints and
the body's mean vertex.
Usage: floor_bodies.py --episodes_dir DIR --out DIR
Writes OUT/episode_*.npz: left_foot, right_foot [T,F,3], joints [T,127,3], centroid [T,3], all float32 and in metres
(the model returns vertices in millimetres and joints in metres).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"
FOOT_JOINTS = {"left_foot": 4, "right_foot": 20}


def foot_vertex_sets(model_path=MODEL):
    buffers = dict(torch.jit.load(str(model_path), map_location="cpu").named_buffers())
    parents = buffers["character_torch.skeleton.joint_parents"].numpy()
    vertex = buffers["character_torch.linear_blend_skinning.vert_indices_flattened"].numpy()
    joint = buffers["character_torch.linear_blend_skinning.skin_indices_flattened"].numpy()
    weight = buffers["character_torch.linear_blend_skinning.skin_weights_flattened"].numpy()
    order = np.lexsort((-weight, vertex))
    first = order[np.r_[True, vertex[order][1:] != vertex[order][:-1]]]
    dominant = np.full(vertex.max() + 1, -1)
    dominant[vertex[first]] = joint[first]
    sets = {}
    for name, root in FOOT_JOINTS.items():
        members, frontier = {root}, [root]
        while frontier:
            frontier = [j for j, p in enumerate(parents) if p in frontier]
            members.update(frontier)
        sets[name] = np.flatnonzero(np.isin(dominant, sorted(members)))
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), device, chunk=128, precision="float64")
    sets = foot_vertex_sets()
    print({k: len(v) for k, v in sets.items()}, flush=True)
    for path in sorted(args.episodes_dir.glob("episode_??????.npz")):
        with np.load(path) as archive:
            pose, identity = archive["pose"], np.concatenate([archive["scales"], archive["shape"]])
        identity = torch.as_tensor(identity[None], dtype=torch.float64, device=device)
        out = {name: [] for name in sets} | {"joints": [], "centroid": []}
        for start in range(0, len(pose), 128):
            chunk = torch.as_tensor(pose[start:start + 128], dtype=torch.float64, device=device)
            vertices, joints = mhr.run(chunk, identity)
            vertices = vertices.cpu().numpy() / 1000.0
            for name, index in sets.items():
                out[name].append(vertices[:, index].astype(np.float32))
            out["joints"].append(joints.cpu().numpy().astype(np.float32))
            out["centroid"].append(vertices.mean(1).astype(np.float32))
        np.savez(args.out / path.name, **{k: np.concatenate(v) for k, v in out.items()})
        print(path.stem, len(pose), flush=True)
    print(f"PASS: floor bodies -> {args.out}")


if __name__ == "__main__":
    main()
