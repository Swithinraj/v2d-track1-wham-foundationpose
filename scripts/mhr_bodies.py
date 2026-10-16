#!/usr/bin/env python3
"""MHR joints and dense hand vertices for every episode file of a directory (mhr_converter.sif, GPU).

The scorer runs MHR on the host and scores 22 of the 127 joints (ACC-H); this reproduces them locally
with the kit's MHR wrapper. Dense hand vertices = vertices whose dominant skinning joint lies in the wrist
subtrees (joint 42 = right first, then 78 = left; 2318 each).
  default:      joints [T,127,3] + hands [T,4636,3] (metres), model in float64
  --hands-only: hands + split, as used for contact analysis (model in --precision, default float32)
Usage: mhr_bodies.py --episodes_dir DIR --out DIR [--hands-only] [--precision float32|float64] [--chunk N]
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
WRISTS = (42, 78)


def hand_vertex_sets(model_path=MODEL):
    buffers = dict(torch.jit.load(str(model_path), map_location="cpu").named_buffers())
    parents = buffers["character_torch.skeleton.joint_parents"].numpy()
    vertex = buffers["character_torch.linear_blend_skinning.vert_indices_flattened"].numpy()
    joint = buffers["character_torch.linear_blend_skinning.skin_indices_flattened"].numpy()
    weight = buffers["character_torch.linear_blend_skinning.skin_weights_flattened"].numpy()
    order = np.lexsort((-weight, vertex))
    first = order[np.r_[True, vertex[order][1:] != vertex[order][:-1]]]
    dominant = np.full(vertex.max() + 1, -1)
    dominant[vertex[first]] = joint[first]
    sets = []
    for root in WRISTS:
        members, frontier = {root}, [root]
        while frontier:
            frontier = [j for j, p in enumerate(parents) if p in frontier]
            members.update(frontier)
        sets.append(np.flatnonzero(np.isin(dominant, list(members))))
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hands-only", action="store_true")
    ap.add_argument("--precision", default=None, choices=("float32", "float64"))
    ap.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args()
    precision = args.precision or ("float32" if args.hands_only else "float64")
    chunk = args.chunk or (256 if args.hands_only else 128)
    sets = hand_vertex_sets()
    idx = np.concatenate(sets)
    mhr = MHR(str(MODEL), "cuda", chunk=chunk, precision=precision)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.episodes_dir.glob("episode_*.npz")):
        d = np.load(path)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=mhr.device)
        verts, joints = mhr.run(torch.as_tensor(d["pose"], dtype=torch.float64, device=mhr.device), ident)
        if args.hands_only:
            hands = (verts[:, torch.as_tensor(idx, device=mhr.device)] / 1000.0).cpu().numpy().astype(np.float32)
            np.savez(args.out / path.name, hands=hands, split=len(sets[0]))
        else:
            np.savez_compressed(args.out / path.name, joints=joints.cpu().numpy().astype(np.float32),
                                hands=(verts[:, idx] / 1000.0).cpu().numpy().astype(np.float32))
        print(path.stem, tuple(verts.shape), flush=True)


if __name__ == "__main__":
    main()
