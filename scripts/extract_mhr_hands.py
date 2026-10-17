#!/usr/bin/env python3
"""Per-frame MHR hand points for the hand-object penetration (PEN) step.

Runs inside containers/mhr_converter.sif. The scorer measures PEN with 512 hand points (left then
right) taken from host-side MHR vertex roles that the kit does not ship. This approximates them
from the public MHR model: hand vertices are those whose dominant skinning joint lies in one of the
two wrist subtrees (joints 42 and 78: 5 finger chains, 23 joints each), and 256 are drawn per hand
with a fixed seed. Bodies come from assembled episode files (--episodes_dir) through the kit's MHR wrapper,
so they are in the Track 1 frame and metres.

Writes outputs/hands/episode_XXXXXX.npz (hands [T, 512, 3]) and outputs/hands/mhr_hand_vertices.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official" / "v2d_submission_kit" / "tools" / "track1"))
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = ROOT / "weights" / "mhr" / "assets" / "mhr_model.pt"
WRIST_JOINTS = (42, 78)
POINTS_PER_HAND = 256


def hand_vertex_indices() -> list[np.ndarray]:
    buffers = dict(torch.jit.load(str(MODEL), map_location="cpu").named_buffers())
    parents = buffers["character_torch.skeleton.joint_parents"].numpy()
    vertex = buffers["character_torch.linear_blend_skinning.vert_indices_flattened"].numpy()
    joint = buffers["character_torch.linear_blend_skinning.skin_indices_flattened"].numpy()
    weight = buffers["character_torch.linear_blend_skinning.skin_weights_flattened"].numpy()
    order = np.lexsort((-weight, vertex))
    first = order[np.r_[True, vertex[order][1:] != vertex[order][:-1]]]
    dominant = np.full(vertex.max() + 1, -1)
    dominant[vertex[first]] = joint[first]

    def subtree(root: int) -> set[int]:
        members, frontier = {root}, [root]
        while frontier:
            children = [j for j, p in enumerate(parents) if p in frontier]
            members.update(children)
            frontier = children
        return members

    rng = np.random.default_rng(0)
    hands = []
    for wrist in WRIST_JOINTS:
        candidates = np.flatnonzero(np.isin(dominant, list(subtree(wrist))))
        hands.append(np.sort(rng.choice(candidates, POINTS_PER_HAND, replace=False)))
    return hands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes_dir", type=Path, required=True, help="assembled episode NPZ files")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    out = ROOT / "outputs" / "hands"
    out.mkdir(parents=True, exist_ok=True)
    first_hand, second_hand = hand_vertex_indices()
    indices = np.concatenate([first_hand, second_hand])
    (out / "mhr_hand_vertices.json").write_text(json.dumps({
        "wrist_joints": list(WRIST_JOINTS), "points_per_hand": POINTS_PER_HAND,
        "first_hand": first_hand.tolist(), "second_hand": second_hand.tolist(),
        "note": "approximation of the scorer's host-side hand roles"}, indent=1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), device, chunk=256 if device == "cuda" else 64, precision="float32")
    paths = sorted(args.episodes_dir.glob("episode_*.npz"))
    tag_of = lambda p: p.stem  # noqa: E731
    if args.episodes is not None:
        paths = [p for p in paths if int(tag_of(p).split("_")[1]) in args.episodes]
    for path in paths:
        data = dict(np.load(path))
        pose = torch.as_tensor(data["pose"], dtype=torch.float64, device=mhr.device)
        identity = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None], dtype=torch.float64,
                                   device=mhr.device)
        vertices_mm = mhr.run(pose, identity)[0]
        hands = (vertices_mm[:, torch.as_tensor(indices, device=mhr.device)] / 1000.0).cpu().numpy()
        np.savez(out / f"{tag_of(path)}.npz", hands=hands.astype(np.float32))
        print(f"{tag_of(path)}: hands {hands.shape}", flush=True)
    print(f"PASS: hand points for {len(paths)} episodes")


if __name__ == "__main__":
    main()
