"""Scorer-budgeted object meshes and scored frames of an episodes dir, for container-side tools (submission venv;
pipeline/stages/hand_fix.sh).

The scorer simplifies every object mesh to the sample submission's face/vertex budget (v2dlb.mesh_budget) and
measures PEN against that mesh; containers without trimesh/fast_simplification read the result from here.
Writes <out>/<episode>_budget.npz (vertices, faces; padding faces removed) and <out>/scored_frames.json.
Usage: budget_object_meshes.py --episodes DIR --out DIR
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_hand_penetration as rp  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, budget = rp.scored_frames_and_budget()
    for path in sorted(args.episodes.glob("episode_??????.npz")):
        tag = path.stem
        v, f = budget_mesh(rp.find_mesh(path.parent / f"{tag}_object"), *budget)
        keep = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 0
        np.savez(args.out / f"{tag}_budget.npz", vertices=v, faces=f[keep])
    (args.out / "scored_frames.json").write_text(json.dumps({str(k): [int(x) for x in v] for k, v in frames.items()}))
    print("budget", budget, "->", args.out)


if __name__ == "__main__":
    main()
