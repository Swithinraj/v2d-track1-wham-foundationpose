#!/usr/bin/env python3
"""Ground-truth-free checks of a candidate before it costs a Kaggle submission (submission venv, CPU).

Compares episode directories on the scored frames with the kit's own penetration code:
  self-PEN    mean depth of hand vertices inside the object mesh (cm); the scorer's PEN uses 512 of these points
  self-ACC-H  mean size of the 22 scored joints' second difference within each stretch (cm/frame^2)
  self-ACC-O  the same for the object translation
  anchor      largest joint move at the first scored frame against a reference directory (mm); a change here moves the
              scorer's alignment, and with it CD-O
These are not CD-H or CD-O. No reference exists locally, and only Kaggle measures those.

Usage: local_gates.py --scored JSON NAME=EPISODES_DIR:BODIES_DIR [NAME=...]   (BODIES_DIR from mhr_bodies.py)
The first candidate is the reference for the anchor check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit"))
from v2dlb.mhr_submission import _penetration_depth  # noqa: E402

SCORED_JOINTS = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)


def stretches(frames):
    return np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)


def self_acceleration(points, frames):
    values = []
    for part in stretches(frames):
        if len(part) >= 3:
            second = np.diff(points[part], n=2, axis=0)
            values.append(np.linalg.norm(second, axis=-1).reshape(len(second), -1).mean(1))
    return float(np.concatenate(values).mean() * 100) if values else 0.


def self_penetration(hands, data, mesh, frames, sample=40):
    use = frames[np.unique(np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int))]
    scale = float(data["object_scale"])
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    values = []
    for f in use:
        local = (hands[f, ::9] - data["object_translation"][f]) @ data["object_rotation"][f].astype(float) / scale
        values.append(_penetration_depth(local.astype(float), vertices, faces).mean() * scale * 100)
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("candidates", nargs="+", help="NAME=EPISODES_DIR:BODIES_DIR")
    args = parser.parse_args()
    scored = json.loads(args.scored.read_text())
    reference = {}
    for k, item in enumerate(args.candidates):
        name, paths = item.split("=", 1)
        episodes, bodies = (Path(p) for p in paths.split(":"))
        pen, acc_h, acc_o, anchor = [], [], [], []
        for episode in range(30):
            tag = f"episode_{episode:06d}"
            frames = np.asarray(scored[str(episode)])
            data = np.load(episodes / f"{tag}.npz")
            body = np.load(bodies / f"{tag}.npz")
            joints = body["joints"][:, list(SCORED_JOINTS)].astype(float)
            mesh = trimesh.load(episodes / f"{tag}_object.glb", force="mesh", process=False)
            pen.append(self_penetration(body["hands"].astype(float), data, mesh, frames))
            acc_h.append(self_acceleration(joints, frames))
            acc_o.append(self_acceleration(data["object_translation"].astype(float), frames))
            if k == 0:
                reference[tag] = joints[frames[0]]
            anchor.append(float(np.linalg.norm(joints[frames[0]] - reference[tag], axis=-1).max() * 1000))
        print(f"{name:12s} self-PEN {np.mean(pen):.5f}  self-ACC-H {np.mean(acc_h):.5f}  self-ACC-O {np.mean(acc_o):.5f}  "
              f"anchor move mm: median {np.median(anchor):.3f} max {np.max(anchor):.3f}", flush=True)


if __name__ == "__main__":
    main()
