#!/usr/bin/env python3
"""Probe MHR identity params (scales|shape) for a HAND-SHRINK direction.

For each of the 113 identity params, finite-difference the MHR forward on a few frames
and measure: hand-vertex radial change (toward per-hand centroid), total hand vertex
motion, joint motion, and non-hand body motion. Prints a ranked table; the goal is a
direction that shrinks hands several mm with minimal body/joint impact, to be applied
as a constant per-episode z shift (constant offset => zero ACC-H second difference).

CPU-only (no cluster job). Usage: mhr_probe_shrink.py [--frames N] [--step S]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_to_mhr_params import MHR, NS, NB, HAND_COLS  # noqa: E402
from mhr_bodies import hand_vertex_sets  # noqa: E402

MODEL = ROOT / "weights/mhr/assets/mhr_model.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", type=Path, default=ROOT / "submissions/episodes_cand_P2")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.02)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    mhr = MHR(str(MODEL), "cpu", chunk=64, precision="float32")
    sets = hand_vertex_sets()
    idx = np.concatenate(sets)
    body = np.setdiff1d(np.arange(18439), idx)

    d = np.load(a.episodes_dir / f"episode_{a.episode:06d}.npz")
    T = len(d["pose"])
    fr = np.linspace(0, T - 1, min(a.frames, T)).astype(int)
    pose = torch.as_tensor(d["pose"][fr], dtype=torch.float64)
    z0 = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64)

    v0, j0 = mhr.run(pose, z0)                    # mm, m
    v0 = v0.reshape(len(fr), -1, 3).numpy()
    j0 = j0.reshape(len(fr), -1, 3).numpy()
    hands0 = v0[:, idx]                            # [F, 4636, 3] mm
    # per-hand centroids
    n0 = len(sets[0])
    hc = np.concatenate([hands0[:, :n0].mean(1, keepdims=True), hands0[:, n0:].mean(1, keepdims=True)], 1)
    which = np.where(idx < n0, 0, 1)
    rad0 = np.linalg.norm(hands0 - hc[:, which], axis=2)   # [F, 4636]

    NZ = NS + NB
    rows = []
    for k in range(NZ):
        z = z0.clone()
        z[0, k] += a.step
        v1, j1 = mhr.run(pose, z)
        v1 = v1.reshape(len(fr), -1, 3).numpy()
        j1 = j1.reshape(len(fr), -1, 3).numpy()
        h1 = v1[:, idx]
        rad1 = np.linalg.norm(h1 - hc[:, which], axis=2)
        rows.append((k,
                     float((rad1 - rad0).mean()),                       # radial change mm (neg = shrink)
                     float(np.linalg.norm(h1 - hands0, axis=2).mean()),  # hand motion mm
                     float(np.linalg.norm(j1 - j0, axis=2).mean() * 1000),  # joint motion mm
                     float(np.linalg.norm(v1[:, body] - v0[:, body], axis=2).mean())))  # body motion mm
    rows.sort(key=lambda r: r[1])
    print(f"{'k':>4s} {'name':>14s} {'radial_mm':>10s} {'hand_mm':>9s} {'joint_mm':>9s} {'body_mm':>9s}")
    for k, r, h, j, b in rows[:20]:
        name = f"scales[{k}]" if k < NS else f"shape[{k-NS}]"
        print(f"{k:4d} {name:>14s} {r:10.3f} {h:9.3f} {j:9.3f} {b:9.3f}")
    print("...")
    for k, r, h, j, b in rows[-5:]:
        name = f"scales[{k}]" if k < NS else f"shape[{k-NS}]"
        print(f"{k:4d} {name:>14s} {r:10.3f} {h:9.3f} {j:9.3f} {b:9.3f}")


if __name__ == "__main__":
    main()
