#!/usr/bin/env python3
"""Human motion refit: temporally smoothed MHR mesh (+ fused wrists, object-anchored hands) -> MHR pose
(mhr_converter.sif, GPU; human refinement stage).

WHAM's own wrists/elbows are the jerkiest scored joints and the WHAM->MHR conversion adds nothing to that,
so the fix is at motion level. MHR Euler parameters wrap by 2*pi and pass near gimbal lock, so instead of
smoothing parameters:
  1. vertex targets X = argmin |X - V|^2 + LAM |D2 X|^2 per vertex coordinate over time (V = input mesh);
  2. --fusion DIR: every frame's hand-vertex targets are shifted by the wrist offset of wrist_fusion.py;
  3. --anchors DIR: inside stable grasp windows (hand_anchors.py) the hand-vertex targets become the hand
     rigidly attached to the object track (mean hand shape in the object frame, re-posed by the object pose
     each frame), blended in/out over RAMP frames;
  4. each frame's 136 pose parameters are re-fitted to the targets with the kit's lm_pose (identity fixed).
Objects and identity stay byte-identical. Writes <out>/episode_*.npz (+ meshes) and human_refit_report.json.
Usage: human_refit.py --input DIR --out DIR --lam LAM [--fusion DIR] [--anchors DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit/tools/track1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from mhr_bodies import MODEL, WRISTS, hand_vertex_sets  # noqa: E402

SCORED_J = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)
HAND_SPLIT = 2318
RAMP = 15


def smoother(T, lam, device):
    D = torch.zeros(T - 2, T, dtype=torch.float64, device=device)
    i = torch.arange(T - 2, device=device)
    D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return torch.eye(T, dtype=torch.float64, device=device) + lam * D.T @ D


def object_anchor_targets(X, hsets, data, anchor_file):
    """Hand-vertex targets inside stable grasp windows (X [T, V, 3] metres, Track 1 frame)."""
    z = np.load(anchor_file)
    rows = json.loads(str(z["meta"]))
    R, t = data["object_rotation"].astype(np.float64), data["object_translation"].astype(np.float64)
    for k, r in enumerate(rows):
        a, b, idx = r["start"], r["end"], hsets[r["hand"]]
        anchored = np.einsum("tij,vj->tvi", R[a:b], z[f"w{k}"].astype(np.float64)) + t[a:b, None]
        n = np.arange(b - a)
        beta = np.clip(np.minimum(n + 1, b - a - n) / RAMP, 0, 1)                  # 0 -> 1 -> 0 at the edges
        beta = 0.5 - 0.5 * np.cos(np.pi * beta)
        X[a:b, idx] = (1 - beta[:, None, None]) * X[a:b, idx] + beta[:, None, None] * anchored
    return X, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--fusion", type=Path, default=None)
    ap.add_argument("--anchors", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda"
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64")
    hsets = hand_vertex_sets(MODEL)
    assert len(hsets[0]) == HAND_SPLIT and len(hsets[1]) == HAND_SPLIT, [len(h) for h in hsets]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(args.input.glob("episode_*.npz")):
        tag = path.stem
        data = dict(np.load(path))
        T = len(data["pose"])
        ident = torch.as_tensor(np.concatenate([data["scales"], data["shape"]])[None], dtype=torch.float64, device=dev)
        pose0 = torch.as_tensor(data["pose"], dtype=torch.float64, device=dev)
        V, J0 = mhr.run(pose0, ident)                                                     # mm, m
        X = torch.linalg.solve(smoother(T, args.lam, dev), V.reshape(T, -1)).reshape(V.shape) if args.lam > 0 else V.clone()
        used = []
        if args.fusion is not None:
            off = torch.as_tensor(np.load(args.fusion / f"{tag}.npz")["offsets"].astype(np.float64) * 1000.0, device=dev)
            for h in range(2):
                X[:, hsets[h]] += off[:T, h, None]
        if args.anchors is not None:
            Xm, used = object_anchor_targets(X.cpu().numpy() / 1000.0, hsets, data, args.anchors / f"{tag}.npz")
            X = torch.as_tensor(Xm * 1000.0, dtype=torch.float64, device=dev)
        pose1, err = lm_pose(mhr, X.reshape(T, -1), pose0.clone(), ident, iters=6)
        V1, J1 = mhr.run(pose1, ident)
        data["pose"] = pose1.cpu().numpy().astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **data)
        for mesh in args.input.glob(f"{tag}_object.*"):
            shutil.copy(mesh, args.out / mesh.name)
        acc = lambda J, cols: float((torch.linalg.norm(J[:-2] - 2 * J[1:-1] + J[2:], dim=-1)[:, list(cols)].mean() * 100).item())
        report[tag] = {"fit_err_mm": float(err.mean()), "vertex_move_mm": float((V1 - V).norm(dim=-1).mean()),
                       "joint_acc_cm": [acc(J0, SCORED_J), acc(J1, SCORED_J)], "wrist_acc_cm": [acc(J0, WRISTS), acc(J1, WRISTS)],
                       "anchored_windows": len(used)}
        print(tag, json.dumps(report[tag]), flush=True)
    (args.out / "human_refit_report.json").write_text(json.dumps({"lam": args.lam, "fusion": str(args.fusion), "anchors": str(args.anchors),
                                                                  "episodes": report}, indent=2))
    r = list(report.values())
    print(f"refit lam {args.lam:g}: joint acc {np.mean([x['joint_acc_cm'][0] for x in r]):.4f} -> {np.mean([x['joint_acc_cm'][1] for x in r]):.4f} "
          f"cm/frame^2 | vertex move {np.mean([x['vertex_move_mm'] for x in r]):.1f} mm | fit err {np.mean([x['fit_err_mm'] for x in r]):.2f} mm")


if __name__ == "__main__":
    main()
