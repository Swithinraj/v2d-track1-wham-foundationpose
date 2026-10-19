#!/usr/bin/env python3
"""Human motion refit: temporally smoothed MHR mesh (+ fused wrists, object-anchored hands) -> MHR pose
(mhr_converter.sif, GPU; human refinement stage).

Per-frame human estimates leave wrists/elbows as the jerkiest scored joints, so the fix is at motion level.
MHR Euler parameters wrap by 2*pi and pass near gimbal lock, so instead of smoothing parameters:
  1. vertex targets X = argmin |X - V|^2 + LAM |D2 X|^2 per vertex coordinate over time (V = input mesh);
  2. --fusion DIR: every frame's hand-vertex targets are shifted by the wrist offset of wrist_fusion.py;
  3. --anchors DIR: inside stable grasp windows (hand_anchors.py) the hand-vertex targets become the hand
     rigidly attached to the object track (mean hand shape in the object frame, re-posed by the object pose
     each frame), blended in/out over RAMP frames;
  4. each frame's 136 pose parameters are re-fitted to the targets with the kit's lm_pose (identity fixed).
Options (physics + occlusion):
  --visibility DIR: per-frame weights of the 12 limb parts (body_visibility.py); step 1 becomes a weighted smoother per
     part, argmin sum w_t |X - V|^2 + LAM |D2 X|^2, so a part's hidden (guessed) frames are bridged from its seen ones
     (vertices -> part by their dominant skinning joint's nearest ancestor among the part joints; the rest unweighted);
  --feet FLOOR --kappa REPORT: planted feet: a foot whose lowest vertex is within FOOT_NEAR of the floor and moves less
     than FOOT_STILL per frame for >= FOOT_RUN frames is held still at its run mean and set onto the floor (the
     objects' kappa-scaled floor, n.X = kappa * c / depth_scale), blended in/out over FOOT_RAMP frames.
Objects and identity stay byte-identical. Writes <out>/episode_*.npz (+ meshes) and human_refit_report.json.
Usage: human_refit.py --input DIR --out DIR --lam LAM [--fusion DIR] [--anchors DIR] [--visibility DIR]
       [--feet FLOOR_JSON --kappa KAPPA_REPORT_JSON]
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
FOOT_NEAR, FOOT_STILL, FOOT_RUN, FOOT_RAMP, FOOT_MAX_SHIFT = 0.03, 0.005, 5, 4, 0.05
FOOT_JOINTS = (4, 20)          # l_foot, r_foot (MediaPipe ankles in human_refine_common.PAIRS)


def smoother(T, lam, device):
    D = torch.zeros(T - 2, T, dtype=torch.float64, device=device)
    i = torch.arange(T - 2, device=device)
    D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return torch.eye(T, dtype=torch.float64, device=device) + lam * D.T @ D


def vertex_parts(model_path, part_joints):
    """Part index per vertex: its dominant skinning joint's nearest ancestor among part_joints (-1 = none)."""
    b = dict(torch.jit.load(str(model_path), map_location="cpu").named_buffers())
    parents = b["character_torch.skeleton.joint_parents"].numpy()
    vertex = b["character_torch.linear_blend_skinning.vert_indices_flattened"].numpy()
    joint = b["character_torch.linear_blend_skinning.skin_indices_flattened"].numpy()
    weight = b["character_torch.linear_blend_skinning.skin_weights_flattened"].numpy()
    order = np.lexsort((-weight, vertex))
    first = order[np.r_[True, vertex[order][1:] != vertex[order][:-1]]]
    dominant = np.full(vertex.max() + 1, -1)
    dominant[vertex[first]] = joint[first]
    lookup = {int(j): k for k, j in enumerate(part_joints)}
    part_of_joint = np.full(len(parents), -1)
    for j in range(len(parents)):
        a = j
        while a >= 0 and a not in lookup:
            a = int(parents[a]) if parents[a] != a else -1
        part_of_joint[j] = lookup.get(a, -1) if a >= 0 else -1
    return part_of_joint[dominant]


def weighted_smooth(V, lam, parts, weights, device):
    """Per part: argmin sum_t w_t |X_t - V_t|^2 + lam |D2 X|^2 (V [T, Nv, 3] mm); unassigned vertices: w = 1."""
    T = V.shape[0]
    D = torch.zeros(T - 2, T, dtype=torch.float64, device=device)
    i = torch.arange(T - 2, device=device)
    D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    DtD = lam * D.T @ D
    X = torch.empty_like(V)
    for k in range(-1, weights.shape[1]):
        idx = torch.as_tensor(np.flatnonzero(parts == k), device=device)
        if idx.numel() == 0:
            continue
        w = torch.ones(T, dtype=torch.float64, device=device) if k < 0 else torch.as_tensor(weights[:T, k], dtype=torch.float64, device=device)
        Y = V[:, idx].reshape(T, -1)
        X[:, idx] = torch.linalg.solve(torch.diag(w) + DtD, w[:, None] * Y).reshape(T, len(idx), 3)
    return X


def plant_feet(X, foot_idx, n, c_eff):
    """Planted-foot targets (X [T, Nv, 3] metres). Returns X and the number of planted frames per foot."""
    T = X.shape[0]
    counts = []
    for idx in foot_idx:
        F_ = X[:, idx]
        sole = (F_ @ n).min(1) - c_eff
        speed = np.r_[0.0, np.linalg.norm(np.diff(F_.mean(1), axis=0), axis=1)]
        planted = (sole < FOOT_NEAR) & (speed < FOOT_STILL)
        planted = np.array([np.median(planted[max(0, f - 2):f + 3]) > 0.5 for f in range(T)])
        frames, f = 0, 0
        while f < T:
            if not planted[f]:
                f += 1
                continue
            g = f
            while g < T and planted[g]:
                g += 1
            if g - f >= FOOT_RUN:
                held = F_[f:g].mean(0)
                shift = (held @ n).min() - c_eff
                if abs(shift) <= FOOT_MAX_SHIFT:
                    held = held - shift * n
                    m = np.arange(g - f)
                    beta = np.clip(np.minimum(m + 1, g - f - m) / FOOT_RAMP, 0, 1)
                    beta = 0.5 - 0.5 * np.cos(np.pi * beta)
                    X[f:g, idx] = (1 - beta[:, None, None]) * X[f:g, idx] + beta[:, None, None] * held[None]
                    frames += g - f
            f = g
        counts.append(frames)
    return X, counts


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
    ap.add_argument("--visibility", type=Path, default=None)
    ap.add_argument("--feet", type=Path, default=None, help="floor planes JSON {episode: {normal, c}} (MoGe metres)")
    ap.add_argument("--kappa", type=Path, default=None, help="kappa_gravity_report.json (the objects' kappa)")
    args = ap.parse_args()
    dev = "cuda"
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64")
    hsets = hand_vertex_sets(MODEL)
    parts = None
    if args.visibility is not None:
        part_joints = np.load(next(args.visibility.glob("episode_*.npz")))["joints"]
        parts = vertex_parts(MODEL, part_joints)
    if args.feet is not None:
        floors = json.loads(args.feet.read_text())
        kappas = json.loads(args.kappa.read_text())["episodes"] if args.kappa else {}
        foot_parts = vertex_parts(MODEL, np.array(FOOT_JOINTS))
        foot_idx = [np.flatnonzero(foot_parts == k) for k in range(len(FOOT_JOINTS))]
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
        if parts is not None and args.lam > 0:
            X = weighted_smooth(V, args.lam, parts, np.load(args.visibility / f"{tag}.npz")["weights"].astype(np.float64), dev)
        else:
            X = torch.linalg.solve(smoother(T, args.lam, dev), V.reshape(T, -1)).reshape(V.shape) if args.lam > 0 else V.clone()
        used, planted = [], []
        if args.feet is not None:
            align = json.loads((ROOT / "outputs/alignment_sam" / tag / "alignment.json").read_text())
            n = np.asarray(floors[tag]["normal"], float)
            c_eff = floors[tag]["c"] / align["depth_scale"] * float(kappas.get(tag, {}).get("kappa", 1.0))
            Xm, planted = plant_feet(X.cpu().numpy() / 1000.0, foot_idx, n, c_eff)
            X = torch.as_tensor(Xm * 1000.0, dtype=torch.float64, device=dev)
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
                       "anchored_windows": len(used), "planted_frames": planted}
        print(tag, json.dumps(report[tag]), flush=True)
    (args.out / "human_refit_report.json").write_text(json.dumps({"lam": args.lam, "fusion": str(args.fusion), "anchors": str(args.anchors),
                                                                  "visibility": str(args.visibility), "feet": str(args.feet),
                                                                  "episodes": report}, indent=2))
    r = list(report.values())
    print(f"refit lam {args.lam:g}: joint acc {np.mean([x['joint_acc_cm'][0] for x in r]):.4f} -> {np.mean([x['joint_acc_cm'][1] for x in r]):.4f} "
          f"cm/frame^2 | vertex move {np.mean([x['vertex_move_mm'] for x in r]):.1f} mm | fit err {np.mean([x['fit_err_mm'] for x in r]):.2f} mm")


if __name__ == "__main__":
    main()
