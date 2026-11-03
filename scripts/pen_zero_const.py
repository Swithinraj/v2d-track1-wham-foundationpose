#!/usr/bin/env python3
"""PEN-zero by minimal CONSTANT per-episode object offset (LP).

Finds, per episode, the minimum-norm constant translation c added to object_translation
such that NO dense hand point is inside (or within --margin of) any posed object mesh at
ANY scored frame. A constant offset has exactly zero ACC-O cost (second differences of a
constant are 0) and a CD-O cost of |c| on that episode's frames only, which the report
summarises. Constraints come from inside/near-boundary hand points: with outward face
normal n and closest surface point q, requiring the shifted point p - c to lie outside by
margin gives  n . c <= n . (p - q) - margin.  Solved as an L1-minimisation LP (HiGHS).
Iterated up to --rounds times: after each solve the scene is re-tested and any frame that
still has inside points contributes new constraints.

Usage: pen_zero_const.py --cand DIR --hands DIR --out DIR [--margin M] [--episodes N ...]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

import os
import sys
ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "official/v2d_submission_kit"))
from v2dlb import mhr_submission as ms  # noqa: E402

from scipy.optimize import linprog  # noqa: E402


def scene_inside(V, F, s, R, t, pts, kernels):
    """Return (inside_pts, near_pts) for one frame."""
    w, d = kernels
    X = s * (V @ R.T) + t
    lo, hi = X.min(axis=0), X.max(axis=0)
    near = np.flatnonzero(((pts >= lo) & (pts <= hi)).all(axis=1))
    if near.size == 0:
        return np.empty(0), np.empty(0), X
    q = pts[near]
    win = w(q, X[F[:, 0]], X[F[:, 1]], X[F[:, 2]])
    return near[np.abs(win) > 0.5], near, X


def solve_episode(V, F, s, R, T, hands, margin, rounds, max_shift):
    """Iterative LP; returns (c, n_frames_still_inside)."""
    kernels = ms._PENETRATION_KERNELS
    A_rows, b_rows = [], []
    c = np.zeros(3)
    still = -1
    for _ in range(rounds):
        Tc = T + c
        new_inside = 0
        for i in range(len(T)):
            if not np.isfinite(T[i]).all():
                continue
            ins, near, X = scene_inside(V, F, s, R[i], Tc[i], hands[i], kernels)
            if ins.size == 0:
                continue
            new_inside += 1
            mesh = trimesh.Trimesh(vertices=X, faces=F, process=False)
            mesh.fix_normals()
            # constrain inside points AND outside points within margin (shift must not
            # push them through the surface either)
            (cp, d, tri) = trimesh.proximity.closest_point(mesh, hands[i][near])
            n = mesh.face_normals[tri]
            off = np.einsum("ij,ij->i", n, hands[i][near] - cp)
            keep = (off < margin) | np.isin(near, ins)
            A_rows.append(n[keep])
            b_rows.append(off[keep] - margin)
        if new_inside == 0:
            still = 0
            break
        A = np.vstack(A_rows)
        b = np.concatenate(b_rows)
        # L1 min-norm: vars [c(3), u(3)]; min sum u;  n.c <= b ; -u <= c <= u
        A_ub = np.zeros((len(A) + 6, 6))
        b_ub = np.zeros(len(A) + 6)
        A_ub[:len(A), :3] = A
        b_ub[:len(A)] = b
        A_ub[len(A):len(A) + 3, 3:] = np.eye(3)    # c_k - u_k <= 0
        A_ub[len(A) + 3:, :3] = np.eye(3)          # -c_k - u_k <= 0
        A_ub[len(A) + 3:, 3:] = -np.eye(3)
        bounds = [(None, None)] * 3 + [(0, max_shift)] * 3
        res = linprog(np.r_[np.zeros(3), np.ones(3)], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return None, new_inside
        c = c + res.x[:3]
        still = new_inside
    return c, still


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.001, help="clearance (m)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-shift", type=float, default=0.30, help="LP upper bound per axis (m)")
    ap.add_argument("--episodes-list", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    if ms._PENETRATION_KERNELS is None:
        raise SystemExit("numba penetration kernels unavailable -- run in .venvs/submission")
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for ep in a.episodes_list:
        tag = f"episode_{ep:06d}"
        src = a.cand / f"{tag}.npz"
        if not src.exists():
            continue
        t0 = time.time()
        data = dict(np.load(src))
        f_hands = a.hands / f"{tag}.npz"
        if not f_hands.exists():
            shutil.copy2(src, a.out / src.name)
            print(f"{tag}: no hands npz, copied unchanged", flush=True)
            continue
        hands = np.load(f_hands)["hands"].astype(np.float64)
        mesh_p = a.cand / f"{tag}_object.glb"
        if not mesh_p.exists():
            mesh_p = a.cand / f"{tag}_object.ply"
        mesh = trimesh.load(mesh_p, force="mesh")
        V = np.asarray(mesh.vertices, np.float64)
        F = np.asarray(mesh.faces, np.int64)
        s = float(np.asarray(data["object_scale"]).reshape(()))
        R = data["object_rotation"].astype(np.float64)
        T = data["object_translation"].astype(np.float64)
        n = min(len(T), len(hands))
        c, still = solve_episode(V[:], F, s, R[:n], T[:n], hands[:n], a.margin, a.rounds, a.max_shift)
        if c is None:
            print(f"{tag}: LP INFEASIBLE, left unchanged", flush=True)
            rep[tag] = {"status": "infeasible"}
            np.savez(a.out / src.name, **data)
        else:
            data["object_translation"][:n] = T[:n] + c
            np.savez(a.out / src.name, **data)
            rep[tag] = {"offset_cm": [round(x * 100, 3) for x in c],
                        "norm_cm": round(float(np.linalg.norm(c)) * 100, 3),
                        "frames_still_inside": int(still),
                        "sec": round(time.time() - t0, 1)}
            print(f"{tag}: offset {rep[tag]['offset_cm']} cm |{rep[tag]['norm_cm']:.2f}|, "
                  f"frames still inside {still}, {rep[tag]['sec']}s", flush=True)
        for ext in ("_object.glb", "_object.ply"):
            p = a.cand / f"{tag}{ext}"
            if p.exists():
                shutil.copy2(p, a.out / p.name)
    (a.out / "pen_zero_const_report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
