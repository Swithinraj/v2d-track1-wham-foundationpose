#!/usr/bin/env python3
"""PEN-zero by per-frame object ROTATION about the centroid (ACC-O-free expulsion).

The kit's ACC-O metric uses only the object_translation second differences -- object
rotation never enters it. So rotating each frame's object to expel hand points costs
EXACTLY zero ACC-O regardless of temporal jumpiness, while the CD-O cost is the surface
displacement, which the LP minimises.

Per frame: inside hand points (winding > 0.5) with closest surface point q, outward
normal n, depth d. A small rotation omega about centroid c moves surface point x by
omega x (x - c). Requiring the surface at p to move away from hand point p:
    n . (omega x (x - c)) <= -(d + margin)   <=>   omega . ((x - c) x n) <= -(d + margin)
-- linear in omega. L1-minimal omega via LP (HiGHS), applied as a true Rodrigues
rotation, iterated --rounds times until no hand point is inside anywhere.

Translation is never modified, so ACC-O is bit-identical to the input candidate.
Frames the rotation cannot clear (point too close to the pivot) are reported.

Usage: pen_zero_rot.py --cand DIR --hands DIR --out DIR [--margin M] [--episodes N ...]
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


def rodrigues(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def solve_frame_omega(V, F, s, R, t, pts, margin, rounds, max_angle):
    """Rotate the posed mesh about its translation anchor t (the pivot implied by
    R' = dR @ R with translation fixed) until no point of pts is inside.
    Returns (dR, ok)."""
    kernels = ms._PENETRATION_KERNELS
    w, _ = kernels
    R_acc = np.eye(3)
    for _ in range(rounds):
        X = (s * (V @ R.T)) @ R_acc.T + t            # rotate local scaled verts about anchor t
        lo, hi = X.min(axis=0), X.max(axis=0)
        near = np.flatnonzero(((pts >= lo) & (pts <= hi)).all(axis=1))
        if near.size == 0:
            return R_acc, True
        q = pts[near]
        win = w(q, X[F[:, 0]], X[F[:, 1]], X[F[:, 2]])
        ins = near[np.abs(win) > 0.5]
        if ins.size == 0:
            return R_acc, True
        mesh = trimesh.Trimesh(vertices=X, faces=F, process=False)
        mesh.fix_normals()
        cp, dist, tri = trimesh.proximity.closest_point(mesh, pts[ins])
        n = mesh.face_normals[tri]
        depth = np.einsum("ij,ij->i", -n, pts[ins] - cp)      # positive inside
        r = cp - t                                            # lever arms about the pivot
        # constraint: n_i . (omega x r_i) <= -(depth_i + margin)
        # n . (omega x r) = omega . (r x n)
        A = np.cross(r, n)
        b = -(depth + margin)
        A_ub = np.zeros((len(A) + 6, 6))
        b_ub = np.zeros(len(A) + 6)
        A_ub[:len(A), :3] = A
        b_ub[:len(A)] = b
        A_ub[len(A):len(A) + 3, 3:] = np.eye(3)
        A_ub[len(A) + 3:, :3] = np.eye(3)
        A_ub[len(A) + 3:, 3:] = -np.eye(3)
        res = linprog(np.r_[np.zeros(3), np.ones(3)], A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * 3 + [(0, None)] * 3, method="highs")
        if not res.success:
            return R_acc, False
        omega = res.x[:3]
        if np.linalg.norm(omega) > max_angle:                 # cap per-round angle
            omega = omega / np.linalg.norm(omega) * max_angle
        R_acc = rodrigues(omega) @ R_acc
    # final check
    X = (s * (V @ R.T) + t - t) @ R_acc.T + t
    lo, hi = X.min(axis=0), X.max(axis=0)
    near = np.flatnonzero(((pts >= lo) & (pts <= hi)).all(axis=1))
    if near.size == 0:
        return R_acc, True
    win = w(pts[near], X[F[:, 0]], X[F[:, 1]], X[F[:, 2]])
    return R_acc, bool((np.abs(win) <= 0.5).all())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.0005, help="clearance (m)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-angle", type=float, default=0.35, help="per-round cap (rad)")
    ap.add_argument("--max-total-angle", type=float, default=1.2, help="abort if total rotation exceeds (rad)")
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
        R_new = R.copy()
        frames_touched, frames_failed = 0, 0
        disp_sum, disp_frames = 0.0, 0
        for i in range(n):
            if not np.isfinite(T[i]).all():
                continue
            X = s * (V @ R[i].T) + T[i]
            lo, hi = X.min(axis=0), X.max(axis=0)
            near = np.flatnonzero(((hands[i] >= lo) & (hands[i] <= hi)).all(axis=1))
            if near.size == 0:
                continue
            w, _ = ms._PENETRATION_KERNELS
            if not (np.abs(w(hands[i][near], X[F[:, 0]], X[F[:, 1]], X[F[:, 2]])) > 0.5).any():
                continue
            dR, ok = solve_frame_omega(V, F, s, R[i], T[i], hands[i], a.margin, a.rounds, a.max_angle)
            ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            if (not ok) or ang > a.max_total_angle:
                frames_failed += 1
                continue
            Rn = dR @ R[i]
            U, _, Vt = np.linalg.svd(Rn)                   # re-orthonormalize (float drift)
            Rn = U @ Vt
            if np.linalg.det(Rn) < 0:
                Rn = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            R_new[i] = Rn
            frames_touched += 1
            Xn = s * (V @ R_new[i].T) + T[i]
            disp_sum += float(np.linalg.norm(Xn - X, axis=1).mean())
            disp_frames += 1
        data["object_rotation"] = R_new
        np.savez(a.out / src.name, **data)
        for ext in ("_object.glb", "_object.ply"):
            p = a.cand / f"{tag}{ext}"
            if p.exists():
                shutil.copy2(p, a.out / p.name)
        rep[tag] = {"frames_touched": frames_touched, "frames_failed": frames_failed,
                    "mean_surface_disp_cm": round(disp_sum / max(disp_frames, 1) * 100, 3),
                    "sec": round(time.time() - t0, 1)}
        print(f"{tag}: rotated {frames_touched} frames ({frames_failed} unresolved), "
              f"mean surface disp {rep[tag]['mean_surface_disp_cm']} cm, {rep[tag]['sec']}s", flush=True)
    (a.out / "pen_zero_rot_report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
