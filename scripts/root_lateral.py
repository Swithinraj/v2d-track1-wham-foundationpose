#!/usr/bin/env python3
"""Lateral (image-plane) root drift of the human from MediaPipe 2D, fixed-interval RTS smoother
(submission venv, CPU; last step of the human refinement stage).

WHAM/MHR gives smooth relative motion but can drift sideways relative to the camera ray. Per episode, in the
camera frame:
  state s_t = [u_x, u_y, v_x, v_y] (lateral body offset in metres and its velocity), constant-velocity model
  with spectral density Q;
  observations: every matched MediaPipe joint j (wrists, elbows, shoulders, hips, knees, ankles, visibility
  >= 0.5) measures u directly, because a lateral shift moves its projection exactly by f * u / Z_j:
  m_j = (z_j - pi(c_j)) * Z_j / f - bias_j, noise SIGMA_PX * Z_j / f; bias_j = the global median of that
  joint's offset over all episodes (MediaPipe landmarks are not MHR joint centres); chi-square gating.
  Forward Kalman filter + backward RTS pass (no phase lag).
WHAM's ViTPose keypoints are NOT used by the filter and serve as the independent check: an episode is
corrected only if its ViTPose error also drops by > 5 %. The correction (R_cw^T [u_x, u_y, 0] / depth_scale,
a pure per-frame root translation) is measured on --base (the body before refitting) and added to the root
of --refined, rounded through float32 like a stored pose.
Writes <out>/episode_*.npz (+ meshes) and root_lateral_report.json.
Usage: root_lateral.py --bodies DIR --mediapipe DIR --vitpose DIR --base DIR --refined DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import human_refine_common as M
from human_refine_common import FLIP

Q = 1e-10                                     # m^2/frame^3: a smoothing time constant of about 1.3 s
SIGMA_PX, GATE = 12.0, 9.21                   # chi-square 99 % for 2 dof
MP_PAIRS = [p for g in ("wrist", "elbow", "shoulder", "hip", "knee", "ankle") for p in M.PAIRS[g]]
VIT_PAIRS = [(75, 5), (39, 6), (76, 7), (40, 8), (78, 9), (42, 10), (2, 11), (18, 12), (3, 13), (19, 14), (4, 15), (20, 16)]


camera = M.camera


def offsets(c, obs, pairs, k, min_conf):
    """[T, P, 2] metric lateral offsets (metres, camera frame) and [T, P] metric noise; NaN where missing."""
    T = len(c)
    m, sd = np.full((T, len(pairs), 2), np.nan), np.full((T, len(pairs)), np.nan)
    f = np.array([k["fx"], k["fy"]]); cc = np.array([k["cx"], k["cy"]])
    for p, (j, l) in enumerate(pairs):
        Z = c[:, j, 2]
        uv = f * c[:, j, :2] / Z[:, None] + cc
        ok = np.isfinite(obs[:T, l, 0]) & (obs[:T, l, 2] >= min_conf)
        m[ok, p] = (obs[ok, l, :2] - uv[ok]) * Z[ok, None] / f
        sd[ok, p] = SIGMA_PX * Z[ok] / k["fx"]
    return m, sd


def rts(m, sd, q):
    """Fixed-interval RTS smoother of a 2D constant-velocity state from per-joint direct measurements."""
    T = len(m)
    F = np.block([[np.eye(2), np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    Q = q * np.block([[np.eye(2) / 3, np.eye(2) / 2], [np.eye(2) / 2, np.eye(2)]])
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    x, P = np.zeros(4), np.diag([1.0, 1.0, 1e-2, 1e-2])
    xs_p, Ps_p, xs_f, Ps_f, used = [], [], [], [], 0
    for t in range(T):
        if t:
            x, P = F @ x, F @ P @ F.T + Q
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        for p in np.flatnonzero(np.isfinite(m[t, :, 0])):
            Rm = np.eye(2) * sd[t, p] ** 2
            S = H @ P @ H.T + Rm
            r = m[t, p] - H @ x
            if r @ np.linalg.solve(S, r) > GATE:
                continue
            K = P @ H.T @ np.linalg.inv(S)
            x, P = x + K @ r, (np.eye(4) - K @ H) @ P
            used += 1
        xs_f.append(x.copy()); Ps_f.append(P.copy())
    xs = [xs_f[-1]]
    for t in range(T - 2, -1, -1):
        G = Ps_f[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs.append(xs_f[t] + G @ (xs[-1] - xs_p[t + 1]))
    return np.array(xs[::-1])[:, :2], used


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bodies", type=Path, required=True, help="mhr_bodies.py output of --base")
    ap.add_argument("--mediapipe", type=Path, required=True)
    ap.add_argument("--vitpose", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="episode dir the correction is measured on")
    ap.add_argument("--refined", type=Path, required=True, help="episode dir the correction is added to")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scored = {str(k): v for k, v in M.scored_and_budget()[0].items()}
    data = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        a = M.alignment(tag)
        J = np.load(args.bodies / f"{tag}.npz")["joints"].astype(np.float64)
        c = camera(J, a)
        mp = np.load(args.mediapipe / f"{tag}.npz")["pose"].astype(np.float64)
        T = min(len(c), len(mp))
        actor = M.actor_landmarks(mp, M.project(J, a))
        vit = np.load(args.vitpose / f"{tag}.npz")["kps"].astype(np.float64)
        data[ep] = (a, c[:T], actor, vit[:T])
    # global per-joint definitional bias (metres, camera frame), MediaPipe and ViTPose separately
    bias_mp = np.nanmedian(np.concatenate([offsets(c, mp, MP_PAIRS, a["intrinsics"], 0.5)[0] for a, c, mp, _ in data.values()]), axis=0)
    bias_vit = np.nanmedian(np.concatenate([offsets(c, v, VIT_PAIRS, a["intrinsics"], 0.5)[0] for a, c, _, v in data.values()]), axis=0)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        d = dict(np.load(args.base / f"{tag}.npz"))
        a, c, mp, vit = data[ep]
        k = a["intrinsics"]
        T = len(c)
        sf = np.array([f for f in scored[str(ep)] if f < T])
        m, sd = offsets(c, mp, MP_PAIRS, k, 0.5)
        u, used = rts(m - bias_mp[None], sd, Q)
        mv, _ = offsets(c, vit, VIT_PAIRS, k, 0.5)
        vit_before = float(np.nanmedian(np.linalg.norm((mv - bias_vit[None])[sf], axis=-1)) * 100)
        vit_after = float(np.nanmedian(np.linalg.norm((mv - bias_vit[None] - u[:, None])[sf], axis=-1)) * 100)
        mp_before = float(np.nanmedian(np.linalg.norm((m - bias_mp[None])[sf], axis=-1)) * 100)
        mp_after = float(np.nanmedian(np.linalg.norm((m - bias_mp[None] - u[:, None])[sf], axis=-1)) * 100)
        Tp = len(d["pose"])
        u_full = np.zeros((Tp, 2)); u_full[:T] = u; u_full[T:] = u[-1]
        dw = (np.concatenate([u_full, np.zeros((Tp, 1))], 1) @ np.asarray(a["R_cw"])) / float(a["depth_scale"])   # R^T u / ds
        passed = vit_after < 0.95 * vit_before
        refined = dict(np.load(args.refined / f"{tag}.npz"))
        if passed:
            base_root = d["pose"][:, :3].astype(np.float64)
            shifted = (d["pose"].astype(np.float64) + np.concatenate([FLIP * dw * 10.0, np.zeros((Tp, d["pose"].shape[1] - 3))], 1)).astype(np.float32)
            delta = shifted[:, :3].astype(np.float64) - base_root
            if np.abs(delta).max() > 0:
                pose = refined["pose"].astype(np.float64)
                pose[:, :3] += delta
                refined["pose"] = pose.astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **refined)
        for mesh in args.refined.glob(f"{tag}_object.*"):
            shutil.copy(mesh, args.out / mesh.name)
        inner = sf[(sf >= 1) & (sf < Tp - 1)]
        acc = np.linalg.norm(dw[inner - 1] - 2 * dw[inner] + dw[inner + 1], axis=1) * 100
        mag = np.linalg.norm(u, axis=1) * 100
        report[tag] = {"applied": bool(passed), "measurements_used": used, "u_median_cm": float(np.median(mag[sf])),
                       "u_range_cm": float(np.ptp(u[sf], axis=0).max() * 100),
                       "anchor_shift_cm": float(np.linalg.norm(dw[sf[0]]) * 100), "acc_cm": float(acc.mean()),
                       "mediapipe_cm": [mp_before, mp_after], "vitpose_cm": [vit_before, vit_after]}
        r = report[tag]
        print(f"{tag} {'APPLY' if passed else 'skip '} |u| med {r['u_median_cm']:5.1f} cm range {r['u_range_cm']:5.1f} | "
              f"MediaPipe {mp_before:4.1f}->{mp_after:4.1f} cm | ViTPose {vit_before:4.1f}->{vit_after:4.1f} cm", flush=True)
    (args.out / "root_lateral_report.json").write_text(json.dumps({"q": Q, "bias_mp_cm": (bias_mp * 100).round(2).tolist(),
                                                                  "bias_vit_cm": (bias_vit * 100).round(2).tolist(), "episodes": report}, indent=2))
    applied = [k for k, r in report.items() if r["applied"]]
    print(f"root lateral: applied to {len(applied)} episodes {[int(k[-2:]) for k in applied]}")

if __name__ == "__main__":
    main()
