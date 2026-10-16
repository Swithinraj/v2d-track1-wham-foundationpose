#!/usr/bin/env python3
"""Per-hand wrist correction by sensor fusion: MHR process + MediaPipe 2D + object anchors, RTS smoother
(submission venv, CPU; human refinement stage).

Process model: the smoothed (lambda 10) MHR wrist is right up to a slowly varying 3D offset d_t, kept in the CAMERA
frame (metres), constant-velocity model with spectral density Q_D laterally and Q_D * DEPTH_Q along the
ray, with a 1 cm prior on depth: a 2D detector cannot see depth (a first version that let the perspective
term observe it explained 10 px residuals with 30-117 cm moves along the ray). Observations:
  - MediaPipe wrist landmark (left 15 -> MHR 78, right 16 -> MHR 42), visibility >= 0.5: lateral only,
    z - pi(c_t) = f / Z * (d_x, d_y), minus the wrist's global median 2D bias (metric, per joint), noise
    SIGMA_PX, residuals > 40 px dropped as wrong-hand detections, chi-square gated (MediaPipe is ~9x
    jerkier than MHR, so it only steers the low frequencies); final offsets capped at 10 cm; no 2D update
    while the hand is within 5 cm of the object, where a zero-offset observation (2 cm) keeps MHR's hand
    unless an object anchor applies (contact guard: a first version pushed a hand into the cart, ep 27);
  - inside stable grasp windows (hand_anchors.py): the object-anchored hand centroid minus the MHR hand centroid,
    a direct 3D observation of d (mapped to the camera frame) with noise SIGMA_ANCHOR (FoundationPose is
    the smooth, trusted track there; the only source that moves the depth component).
Forward Kalman filter + backward RTS pass. Writes <out>/episode_XXXXXX.npz (offsets [T, 2, 3], hand order
0 = right/joint 42, 1 = left/joint 78) + fusion_report.json; human_refit.py --fusion applies them.
Usage: wrist_fusion.py --bodies DIR --episodes DIR --anchors DIR --mediapipe DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import human_refine_common as C
from human_refine_common import actor_landmarks, camera

Q_D = 1e-9                                 # m^2/frame^3: a smoothing time constant of about a second
SIGMA_PX, SIGMA_ANCHOR, GATE, DEPTH_Q = 15.0, 0.01, 9.21, 1e-2
MAX_RESID_PX, MAX_OFFSET_M = 40.0, 0.10    # MediaPipe agrees with MHR wrists to ~11 px median; > 40 px is a wrong hand
CONTACT_M, SIGMA_CONTACT = 0.05, 0.02      # contact guard: no 2D update near the object; pull the offset to 0 there
HANDS = ((42, 16), (78, 15))             # (MHR wrist joint, MediaPipe wrist landmark): index 0 = right, 1 = left
SPLIT = 2318


def rts3(obs, q):
    """obs: list per frame of (H [k,3], r [k], Rm [k,k]) linear observations of d; CV model in 3D."""
    T = len(obs)
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    F = np.block([[I3, I3], [Z3, I3]])
    Dq = np.diag([1.0, 1.0, DEPTH_Q])
    Q = q * np.block([[Dq / 3, Dq / 2], [Dq / 2, Dq]])
    x, P = np.zeros(6), np.diag([1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 1e-6])
    xp, Pp, xf, Pf, used = [], [], [], [], 0
    for t in range(T):
        if t:
            x, P = F @ x, F @ P @ F.T + Q
        xp.append(x.copy()); Pp.append(P.copy())
        for H3, r, Rm in obs[t]:
            H = np.hstack([H3, np.zeros((len(r), 3))])
            S = H @ P @ H.T + Rm
            e = r - H @ x
            if e @ np.linalg.solve(S, e) > (GATE if len(r) == 2 else 11.34):
                continue
            K = P @ H.T @ np.linalg.inv(S)
            x, P = x + K @ e, (np.eye(6) - K @ H) @ P
            used += 1
        xf.append(x.copy()); Pf.append(P.copy())
    xs = [xf[-1]]
    for t in range(T - 2, -1, -1):
        G = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs.append(xf[t] + G @ (xs[-1] - xp[t + 1]))
    return np.array(xs[::-1])[:, :3], used


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bodies", type=Path, required=True, help="mhr_bodies.py output of the process-model refit")
    ap.add_argument("--episodes", type=Path, required=True, help="episode dir with the object tracks")
    ap.add_argument("--anchors", type=Path, required=True)
    ap.add_argument("--mediapipe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    q = Q_D
    scored = {str(k): v for k, v in C.scored_and_budget()[0].items()}
    data = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        a = C.alignment(tag)
        z = np.load(args.bodies / f"{tag}.npz")
        J, Hd = z["joints"].astype(np.float64), z["hands"].astype(np.float64)
        mp = actor_landmarks(np.load(args.mediapipe / f"{tag}.npz")["pose"].astype(np.float64), C.project(J, a))
        data[ep] = (a, J, Hd, mp)
    # global metric 2D bias per wrist (MediaPipe landmark vs MHR joint), camera frame
    bias = []
    for h, (j, l) in enumerate(HANDS):
        res = []
        for a, J, _, mp in data.values():
            k, c = a["intrinsics"], camera(J[:len(mp)], a)
            uv = np.stack([k["fx"] * c[:, j, 0] / c[:, j, 2] + k["cx"], k["fy"] * c[:, j, 1] / c[:, j, 2] + k["cy"]], 1)
            ok = np.isfinite(mp[:, l, 0]) & (mp[:, l, 2] >= 0.5)
            res.append((mp[ok, l, :2] - uv[ok]) * c[ok, j, 2:3] / np.array([k["fx"], k["fy"]]))
        bias.append(np.median(np.concatenate(res), axis=0))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ep, (a, J, Hd, mp) in data.items():
        tag = f"episode_{ep:06d}"
        k, R, ds = a["intrinsics"], np.asarray(a["R_cw"]), float(a["depth_scale"])
        T = len(J)
        c = camera(J, a)
        anchors = np.load(args.anchors / f"{tag}.npz")
        rows = json.loads(str(anchors["meta"]))
        epd = np.load(args.episodes / f"{tag}.npz")
        Ro, to = epd["object_rotation"].astype(np.float64), epd["object_translation"].astype(np.float64)
        offsets, stats = np.zeros((T, 2, 3)), []
        frames_b, budget = C.scored_and_budget()
        vv, _ = C.budget_mesh(args.episodes / f"{tag}_object.glb", *budget)
        rng = np.random.default_rng(0)
        vsub = np.asarray(vv, float)[rng.choice(len(vv), min(3000, len(vv)), replace=False)]
        s_obj = float(epd["object_scale"])
        near = np.zeros((T, 2), bool)
        for h in range(2):
            pts = Hd[:, slice(0, SPLIT) if h == 0 else slice(SPLIT, None)][:, ::18]
            for f in range(T):
                ov = s_obj * vsub @ Ro[f].T + to[f]
                near[f, h] = np.linalg.norm(pts[f][:, None] - ov[None], axis=-1).min() < CONTACT_M
        for h, (j, l) in enumerate(HANDS):
            obs = [[] for _ in range(T)]
            for f in range(min(T, len(mp))):
                if not (np.isfinite(mp[f, l, 0]) and mp[f, l, 2] >= 0.5) or near[f, h]:
                    continue
                X, Y, Z = c[f, j]
                Jp = np.array([[k["fx"] / Z, 0.0, 0.0], [0.0, k["fy"] / Z, 0.0]])
                uv = np.array([k["fx"] * X / Z + k["cx"], k["fy"] * Y / Z + k["cy"]])
                r = mp[f, l, :2] - uv - bias[h] * np.array([k["fx"], k["fy"]]) / Z
                if np.linalg.norm(r) <= MAX_RESID_PX:
                    obs[f].append((Jp, r, np.eye(2) * SIGMA_PX ** 2))
            n_anchor = 0
            for wi, w in enumerate(rows):
                if w["hand"] != h:
                    continue
                qbar = anchors[f"w{wi}"].astype(np.float64)
                sl = slice(0, SPLIT) if h == 0 else slice(SPLIT, None)
                for f in range(w["start"], w["end"]):
                    target = (Ro[f] @ qbar.T).T.mean(0) + to[f]
                    obs[f].append((np.eye(3), ds * R @ (target - Hd[f, sl].mean(0)), np.eye(3) * (ds * SIGMA_ANCHOR) ** 2))
                    n_anchor += 1
            for f in range(T):                                       # contact frames without an anchor: keep MHR's hand
                if near[f, h] and not any(len(r) == 3 for _, r, _ in obs[f]):
                    obs[f].append((np.eye(3), np.zeros(3), np.eye(3) * (ds * SIGMA_CONTACT) ** 2))
            d_cam, used = rts3(obs, q)
            d = d_cam @ R / ds                                           # camera -> world: R^T d / ds
            n = np.linalg.norm(d, axis=1, keepdims=True)
            d = d * np.minimum(1.0, MAX_OFFSET_M / np.maximum(n, 1e-12))       # safety cap
            offsets[:, h] = d
            sf = np.array([f for f in scored[str(ep)] if 1 <= f < T - 1])
            stats.append({"hand": h, "updates": used, "anchor_obs": n_anchor, "contact_frames": int(near[:, h].sum()), "offset_median_cm": float(np.median(np.linalg.norm(d[sf], axis=1)) * 100),
                          "offset_acc_cm": float((np.linalg.norm(d[sf - 1] - 2 * d[sf] + d[sf + 1], axis=1) * 100).mean())})
        np.savez(out / f"{tag}.npz", offsets=offsets.astype(np.float32))
        report[tag] = stats
        print(tag, " | ".join(f"hand {s['hand']}: |d| {s['offset_median_cm']:.1f} cm, acc {s['offset_acc_cm']:.4f}, anchors {s['anchor_obs']}" for s in stats), flush=True)
    (out / "fusion_report.json").write_text(json.dumps({"q": q, "bias_cm": (np.array(bias) * 100).round(2).tolist(), "episodes": report}, indent=2))


if __name__ == "__main__":
    main()
