#!/usr/bin/env python3
"""Stage 4 of the joint scene solve: offline interaction states per hand and frame (host venv, numpy).

States: 0 free, 1 contact (touching, sliding over the surface), 2 grasp (contact region keeps its place on the object).
Per frame and hand, from the current scene (dense MHR hand vertices + object pose/mesh + scorer-consistent SDF):
  dmin     smallest signed distance of the hand to the object surface (m)
  slide    mean motion (f-1 -> f+1) of the hand's near-surface vertices in the OBJECT frame (m/frame)
  carried  the object moves (> 2 mm/frame) while the hand touches it
  vis      MediaPipe hand confidence of the actor (occluded or unseen hands -> emissions flattened toward uniform)
Emission log-likelihoods are soft (logistic) in those quantities; transitions are sticky (stay 0.97); a forward-backward
pass over the WHOLE video gives posteriors, so a later unoccluded view resolves an earlier occluded one. Output per
episode: posteriors [T, 2, 3], argmax state, confidence (max posterior), visibility.
Writes <out>/<episode>.npz and summary.json. Usage: interaction_states.py OUT_DIR EPISODES_DIR BODIES_DIR SDF_DIR
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1]); R0 = ROOT

STAY = 0.97


def sdf_fn(g):
    """trilinear lookup in a contact_sdf.py grid (local mesh units)"""
    grid, lo, hi = g["sdf"].astype(np.float32), g["lo"].astype(float), g["hi"].astype(float); n = np.array(grid.shape)

    def f(P):
        x = np.clip((P - lo) / (hi - lo) * (n - 1), 0, n - 1.001); i = x.astype(int); w = x - i; out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ww = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1]) * (w[..., 2] if dz else 1 - w[..., 2])
                    out = out + ww * grid[i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz]
        return out
    return f


def logistic(x):
    return 1 / (1 + np.exp(-x))


def emissions(dmin, slide, carried, vis):
    """log p(obs | state) for free / contact / grasp, softened by visibility."""
    near = logistic((0.015 - dmin) / 0.004)          # ~1 when within 1.5 cm of the surface
    still = logistic((0.004 - slide) / 0.0015)       # ~1 when the contact region barely slides on the object
    p_free = (1 - near)
    p_contact = near * (1 - still) * (1 - 0.5 * carried)
    p_grasp = near * (0.6 * still + 0.4 * carried)
    P = np.stack([p_free, p_contact, p_grasp], -1) + 1e-3
    P = P / P.sum(-1, keepdims=True)
    w = 0.3 + 0.7 * vis[:, None]                     # unseen hand: emissions pulled toward uniform
    P = P ** w
    return np.log(P / P.sum(-1, keepdims=True))


def forward_backward(logE):
    T, K = logE.shape
    A = np.full((K, K), (1 - STAY) / (K - 1)); np.fill_diagonal(A, STAY); A[0, 2] = A[2, 0] = (1 - STAY) * 0.2
    A = A / A.sum(1, keepdims=True); lA = np.log(A)
    la = np.zeros((T, K)); lb = np.zeros((T, K)); la[0] = logE[0] - np.log(K)
    for t in range(1, T):
        la[t] = logE[t] + np.logaddexp.reduce(la[t - 1][:, None] + lA, axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = np.logaddexp.reduce(lA + (logE[t + 1] + lb[t + 1])[None], axis=1)
    lp = la + lb
    return np.exp(lp - np.logaddexp.reduce(lp, axis=1, keepdims=True))


def main():
    out, cdir, bdir, sdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True); summ = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"
        d = np.load(cdir / f"{tag}.npz"); H = np.load(bdir / f"{tag}.npz")["hands"].astype(float)
        R, t, s = d["object_rotation"].astype(float), d["object_translation"].astype(float), float(d["object_scale"])
        sdf = sdf_fn(np.load(sdir / f"{tag}.npz")); T = min(len(H), len(t)); half = H.shape[1] // 2
        mp = np.load(R0 / f"outputs/mediapipe/{tag}.npz"); hs = mp["hand_score"]           # [T, persons, 2] (side 0 = left)
        speed = np.r_[0, np.linalg.norm(np.diff(t[:T], axis=0), axis=1)]
        post = np.zeros((T, 2, 3)); vis_all = np.zeros((T, 2))
        for k in range(2):                                                                # k 0 = right hand, 1 = left
            P = H[:T, k * half:(k + 1) * half]
            loc = np.einsum("fvi,fij->fvj", P - t[:T, None], R[:T]) / s                 # object frame (mesh units)
            dist = sdf(loc) * s
            dist = np.nan_to_num(dist, nan=1.0); dmin = dist.min(1)
            nearm = np.abs(dist) < 0.02
            slide = np.full(T, 0.02)
            for f in range(1, T - 1):
                m = nearm[f]
                if m.sum() >= 10:
                    slide[f] = float(np.linalg.norm((loc[f + 1, m] - loc[f - 1, m]) * s, axis=1).mean() / 2)
            carried = ((speed > 0.002) & (dmin < 0.015)).astype(float)
            side = 1 - k                                                                  # MediaPipe side 0 = left = our k 1
            vis = np.clip(np.nan_to_num(np.nanmax(np.where(np.isfinite(hs[:T, :, side]), hs[:T, :, side], 0.0), axis=1), nan=0.0), 0, 1)
            vis_all[:, k] = vis
            post[:, k] = forward_backward(emissions(dmin, slide, carried, vis))
        state = post.argmax(-1); conf = post.max(-1)
        np.savez_compressed(out / f"{tag}.npz", posterior=post.astype(np.float32), state=state.astype(np.int8), conf=conf.astype(np.float32), vis=vis_all.astype(np.float32))
        sc = [f for f in S[str(ep)] if f < T]
        summ[tag] = {"frames_free/contact/grasp (any hand)": [int((state == 0).all(1).sum()), int(((state == 1).any(1) & ~(state == 2).any(1)).sum()), int((state == 2).any(1).sum())],
                     "low_confidence_frames": int((conf < 0.7).any(1).sum()), "grasp_frames_scored": int((state[sc] == 2).any(1).sum()), "T": int(T)}
        print(tag, summ[tag], flush=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))


S = json.loads((R0 / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
if __name__ == "__main__":
    main()
