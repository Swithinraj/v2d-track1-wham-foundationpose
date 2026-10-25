#!/usr/bin/env python3
"""GT-free per-frame depth cues for the learned depth fusion (host, v2d-host venv: numpy + PIL). Each cue estimates the
relative depth error of OUR human, log(true distance / our distance), from evidence independent of our depth:

  c_moge   MoGe scene depth at our torso joints' pixels (5x5 median, decoded metres = 65535 / v - 1, divided by the
           episode's depth scale = our frame) over our joints' depth: log of the median ratio over 11 torso joints
           (biased by the body's half-thickness; the model sees it standardised). n_moge = joints used
  c_size   apparent size: our projected bone lengths over ViTPose's (12 bones, both ends confidence > 0.5): if our body
           is too close its projection is too large, so log(l_ours / l_vitpose) ~ log(z_true / z_ours). n_size = bones
  c_floor  floor contact: the lower ball of foot in stance (< 1 cm/frame) vs the floor (orientation from MoGe scene depth
           (RANSAC within 25 deg of our body-up, body pixels removed), height from our own median stance ball height,
           so the cue is zero-mean per episode); the slide along that foot's camera ray that puts it at the median height,
           over our distance. st_floor = 1 on stance frames (0 elsewhere, cue 0)
  *_rel    c_moge / c_size relative to their median over the 11 frames around the anchor (the scorer aligns there)
Inputs per episode: OUR joints (a joint cache's 'ours' [T, 22, 3], our camera frame) and the root's outputs
(camera/<tag>/intrinsics.json, object/<tag>/depth, sam3d_body/<tag>/prepare.npz (or --scale_root), kp2d_vpx/<tag>.npz).
Writes OUT/<name>.npz. Usage: depth_cues.py --joints DIR --prefix NAME --root ROOT_DIR --out DIR [--scale_root DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

TORSO = [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17]                       # scored-joint indices (pelvis .. shoulders)
SC2MP = {16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28}
BONES = [(16, 17), (1, 2), (16, 18), (17, 19), (18, 20), (19, 21), (1, 4), (2, 5), (4, 7), (5, 8), (16, 1), (17, 2)]
BALL, ANK, HEAD = (10, 11), (7, 8), 15
H0 = 0.038                                                             # stance ball-of-foot height above the floor (m)


def proj(X, K):
    return np.stack([K["fx"] * X[..., 0] / X[..., 2] + K["cx"], K["fy"] * X[..., 1] / X[..., 2] + K["cy"]], -1)


def moge_floor_normal(J, depth_dir, scale, K, up, rng, n_frames=12, step=6):
    pts = []
    for f in np.linspace(0, len(J) - 1, n_frames).astype(int):
        dp = depth_dir / f"{f:06d}.png"
        if not dp.exists():
            continue
        D = np.asarray(Image.open(dp)).astype(float)[::step, ::step]; vv, uu = np.mgrid[0:D.shape[0], 0:D.shape[1]] * step
        ok = D > 0; z = (65535.0 / np.where(ok, D, 1) - 1.0) / scale
        X = np.stack([(uu - K["cx"]) / K["fx"] * z, (vv - K["cy"]) / K["fy"] * z, z], -1)[ok & (z > 0.3) & (z < 12)]
        X = X[np.min(np.linalg.norm(X[:, None] - J[f][None], axis=-1), 1) > 0.35]
        pts.append(X)
    P = np.concatenate(pts); P = P[rng.choice(len(P), min(len(P), 60000), replace=False)]
    best, bn, bd = 0, up, 0.0
    for _ in range(600):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.radians(25)):
            continue
        d = -n @ a; inl = np.abs(P @ n + d) < 0.03
        if inl.sum() > best and np.median(J[:, list(BALL)].reshape(-1, 3) @ n + d) > -0.15:
            best, bn, bd = inl.sum(), n, d
    Q = P[np.abs(P @ bn + bd) < 0.03]; c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); n = Vt[-1]
    return n if n @ up > 0 else -n


def episode(J, tag, root, scale_root, anchor, lock=None):
    """lock: a background-locking result (unshipped experiment) (dict with 'a' [T] per-frame MoGe scale and 'n_master' floor normal) -> the MoGe
    cue uses the background-locked depth z_t / a_t and the floor cue the master map's normal; None -> per-frame MoGe."""
    rng = np.random.default_rng(int(tag.split("_")[-1]))                       # per-episode seed: order-independent
    T = len(J); K = json.loads((root / "outputs/camera" / tag / "intrinsics.json").read_text())
    scale = float(np.load(scale_root / "outputs/sam3d_body" / tag / "prepare.npz")["depth_scale"])
    dd = root / "outputs/object" / tag / "depth"
    zc = np.linalg.norm(J.mean(1), axis=1)                                         # our distance (centroid)
    # MoGe at torso joints
    c_m = np.full(T, np.nan); n_m = np.zeros(T)
    uv = proj(J[:, TORSO], K)
    for f in range(T):
        dp = dd / f"{f:06d}.png"
        if not dp.exists():
            continue
        D = np.asarray(Image.open(dp)); H, W = D.shape; r = []
        for j, (u, v) in enumerate(uv[f]):
            ui, vi = int(round(u)), int(round(v))
            if 2 <= ui < W - 2 and 2 <= vi < H - 2:
                w = D[vi - 2:vi + 3, ui - 2:ui + 3].astype(float); w = w[w > 0]
                if len(w) >= 5:
                    zm = (65535.0 / np.median(w) - 1.0) / scale
                    if zm > 0.2:
                        r.append(zm / J[f, TORSO[j], 2])
        if len(r) >= 3:
            c_m[f] = np.log(np.median(r)) - (np.log(float(lock["a"][f])) if lock is not None else 0.0); n_m[f] = len(r)
    # apparent size vs ViTPose
    kp = np.load(root / "outputs/kp2d_vpx" / f"{tag}.npz")["pose"][:, 0].astype(float)    # [T', 33, 5]: x, y, z, vis, conf
    c_s = np.full(T, np.nan); n_s = np.zeros(T); uvo = proj(J, K)
    for f in range(min(T, len(kp))):
        r = []
        for a, b in BONES:
            pa, pb = kp[f, SC2MP[a]], kp[f, SC2MP[b]]
            if min(pa[3], pb[3]) > 0.5:
                lv = np.linalg.norm(pa[:2] - pb[:2]); lo = np.linalg.norm(uvo[f, a] - uvo[f, b])
                if lv > 8 and lo > 8:
                    r.append(np.log(lo / lv))
        if len(r) >= 3:
            c_s[f] = np.median(r); n_s[f] = len(r)
    # floor contact (MoGe orientation, own median stance height)
    up = (J[:, HEAD] - J[:, list(ANK)].mean(1)).mean(0); up /= np.linalg.norm(up)
    use_master = lock is not None and float(lock.get("floor_inliers", 0.0)) >= 0.10       # a weakly supported master plane -> per-frame fit
    n = np.asarray(lock["n_master"], float) if use_master else moge_floor_normal(J, dd, scale, K, up, rng)
    B = J[:, list(BALL)]; hb = B @ n; low = hb.argmin(1)
    v = np.r_[np.zeros((1, 2)), np.linalg.norm(np.diff(B, axis=0), axis=-1)]
    st = v[np.arange(T), low] < 0.01
    hs = hb[np.arange(T), low]; d0 = H0 - np.median(hs[st]) if st.any() else 0.0
    rf = B[np.arange(T), low] / np.linalg.norm(B[np.arange(T), low], axis=1, keepdims=True); rn = rf @ n
    slide = (H0 - (hs + d0)) / np.where(rn < 0, np.minimum(rn, -0.15), np.maximum(rn, 0.15))
    c_f = np.where(st, slide / zc, 0.0)
    near = slice(max(0, anchor - 5), anchor + 6)
    rel = lambda c: c - (np.nanmedian(c[near]) if np.isfinite(c[near]).any() else np.nanmedian(c))  # noqa: E731
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    return dict(c_moge=f32(c_m), n_moge=f32(n_m), c_size=f32(c_s), n_size=f32(n_s), c_floor=f32(c_f), st_floor=f32(st),
                c_moge_rel=f32(rel(c_m)), c_size_rel=f32(rel(c_s)), floor_normal=f32(n), depth_scale=scale, floor_from_master=float(use_master))


FOOT = {"l": 10, "r": 11}                                              # l_ball / r_ball (scored-joint indices)


def foot_bg(J, lock, K, scale, anchor):
    """Floor evidence without any fitted plane: the background-locked master depth map (unshipped experiment: one static
    scene map per episode, every frame aligned to it, people / objects removed by the temporal median) read at each
    foot's pixel (3x3 median at the master's resolution), over our foot's depth: log(z_background / z_foot). While a
    foot is planted, its pixel's background IS the floor where it stands; the model learns when that holds (both feet's
    cues, validity flags and foot speeds), so no stance rule, plane fit or fallback is hand-set.
    -> dict: c_bg_{l,r} [T] (NaN where unavailable), c_bg_{l,r}_rel (vs the median of the 11 frames around the anchor),
    vb_{l,r} validity, sp_{l,r} foot speed (cm / frame)."""
    T = len(J); M = np.asarray(lock["master"], np.float32); st = int(lock["step"]); h, w = M.shape; out = {}
    near = slice(max(0, anchor - 5), anchor + 6)
    for s, j in FOOT.items():
        X = J[:, j]; uv = proj(X, K); c = np.full(T, np.nan)
        for f in range(T):
            ui, vi = int(round(uv[f, 0] / st)), int(round(uv[f, 1] / st))
            if 1 <= ui < w - 1 and 1 <= vi < h - 1 and X[f, 2] > 0.2:
                zb = np.nanmedian(M[vi - 1:vi + 2, ui - 1:ui + 2]) / scale
                if np.isfinite(zb) and zb > 0.2:
                    c[f] = np.log(zb / X[f, 2])
        out[f"c_bg_{s}"] = c.astype(np.float32); out[f"vb_{s}"] = np.isfinite(c).astype(np.float32)
        out[f"c_bg_{s}_rel"] = (c - (np.nanmedian(c[near]) if np.isfinite(c[near]).any() else np.nanmedian(c))).astype(np.float32)
        out[f"sp_{s}"] = (np.r_[0.0, np.linalg.norm(np.diff(X, axis=0), axis=1)] * 100).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("joints", "root", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--scale_root", type=Path)
    ap.add_argument("--lock", type=Path, help="dir of background-locking results (<prefix>_<tag>.npz; unshipped experiment)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for jp in sorted(a.joints.glob(f"{a.prefix}_episode_*.npz")):
        name = jp.stem; tag = name[len(a.prefix) + 1:]
        if (a.out / f"{name}.npz").exists():
            continue
        h = np.load(jp); J = h["ours"].astype(float); anchor = int(h["anchor"])
        lk = dict(np.load(a.lock / f"{name}.npz")) if a.lock else None
        if lk is not None:
            assert len(lk["a"]) == len(J), f"{name}: lock length {len(lk['a'])} != {len(J)}"
        r = episode(J, tag, a.root, a.scale_root or a.root, anchor, lk)
        np.savez(a.out / f"{name}.npz", **r)
        print(f"{name}: T {len(J)} | moge valid {np.isfinite(r['c_moge']).mean() * 100:.0f} % (median ratio {np.exp(np.nanmedian(r['c_moge'])):.3f}) | "
              f"size valid {np.isfinite(r['c_size']).mean() * 100:.0f} % (median {np.exp(np.nanmedian(r['c_size'])):.3f}) | stance {r['st_floor'].mean() * 100:.0f} %", flush=True)


if __name__ == "__main__":
    main()
