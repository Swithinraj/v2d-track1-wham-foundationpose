#!/usr/bin/env python3
"""Background-locked MoGe depth for a static camera (host, v2d-host venv: numpy + PIL). MoGe predicts every frame
independently, so the whole scene's metric scale wobbles from frame to frame (measured: ~0.85 % std per frame on the
static background). With a static camera the background cannot change, so per episode:
  master Z(p)   per-pixel temporal median of the decoded depth (metres = 65535 / v - 1) after removing each frame's
                scale, iterated twice (moving people / objects fall out of the median)
  static mask   pixels within 10 % of a_t Z(p) in >= 80 % of frames
  a_t           per-frame scale: median over static pixels of z_t(p) / Z(p), normalised to median 1
  -> locked depth z_t(p) / a_t for EVERY pixel of frame t (people and objects included): the per-frame drift removed
  n_master      floor normal from the master map alone (our frame: metres / the episode's depth scale; RANSAC within 25 deg
                of our body-up from the joints, refined by least squares): one static estimate instead of per-frame ones
Writes OUT/<prefix>_<tag>.npz (a [T], static_share, master [H/8, W/8] float16, n_master [3]).
Usage: moge_lock.py --root ROOT_DIR --joints DIR --prefix NAME --out DIR [--scale_root DIR] [--step 8]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BALL, ANK, HEAD = (10, 11), (7, 8), 15


def lock(depth_dir, T, step):
    Z = []
    for f in range(T):
        D = np.asarray(Image.open(depth_dir / f"{f:06d}.png")).astype(np.float32)[::step, ::step]
        Z.append(np.where(D > 0, 65535.0 / np.where(D > 0, D, 1) - 1.0, np.nan))
    Z = np.stack(Z)                                                              # [T, h, w] metres (MoGe)
    a = np.ones(T, np.float32); master = np.nanmedian(Z, 0)
    for _ in range(2):
        rel = Z / master
        static = (np.nanmean(np.abs(rel / a[:, None, None] - 1) < 0.10, 0) >= 0.8) & np.isfinite(master)
        a = np.nanmedian(rel[:, static], 1).astype(np.float32); a /= np.nanmedian(a)
        master = np.nanmedian(Z / a[:, None, None], 0)
    rel = Z / master
    static = (np.nanmean(np.abs(rel / a[:, None, None] - 1) < 0.10, 0) >= 0.8) & np.isfinite(master)
    a = np.nanmedian(rel[:, static], 1).astype(np.float32); a /= np.nanmedian(a)
    return a, master, static


def floor_from_master(master, static, step, K, scale, up, J, rng):
    vv, uu = np.mgrid[0:master.shape[0], 0:master.shape[1]] * step
    z = master / scale; ok = static & np.isfinite(z) & (z > 0.3) & (z < 12)
    P = np.stack([(uu - K["cx"]) / K["fx"] * z, (vv - K["cy"]) / K["fy"] * z, z], -1)[ok]
    best, bn, bd = 0, up, 0.0
    for _ in range(800):
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
    return (n if n @ up > 0 else -n), best / max(len(P), 1)


def main():
    ap = argparse.ArgumentParser()
    for k in ("root", "joints", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--scale_root", type=Path); ap.add_argument("--step", type=int, default=8)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); sr = a.scale_root or a.root
    for jp in sorted(a.joints.glob(f"{a.prefix}_episode_*.npz")):
        name = jp.stem; tag = name[len(a.prefix) + 1:]
        if (a.out / f"{name}.npz").exists():
            continue
        J = np.load(jp)["ours"].astype(float); T = len(J)
        sc, ms, st = lock(a.root / "outputs/object" / tag / "depth", T, a.step)
        K = json.loads((a.root / "outputs/camera" / tag / "intrinsics.json").read_text())
        scale = float(np.load(sr / "outputs/sam3d_body" / tag / "prepare.npz")["depth_scale"])
        up = (J[:, HEAD] - J[:, list(ANK)].mean(1)).mean(0); up /= np.linalg.norm(up)
        n, inl = floor_from_master(ms, st, a.step, K, scale, up, J, np.random.default_rng(int(tag.split("_")[-1])))
        np.savez(a.out / f"{name}.npz", a=sc, static_share=float(st.mean()), master=ms.astype(np.float16), n_master=n.astype(np.float32), floor_inliers=inl, step=a.step)
        print(f"{name}: T {T} | per-frame scale std {np.std(sc) * 100:.2f} % (5-95 % range {(np.percentile(sc, 95) - np.percentile(sc, 5)) * 100:.1f} %) | static {st.mean() * 100:.0f} % | floor inliers {inl:.2f}", flush=True)


if __name__ == "__main__":
    main()
