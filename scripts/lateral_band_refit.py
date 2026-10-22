#!/usr/bin/env python3
"""Restore real image-plane motion that the lambda-300 refit removes (mhr_converter.sif, GPU; before the contact stage).

Evidence: triple collocation of RAW SAM 3D Body / ViTPose / MediaPipe 2D
accelerations shows real visible motion at 1-2 Hz (raw SNR 4-30) that lambda 300 cuts at ~1 Hz, while the Track 1 GT
(4-view SAM fusion with a first-order prior) keeps it. Depth stays as it is (depth content above ~1 Hz is mostly noise).
Per vertex, in perspective image coordinates (u = x/z, v = y/z; the Track 1 frame is the camera frame):
  variant band (B):    target = IN + w * [S_lat(RAW uv) - S_300(RAW uv)] * z_IN      (re-adds only the removed band)
  variant replace (A): target = IN + w * [S_lat(RAW uv) - IN uv] * z_IN              (lateral = smoothed RAW)
with S_lam the Whittaker smoother, IN = the pre-contact human (depth kept), RAW = per-frame SAM 3D Body (refine input).
w = per-frame, per-part evidence weight (no constants per episode, no MediaPipe in the build):
  q = clip((ViTPose confidence - 0.5) / 0.3, 0, 1) * exp(-0.5 (d / (3 * 1.4826 MAD(d)))^2) * vis, d = |RAW 2D - ViTPose 2D| after
  removing its per-episode median (keypoint-convention offset); vis = the projected joint lies inside the actor's SAM 2
  mask (dilated) and outside the object's SAM 2 mask (mask_visibility.py --visibility); Gaussian-smoothed (sigma 2
  frames), squared;
  parts: wrists, elbows, shoulders (upper arms), hips; torso/head = mean of shoulders and hips; legs 0 (no reliable evidence).
  The first scored frame is pinned: w tapers to 0 over +-RAMP frames and the anchor pose row is copied exactly.
Then each frame's pose is re-fitted to the targets (kit lm_pose, identity fixed). Objects unchanged.
Writes OUT/episode_*.npz (+ meshes), OUT_bodies/<ep>.npz (joints of the result) and lateral_band_report.json.
Usage (container): lateral_band_refit.py --input DIR --raw DIR --visibility DIR --out DIR [--mode band|replace] [--lam_lat 100]
  --raw: the refinement's unsmoothed input human (per-frame SAM 3D Body); --visibility: mask_visibility.py output of --raw's bodies
"""
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
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_to_mhr_params import MHR, lm_pose  # noqa: E402
from mhr_bodies import MODEL  # noqa: E402
from human_refit import vertex_parts  # noqa: E402

PARTS = [75, 39, 76, 40, 78, 42, 2, 18, 3, 19, 4, 20]          # shoulder, elbow, wrist, hip, knee, ankle (L, R)
LMK = {75: 11, 39: 12, 76: 13, 40: 14, 78: 15, 42: 16, 2: 23, 18: 24}   # part joint -> MediaPipe-layout landmark (kp2d_vpx)
RAMP = 10


def whittaker_op(T, lam, dev):
    D = torch.zeros(T - 2, T, dtype=torch.float64, device=dev); i = torch.arange(T - 2, device=dev)
    D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return torch.linalg.cholesky(torch.eye(T, dtype=torch.float64, device=dev) + lam * D.T @ D)


def smooth(L, Y):
    sh = Y.shape; return torch.cholesky_solve(Y.reshape(sh[0], -1), L).reshape(sh)


def gauss(x, s):
    r = int(np.ceil(4 * s)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def part_weights(tag, J_raw, T, anchor, vis_dir):
    a = json.loads((ROOT / f"outputs/alignment_sam/{tag}/alignment.json").read_text())["intrinsics"]
    kp = np.load(ROOT / f"outputs/kp2d_vpx/{tag}.npz")["pose"][:, 0].astype(float)          # [Tk, 33, 5]
    n = min(T, len(kp)); W = np.zeros((T, len(PARTS)))
    VIS = np.load(vis_dir / f'{tag}.npz')['vis']
    for k, j in enumerate(PARTS):
        if j not in LMK:
            continue
        l = LMK[j]; c = np.zeros(T); c[:n] = np.nan_to_num(kp[:n, l, 3])
        uv = np.stack([a["fx"] * J_raw[:, j, 0] / J_raw[:, j, 2] + a["cx"], a["fy"] * J_raw[:, j, 1] / J_raw[:, j, 2] + a["cy"]], -1)
        d = np.full(T, np.inf); ok = np.zeros(T, bool); ok[:n] = (c[:n] > 0.3) & np.isfinite(kp[:n, l, 0])
        if ok.sum() >= 10:
            diff = uv[:n][ok[:n]] - kp[:n][ok[:n], l, :2]; diff -= np.median(diff, 0)
            dd = np.linalg.norm(diff, axis=1); s = 3 * 1.4826 * np.median(np.abs(dd - np.median(dd))) + 1e-6
            d[np.flatnonzero(ok)] = dd; agree = np.exp(-0.5 * (np.minimum(d, 1e6) / s) ** 2)
        else:
            agree = np.zeros(T)
        q = np.clip((c - 0.5) / 0.3, 0, 1) * agree * VIS[:T, j]
        W[:, k] = np.clip(gauss(q, 2.0), 0, 1) ** 2
    m = np.arange(T); ramp = np.clip(np.abs(m - anchor) / RAMP, 0, 1); ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return W * ramp[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--mode", default="band", choices=["band", "replace"])
    ap.add_argument("--visibility", type=Path, required=True, help="mask_visibility.py output (vis [T, J])")
    ap.add_argument("--lam_lat", type=float, default=100.0); ap.add_argument("--lam_ref", type=float, default=300.0)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args(); dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True); bdir = Path(str(args.out) + "_bodies"); bdir.mkdir(exist_ok=True)
    for f in args.input.iterdir():
        if f.is_file() and not f.name.startswith("episode_"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads((ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json").read_text())
    mhr = MHR(str(MODEL), dev, chunk=128, precision="float64"); parts = vertex_parts(MODEL, np.array(PARTS))
    torso = torch.as_tensor(parts < 0, device=dev); rep = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"; d = dict(np.load(args.input / f"{tag}.npz")); r = np.load(args.raw / f"{tag}.npz")
        T = len(d["pose"]); anchor = min(f for f in S[str(ep)] if f < T)
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], dtype=torch.float64, device=dev)
        p_in = torch.as_tensor(d["pose"], dtype=torch.float64, device=dev)
        p_raw = torch.as_tensor(r["pose"][:T], dtype=torch.float64, device=dev)
        V_in, J_in = mhr.run(p_in, ident); V_raw, J_raw = mhr.run(p_raw, ident)                 # mm, m
        uv_in = V_in[..., :2] / V_in[..., 2:3]; uv_raw = V_raw[..., :2] / V_raw[..., 2:3]
        L_lat = whittaker_op(T, args.lam_lat, dev)
        if args.mode == "band":
            delta = smooth(L_lat, uv_raw) - smooth(whittaker_op(T, args.lam_ref, dev), uv_raw)
        else:
            delta = smooth(L_lat, uv_raw) - uv_in
        Wp = torch.as_tensor(part_weights(tag, J_raw.cpu().numpy(), T, anchor, args.visibility), device=dev)       # [T, 12]
        wv = torch.zeros(T, V_in.shape[1], dtype=torch.float64, device=dev)
        for k in range(len(PARTS)):
            idx = torch.as_tensor(np.flatnonzero(parts == k), device=dev)
            if idx.numel():
                wv[:, idx] = Wp[:, k:k + 1]
        wv[:, torso] = Wp[:, [0, 1, 6, 7]].mean(1, keepdim=True)
        X = V_in.clone(); X[..., :2] += wv[..., None] * delta * V_in[..., 2:3]
        pose1, err = lm_pose(mhr, X.reshape(T, -1), p_in.clone(), ident, iters=6)
        pose1[anchor] = p_in[anchor]
        V1, J1 = mhr.run(pose1, ident)
        d["pose"] = pose1.cpu().numpy().astype(np.float32); np.savez(args.out / f"{tag}.npz", **d)
        for mesh in args.input.glob(f"{tag}_object*"):
            shutil.copy2(mesh, args.out / mesh.name)
        np.savez(bdir / f"{tag}.npz", joints=J1.cpu().numpy().astype(np.float32))
        lat = ((J1[..., :2] / J1[..., 2:3] - J_in[..., :2] / J_in[..., 2:3]).norm(dim=-1) * J_in[..., 2]).nan_to_num()
        rep[tag] = {"weight_mean_upper": float(Wp[:, :8].mean()), "fit_err_mm": float(err.mean()),
                    "lateral_change_cm_mean": float(lat.mean() * 100), "depth_change_cm_mean": float((J1[..., 2] - J_in[..., 2]).abs().mean() * 100),
                    "anchor_change_cm": float((J1[anchor] - J_in[anchor]).norm(dim=-1).max() * 100)}
        print(tag, json.dumps(rep[tag]), flush=True)
    (args.out / f"lateral_band_report_{'_'.join(map(str, args.episodes))}.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "episodes": rep}, indent=1))


if __name__ == "__main__":
    main()
