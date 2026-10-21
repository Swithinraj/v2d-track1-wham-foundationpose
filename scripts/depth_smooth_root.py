#!/usr/bin/env python3
"""Anisotropic temporal smoothing of the human's camera depth (submission venv, CPU; before the contact solve).

A monocular body estimate is far noisier along the camera ray than across it (raw SAM 3D Body pelvis: depth residual
~5x the lateral one), yet the refit smooths every axis with the same LAM (300). Per episode:
  1. noise ratio from the RAW human (--raw bodies): residuals of the pelvis trajectory in the camera frame from a LAM
     Whittaker smooth, robust std per axis (1.4826 MAD); r2 = s_z^2 / mean(s_x^2, s_y^2);
  2. the final human's pelvis depth z_t (camera frame) gets the extra smoothing a depth-axis LAM of LAM * r2 implies
     (two Whittaker smoothers in series ~ one with summed LAM at low frequency): LAM_add = LAM * max(r2 - 1, 0), with the
     first scored frame pinned (the scorer's Sim(3) anchor: pose there unchanged);
  3. the body moves rigidly along each frame's camera ray through the pelvis by the depth change (root translation
     only), so the pelvis projection is unchanged and the image evidence is respected.
Writes OUT/episode_*.npz (+ the other files of --final) and depth_smooth_report.json.
Usage: depth_smooth_root.py --final DIR --final_bodies DIR --raw DIR --out DIR [--lam 300]
  --final_bodies / --raw: mhr_bodies.py outputs (joints) of --final and of the unsmoothed input human
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
FLIP = np.array([1.0, -1.0, -1.0])     # MHR root translation axes vs world (human_refine_common.FLIP), pose[:, :3] in dm
PELVIS = 1


def whittaker(y, lam, w=None):
    T = len(y); D = np.diff(np.eye(T), 2, axis=0); w = np.ones(T) if w is None else w
    return np.linalg.solve(np.diag(w) + lam * D.T @ D, w[:, None] * y if y.ndim == 2 else w * y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True); ap.add_argument("--final_bodies", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=300.0, help="the refit's LAM (all axes)")
    ap.add_argument("--max_lam", type=float, default=30000.0, help="cap on the extra depth LAM")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs/alignment_sam")
    ap.add_argument("--scored_frames", type=Path, default=ROOT / "outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for f in args.final.iterdir():
        if f.is_file() and not f.name.endswith(".npz"):
            shutil.copy2(f, args.out / f.name)
    S = json.loads(args.scored_frames.read_text())
    rep = {}
    for ep in range(30):
        tag = f"episode_{ep:06d}"; a = json.loads((args.alignment / tag / "alignment.json").read_text())
        R, t, ds = np.asarray(a["R_cw"], float), np.asarray(a["t_cw"], float), float(a["depth_scale"])
        cam = lambda J: ds * (J @ R.T + t)
        raw = cam(np.load(args.raw / f"{tag}.npz")["joints"].astype(float)[:, PELVIS])
        res = raw - whittaker(raw, args.lam)
        s = 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)
        r2 = float(s[2] ** 2 / np.mean(s[:2] ** 2)); lam_add = float(np.clip(args.lam * max(r2 - 1.0, 0.0), 0.0, args.max_lam))
        d = dict(np.load(args.final / f"{tag}.npz")); T = len(d["pose"])
        c = cam(np.load(args.final_bodies / f"{tag}.npz")["joints"].astype(float)[:T, PELVIS])
        anchor = min(f for f in S[str(ep)] if f < T)
        w = np.ones(T); w[anchor] = 1e9
        zs = whittaker(c[:, 2], lam_add, w) if lam_add > 0 else c[:, 2].copy()
        dz = zs - c[:, 2]; dz[anchor] = 0.0
        dc = c * (dz / c[:, 2])[:, None]                                  # along the camera ray through the pelvis
        dJ = dc @ R / ds                                                  # camera -> Track 1 frame displacement
        pose = d["pose"].astype(np.float64); pose[:, :3] += FLIP * dJ * 10.0
        d["pose"] = pose.astype(d["pose"].dtype); np.savez(args.out / f"{tag}.npz", **d)
        acc = lambda z: float(np.abs(np.diff(z, 2)).mean() * 100 / ds)
        rep[tag] = {"noise_std_cam_cm": (s * 100 / ds).round(3).tolist(), "r2": round(r2, 2), "lam_add": round(lam_add),
                    "depth_change_cm_mean": float(np.abs(dz).mean() * 100 / ds), "depth_change_cm_max": float(np.abs(dz).max() * 100 / ds),
                    "pelvis_depth_acc_cm": [acc(c[:, 2]), acc(zs)]}
        print(tag, json.dumps(rep[tag]), flush=True)
    v = list(rep.values())
    print("median r2 %.1f | lam_add median %d | depth change mean %.3f cm | pelvis depth acc %.4f -> %.4f" % (
        np.median([x["r2"] for x in v]), np.median([x["lam_add"] for x in v]), np.mean([x["depth_change_cm_mean"] for x in v]),
        np.mean([x["pelvis_depth_acc_cm"][0] for x in v]), np.mean([x["pelvis_depth_acc_cm"][1] for x in v])))
    (args.out / "depth_smooth_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
