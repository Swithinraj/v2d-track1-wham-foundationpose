#!/usr/bin/env python3
"""Per-frame visibility of the body's limb parts, for the occlusion-aware refit (numpy + PIL python, CPU).

SAM 3D Body guesses hidden limbs, and its guesses jitter more (raw limb acceleration x1.19 overall, ankles +70 %), so the
refit should trust a part's per-frame estimate only where the part is seen. Per frame and part (shoulder, elbow, wrist,
hip, knee, ankle on each side, = the MediaPipe pose landmarks of human_refine_common.PAIRS):
  seen     MediaPipe visibility of the actor's landmark (actor = the person closest to our projected joints)
  hidden   the part's joint projects into the object's SAM 2 mask (dilated) and lies behind the object's centre
weight = 1 when visibility > VIS_HI and not hidden, W_HIDDEN when visibility < VIS_LO or hidden, linear between;
no MediaPipe actor -> W_UNKNOWN; median-filtered over 5 frames.
Writes <out>/episode_*.npz: weights [T, 12], joints [12] (MHR joint per part, PAIRS order).
Usage: body_visibility.py --bodies DIR --episodes DIR --out DIR   (bodies: mhr_bodies.py joints of the episodes)
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter



PAIRS_ = {"wrist": ((78, 15), (42, 16)), "elbow": ((76, 13), (40, 14)), "shoulder": ((75, 11), (39, 12)),
          "hip": ((2, 23), (18, 24)), "knee": ((3, 25), (19, 26)), "ankle": ((4, 27), (20, 28))}   # = human_refine_common


class C:
    """The few helpers of human_refine_common this needs (that module pulls in pandas; this runs where PIL is)."""
    ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
    PAIRS = PAIRS_
    MATCH = [p for g in ("wrist", "elbow", "shoulder", "hip", "knee") for p in PAIRS_[g]]

    @staticmethod
    def alignment(tag):
        root = Path(os.environ.get("TRACK1_ALIGNMENT_DIR") or C.ROOT / "outputs/alignment_sam")
        return json.loads((root / tag / "alignment.json").read_text())

    @staticmethod
    def camera(J, a):
        return float(a["depth_scale"]) * (J @ np.asarray(a["R_cw"]).T + np.asarray(a["t_cw"]))

    @staticmethod
    def project(J, a):
        k, c = a["intrinsics"], C.camera(J, a)
        return np.stack([k["fx"] * c[..., 0] / c[..., 2] + k["cx"], k["fy"] * c[..., 1] / c[..., 2] + k["cy"]], -1)

    @staticmethod
    def actor_landmarks(mp, uv):
        T = min(len(mp), len(uv))
        out = np.full((T, 33, 3), np.nan)
        for f in range(T):
            d = [np.nanmedian([np.linalg.norm(mp[f, i, l, :2] - uv[f, j]) for j, l in C.MATCH]) for i in range(3)]
            if np.isfinite(d).any() and np.nanmin(d) < 60:
                i = int(np.nanargmin(d)); out[f, :, :2] = mp[f, i, :, :2]; out[f, :, 2] = mp[f, i, :, 3]
        return out

VIS_HI, VIS_LO, W_HIDDEN, W_UNKNOWN = 0.8, 0.5, 0.25, 0.6
PARTS = [p for g in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle") for p in C.PAIRS[g]]
ARGS = None
warnings.filterwarnings("ignore", category=RuntimeWarning)


def episode(ep):
    tag = f"episode_{ep:06d}"
    J = np.load(ARGS.bodies / f"{tag}.npz")["joints"].astype(np.float64)
    d = np.load(ARGS.episodes / f"{tag}.npz")
    a = C.alignment(tag)
    uv = C.project(J, a)
    T = len(J)
    mp = C.actor_landmarks(np.load(C.ROOT / "outputs/mediapipe" / f"{tag}.npz")["pose"].astype(np.float64), uv)
    cam = C.camera(J, a)                                                       # depth comparable with the object
    obj_c = C.camera(d["object_translation"].astype(np.float64), a)[:, 2]
    mdir = C.ROOT / "outputs/object" / tag / "sam2_masks/1"
    w = np.full((T, len(PARTS)), W_UNKNOWN)
    hidden_share = np.zeros(len(PARTS))
    for f in range(min(T, len(mp))):
        img = Image.open(mdir / f"{f:06d}.png")
        m = np.array(img.resize((img.width // 4, img.height // 4)).filter(ImageFilter.MaxFilter(3))) > 127
        for k, (j, l) in enumerate(PARTS):
            u, v = np.round(uv[f, j] / 4).astype(int)
            behind = 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u] and cam[f, j, 2] > obj_c[f]
            vis = mp[f, l, 2]
            if behind:
                w[f, k] = W_HIDDEN
            elif np.isfinite(vis):
                w[f, k] = W_HIDDEN + (1 - W_HIDDEN) * np.clip((vis - VIS_LO) / (VIS_HI - VIS_LO), 0, 1)
            hidden_share[k] += w[f, k] <= W_HIDDEN + 1e-9
    w = np.stack([np.array([np.median(w[max(0, f - 2):f + 3, k]) for f in range(T)]) for k in range(len(PARTS))], 1)
    np.savez(ARGS.out / f"{tag}.npz", weights=w.astype(np.float32), joints=np.array([j for j, _ in PARTS]))
    return tag, {"mean_weight": float(w.mean()), "hidden_share": (hidden_share / T).round(3).tolist()}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bodies", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    eps = sorted(int(p.stem[-6:]) for p in ARGS.episodes.glob("episode_??????.npz"))
    with ProcessPoolExecutor(ARGS.workers) as pool:
        rep = dict(pool.map(episode, eps))
    (ARGS.out / "visibility_report.json").write_text(json.dumps({"parts": [list(p) for p in PARTS], "episodes": rep}, indent=1))
    print("body visibility:", len(rep), "episodes, mean weight", round(float(np.mean([r["mean_weight"] for r in rep.values()])), 3))


if __name__ == "__main__":
    main()
