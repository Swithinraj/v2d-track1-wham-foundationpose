"""Object depth follows the human's depth trend (CARI4D's principle, smoothed), SAM 3D Body camera world
(submission venv: numpy only; pack.sh with HUMAN=sam3d OBJECT_FOLLOW_HUMAN=1).

CARI4D rescales every depth frame to the SAM 3D Body human before tracking the object, so its object depth follows
r(t) = MoGe depth / SAM human depth per frame (CARI4D's ACC-O 0.726 comes from the per-frame part of that). Here
only the trend is kept: r (sam3d_body_identity.py, every third frame) is interpolated to all frames, median-filtered
(15 frames) and Gaussian-smoothed (sigma 30), the same filter as the depth-trend correction. The camera-frame object
translation is scaled by f(t) = s / trend(t) with s the episode's constant MoGe/SAM depth scale used so far, i.e.
the object moves along its camera ray from t_FP / s to t_FP / trend(t). Rotation, scale and mesh are unchanged; f
is smooth, so the object's acceleration hardly changes. Episodes without a prepare.npz series are copied.
Only meaningful when the human keeps SAM 3D Body's own depth (no human depth-trend correction): then human and object
follow the same depth. A frame-by-frame variant (no filter) was tested and rejected: the object then shakes ~1 cm per
frame against the smoothed human (object acceleration x63, self-PEN higher).
Usage: object_follow_human.py --input DIR --out DIR [--prep_dir DIR] [--episodes ...]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def median_filter(x, k):
    h = k // 2
    p = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)


def gaussian(x, sigma):
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(x, r, mode="edge"), w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prep_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/sam3d_body")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args()
    shutil.copytree(args.input, args.out, dirs_exist_ok=True)
    report = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        prep = args.prep_dir / tag / "prepare.npz"
        if not prep.exists():
            continue
        p = np.load(prep)
        if "r" not in p.files:
            continue
        d = dict(np.load(args.input / f"{tag}.npz"))
        T = len(d["object_translation"])
        ok = np.isfinite(p["r"])
        r = np.interp(np.arange(T), p["frames"][ok], p["r"][ok])
        trend = gaussian(median_filter(r, 15), 30.0)
        f = float(p["depth_scale"]) / trend
        t = d["object_translation"].astype(np.float64)
        d["object_translation"] = (t * f[:, None]).astype(np.float32)
        np.savez(args.out / f"{tag}.npz", **d)
        move = np.linalg.norm(t * (f[:, None] - 1), axis=1) * 100
        report[tag] = {"factor_median": float(np.median(f)), "factor_p5_p95": [float(np.percentile(f, 5)), float(np.percentile(f, 95))],
                       "move_cm_median": float(np.median(move)), "move_cm_max": float(move.max())}
        print(tag, json.dumps(report[tag]), flush=True)
    (args.out / "obj_follow_human_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
