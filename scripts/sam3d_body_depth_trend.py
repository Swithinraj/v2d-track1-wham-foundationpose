"""SAM 3D Body human: the WHAM pipeline's MoGe depth-trend correction (H2t) in the camera world (wham.sif).

Same rule as TRACK1 scripts/depth_trend_correction.py apply: the background-normalised MoGe/SAM depth ratio qn
(sam3d_body_identity.py, every third frame) is interpolated to every frame, median-filtered (15 frames) and Gaussian-
smoothed (sigma 30 frames); the MHR root moves along the camera ray by alpha * (trend - 1) * centroid. The world
frame is the camera frame, so the camera-frame correction is applied directly (pose[:, :3] += FLIP * d * 10).
Centroid = SAM 3D Body's mesh centroid (pred_vertices + pred_cam_t) on the same frames as qn.
Usage: sam3d_body_depth_trend.py --input DIR --out DIR [--episodes E ...] [--alpha 0.75] [--raw_dir DIR --prep_dir DIR]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

FLIP = np.array([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=Path("/track1/outputs/sam3d_body"))
    ap.add_argument("--prep_dir", type=Path, default=Path("/track1/outputs/sam3d_body"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--alpha", type=float, default=0.75)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(args.input.glob("episode_*.npz")):
        tag = path.stem
        ep = int(tag.split("_")[1])
        data = dict(np.load(path))
        for m in args.input.glob(f"{tag}_object.*"):
            shutil.copy(m, args.out / m.name)
        if ep in args.episodes:
            T = len(data["pose"])
            prep = np.load(args.prep_dir / tag / "prepare.npz")
            raw = np.load(args.raw_dir / tag / f"raw_{tag}.npz")
            frames, qn = prep["frames"], prep["qn"]
            ok = np.isfinite(qn)
            cent_all = (raw["pred_vertices"].astype(np.float64).mean(1) + raw["pred_cam_t"].astype(np.float64))
            q = np.interp(np.arange(T), frames[ok], qn[ok])
            q = median_filter(q, size=15, mode="nearest")
            trend = gaussian_filter1d(q, 30.0, mode="nearest")
            cent = np.stack([np.interp(np.arange(T), frames[ok], cent_all[frames[ok], c]) for c in range(3)], 1)
            delta = args.alpha * ((trend - 1.0)[:, None] * cent)
            pose = data["pose"].astype(np.float64)
            pose[:, :3] += FLIP * delta * 10.0
            data["pose"] = pose.astype(np.float32)
            size = np.linalg.norm(delta, axis=1) * 100
            report[tag] = {"correction_cm_median": float(np.median(size)), "correction_cm_max": float(size.max())}
            print(tag, json.dumps(report[tag]), flush=True)
        np.savez(args.out / path.name, **data)
    (args.out / "sam_depth_trend_report.json").write_text(json.dumps({"alpha": args.alpha, "episodes": report}, indent=2))


if __name__ == "__main__":
    main()
