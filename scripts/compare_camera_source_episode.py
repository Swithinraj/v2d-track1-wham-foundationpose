#!/usr/bin/env python3
"""Paired episode diagnostic for centered-MoGe vs measured-camera SAM 3D Body.

This is a visual/consistency proxy, not the private Track 1 CD-H/CD-O score.
Both runs must be independently present; missing comparison data is an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COCO_FROM_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def load_run(root: Path, tag: str) -> dict:
    work = root / "sam3d_body" / tag
    paths = [work / f"raw_{tag}.npz", work / "prepare.npz",
             root / "camera" / tag / "intrinsics.json"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing run files:\n" + "\n".join(missing))
    with np.load(paths[0]) as z:
        k = z["K"].copy()
        xy = z["pred_keypoints_2d"][:, COCO_FROM_MHR70, :2].copy()
        frames = len(z["pred_keypoints_2d"])
    with np.load(paths[1]) as z:
        reliable = z["reliable"].copy()
        depth_scale = float(z["depth_scale"])
    refit_path = work / "refit.npz"
    if refit_path.is_file():
        with np.load(refit_path) as z:
            fit = z["fit_err_mm"].copy()
    else:
        fit = None
    camera = json.loads(paths[2].read_text())
    if len(reliable) != frames or (fit is not None and len(fit) != frames) or xy.shape != (frames, 17, 2):
        raise ValueError(f"{root}: inconsistent frame count or keypoint shape")
    if not np.isfinite(xy).all() or not np.isfinite(k).all() or (fit is not None and not np.isfinite(fit).all()):
        raise ValueError(f"{root}: nonfinite prediction")
    if not np.allclose(k[0, 2], camera["cx"], atol=0.01) or not np.allclose(k[1, 2], camera["cy"], atol=0.01):
        raise ValueError(f"{root}: raw K disagrees with camera JSON")
    return dict(xy=xy, k=k, reliable=reliable, depth_scale=depth_scale, fit=fit,
                raw_bytes=paths[0].stat().st_size)


def bootstrap_blocks(values: np.ndarray, seed: int = 9, block: int = 15) -> list[float]:
    """95% interval for mean paired delta; contiguous blocks preserve local dependence."""
    rng = np.random.default_rng(seed)
    blocks = [values[i:i + block] for i in range(0, len(values), block)]
    means = []
    for _ in range(2000):
        chosen = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[i] for i in chosen])
        means.append(float(np.mean(sample)))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="baseline outputs root")
    ap.add_argument("--comparison", type=Path, required=True, help="comparison outputs root")
    ap.add_argument("--vitpose", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True, help="official Track 1 sample parquet")
    ap.add_argument("--camera-id", type=Path, required=True)
    ap.add_argument("--rig", type=Path, required=True)
    ap.add_argument("--episode", type=int, default=9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tag = f"episode_{args.episode:06d}"
    a, b = load_run(args.baseline, tag), load_run(args.comparison, tag)
    if a["xy"].shape != b["xy"].shape:
        raise ValueError("Baseline and comparison have different frame counts")
    t = len(a["xy"])
    with np.load(args.vitpose) as z:
        target = z["kps2d"].copy()
    if target.shape != (t, 17, 3):
        raise ValueError(f"ViTPose shape {target.shape} does not match {(t, 17, 3)}")
    camera_id = json.loads(args.camera_id.read_text())[tag]
    rig = json.loads(args.rig.read_text())[camera_id["camera"]]
    if camera_id["corr"] < .8 or camera_id["corr"] - camera_id["corr_other"] < .3:
        raise ValueError("Episode camera identification is ambiguous")
    if not np.allclose(a["k"][:2, 2], [768, 576], atol=.1):
        raise ValueError("Baseline K is not centered MoGe")
    if not np.allclose(b["k"][:2, 2], [rig["cx"], rig["cy"]], atol=.1):
        raise ValueError("Comparison K is not measured rig")
    if not np.allclose(a["k"][0, 0], b["k"][0, 0], atol=.1):
        raise ValueError("Focal changed; expected tcp mode (principal point only)")
    rows = pd.read_parquet(args.sample, columns=["row_id"])["row_id"].str.split("_", expand=True)
    scored = np.sort(rows.loc[(rows[1].astype(int) == args.episode) & (rows[2] != "999999"), 2].astype(int).unique())
    if not len(scored) or scored[0] < 0 or scored[-1] >= t:
        raise ValueError("Scored frames absent or outside video")
    conf = target[:, :, 2] >= .5
    inside = (target[:, :, 0] >= 0) & (target[:, :, 0] < 1536) & (target[:, :, 1] >= 0) & (target[:, :, 1] < 1152)
    valid = conf & inside
    if not np.all(valid.sum(1) >= 10):
        raise ValueError("Fewer than 10 confident ViTPose joints on a frame")
    errors = [np.linalg.norm(run["xy"] - target[:, :, :2], axis=2) for run in (a, b)]
    frame_err = [np.array([np.median(e[i, valid[i]]) for i in range(t)]) for e in errors]
    delta = frame_err[1] - frame_err[0]
    def subset(idx):
        return {
            "frames": int(len(idx)),
            "baseline_median_px": float(np.median(frame_err[0][idx])),
            "comparison_median_px": float(np.median(frame_err[1][idx])),
            "paired_mean_delta_px": float(np.mean(delta[idx])),
            "paired_mean_delta_ci95_px": bootstrap_blocks(delta[idx]),
            "comparison_better_frame_fraction": float(np.mean(delta[idx] < 0)),
            "baseline_pck20": float(np.mean(errors[0][idx][valid[idx]] < 20)),
            "comparison_pck20": float(np.mean(errors[1][idx][valid[idx]] < 20)),
        }
    report = {
        "episode": args.episode, "interpretation": "diagnostic only; not Track 1 ground truth or Kaggle score",
        "camera": camera_id["camera"], "baseline_K": a["k"].tolist(), "comparison_K": b["k"].tolist(),
        "frames": t, "scored_first": int(scored[0]), "scored_last": int(scored[-1]),
        "all_frames": subset(np.arange(t)), "scored_frames": subset(scored),
        "first_scored_frame": {"index": int(scored[0]), "baseline_median_px": float(frame_err[0][scored[0]]),
                               "comparison_median_px": float(frame_err[1][scored[0]])},
        "reliable_frames": [int(a["reliable"].sum()), int(b["reliable"].sum())],
        "refit_error_median_mm": [None if run["fit"] is None else float(np.median(run["fit"])) for run in (a, b)],
        "depth_scale": [a["depth_scale"], b["depth_scale"]],
        "raw_bytes": [a["raw_bytes"], b["raw_bytes"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
