#!/usr/bin/env python3
"""Pick the TRELLIS mesh candidate that FoundationPose fits best in every camera, fix its metric
scale, and install it for tracking.

Runs inside containers/v2d_cari4d.sif after scripts/trellis_object_mesh.py. Every object is
seen by three different exo cameras (one episode each), so a candidate with the wrong shape
(a bench for a stool, a dish for a bowl, an object embedded in a plate) cannot match all three
silhouettes. For each candidate and each episode's selection frame, a coarse-to-fine scale
search (as in FoundationPoseTracker.estimate_scale_grid_search) registers the mesh with
FoundationPose and scores the render against the SAM 2 mask and MoGe-2 depth. The candidate
with the best mean score wins; its scale is refined once more and the per-episode median is
applied. Writes:

  inputs/object_meshes/<object>.glb               consumed by pipeline/stages/track_object.sh
  inputs/object_meshes/<object>.provenance.json   generation + selection record (not ground truth)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
from v2d.common.datatypes import Image as V2dImage
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.mesh.lib.mesh import Mesh

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
WEIGHTS = ROOT / "weights" / "object" / "foundationpose"
BACKEND = "nvlabs_pytorch"
IOU_WEIGHT, DEPTH_WEIGHT = 1.0, 0.5  # silhouette agreement carries most of the shape evidence
# FoundationPose registration always places the mesh at the observed depth, so the depth term is
# ~1 for any scale, and silhouette IoU alone rewards inflating an imperfect (e.g. too slim) mesh
# until it covers the mask: the iron went to 2x its depth-derived size. The MoGe-2 extent sets the
# metric size; the FP search may only refine it within this relative range.
SEARCH_LO, SEARCH_HI = 0.75, 1.33
logging.basicConfig(level=logging.WARNING)


def load_frame(frame_dir: Path):
    bgr = cv2.imread(str(frame_dir / "rgb.png"))
    return (V2dImage(data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), DepthImage.load(str(frame_dir / "depth.png")),
            Mask.load(str(frame_dir / "mask.png")), CameraIntrinsics.load(str(frame_dir / "intrinsics.json")))


def scale_search(tracker: FoundationPoseTracker, frame, n_samples: int, n_levels: int,
                 lo: float = SEARCH_LO, hi: float = SEARCH_HI) -> dict:
    rgb, depth, mask, intrinsics = frame
    log_lo, log_hi = np.log(lo), np.log(hi)
    best = {"scale": float(np.sqrt(lo * hi)), "score": -np.inf}
    for _ in range(n_levels):
        for scale in np.exp(np.linspace(log_lo, log_hi, n_samples)):
            tracker.rescale_to(float(scale))
            pose = tracker.register(rgb, depth, mask, intrinsics, iteration=5)
            iou = tracker._score_scale(depth, mask, intrinsics, pose, 1.0, 0.0, 0.0)
            depth_fit = tracker._score_scale(depth, mask, intrinsics, pose, 0.0, 1.0, 0.0)
            score = IOU_WEIGHT * iou + DEPTH_WEIGHT * depth_fit
            if score > best["score"]:
                best = {"scale": float(scale), "score": float(score), "iou": float(iou),
                        "depth_fit": float(depth_fit), "pose": pose.to_dict()}
        radius = (log_hi - log_lo) / 4
        log_lo = max(np.log(best["scale"]) - radius, np.log(lo))
        log_hi = min(np.log(best["scale"]) + radius, np.log(hi))
    best["at_search_boundary"] = bool(np.isclose(best["scale"], lo, rtol=0.03) or np.isclose(best["scale"], hi, rtol=0.03))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", required=True)
    parser.add_argument("--include_plate_suspects", action="store_true")
    args = parser.parse_args()

    out = ROOT / "outputs" / "object_meshes" / args.object
    provenance = json.loads((out / "provenance.json").read_text())
    frames = {int(d.name.split("_")[1]): load_frame(d) for d in sorted((out / "scale_frames").glob("episode_*"))}
    usable = [c for c in provenance["candidates"] if c["status"] == "ok"]
    clean = [c for c in usable if not c["plate_suspect"]]
    pool = usable if args.include_plate_suspects or len(clean) < 2 else clean

    for candidate in pool:
        mesh = Mesh.load(str(out / "candidates" / candidate["id"] / "coarse_metric.glb"))
        tracker = FoundationPoseTracker(mesh, str(WEIGHTS), backend=BACKEND)
        fits = {e: scale_search(tracker, frame, n_samples=5, n_levels=2) for e, frame in frames.items()}
        candidate["selection"] = {
            "mean_score": float(np.mean([f["score"] for f in fits.values()])),
            "mean_iou": float(np.mean([f["iou"] for f in fits.values()])),
            "per_episode": {str(e): {k: f[k] for k in ("scale", "score", "iou", "depth_fit", "at_search_boundary")}
                            for e, f in fits.items()},
        }
        print(f"{args.object} {candidate['id']} episodes={candidate['condition_episodes']} seed={candidate['seed']} "
              f"mean_score={candidate['selection']['mean_score']:.3f} mean_iou={candidate['selection']['mean_iou']:.3f}",
              flush=True)
        del tracker

    winner = max(pool, key=lambda c: c["selection"]["mean_score"])
    mesh = Mesh.load(str(out / "candidates" / winner["id"] / "coarse_metric.glb"))
    tracker = FoundationPoseTracker(mesh, str(WEIGHTS), backend=BACKEND)
    refined = {e: scale_search(tracker, frame, n_samples=7, n_levels=3) for e, frame in frames.items()}
    for e, fit in refined.items():
        (out / "scale_frames" / f"episode_{e:06d}" / "selected_fit.json").write_text(json.dumps(fit, indent=2))
    scales = [fit["scale"] for fit in refined.values()]
    relative = float(np.median(scales))

    final = trimesh.load(out / "candidates" / winner["id"] / "coarse_metric.glb", force="mesh")
    final.apply_scale(relative)
    target = ROOT / "inputs" / "object_meshes" / f"{args.object}.glb"
    target.parent.mkdir(parents=True, exist_ok=True)
    final.export(target)

    spread = float(max(scales) / min(scales))
    mean_iou = float(np.mean([fit["iou"] for fit in refined.values()]))
    provenance["selection"] = {
        "method": "max mean over cameras of IoU + 0.5 * exp(-median|depth ratio - 1|) after FoundationPose "
                  "registration at the best scale (coarse-to-fine log grid in "
                  f"[{SEARCH_LO}, {SEARCH_HI}] x the MoGe-2 depth-extent scale)",
        "pool": [c["id"] for c in pool],
        "winner": winner["id"],
        "refined_per_episode": {str(e): {k: f[k] for k in ("scale", "score", "iou", "depth_fit", "at_search_boundary")}
                                for e, f in refined.items()},
        "median_relative_scale": relative,
        "max_over_min_scale_spread": spread,
    }
    provenance["final_extents_m"] = [float(x) for x in final.extents]
    provenance["requires_visual_review"] = bool(mean_iou < 0.5 or spread > 1.5
                                                or any(f["at_search_boundary"] for f in refined.values()))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"{args.object}: winner {winner['id']} (episodes {winner['condition_episodes']}, seed {winner['seed']}), "
          f"refined IoU {mean_iou:.3f}, scales {[round(s, 3) for s in scales]}, extents {np.round(final.extents, 3).tolist()} m")
    if provenance["requires_visual_review"]:
        print(f"REVIEW: {args.object} has low silhouette agreement, inconsistent scales, or a boundary scale")
    print(f"PASS: installed generated metric mesh {target}")


if __name__ == "__main__":
    main()
