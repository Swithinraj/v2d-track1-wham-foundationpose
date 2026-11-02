#!/usr/bin/env python3
"""Measure the resting-object floor gap of an episodes directory (read-only; numpy + trimesh, CPU).

docs/PIPELINE.md lists as a known limit that objects resting on the floor sit 8-14 cm below the
floor the body stands on, and every current recipe runs scripts/physics_objects.py with
--no_gravity, so the gravity term that would correct it never runs. This script measures exactly
what that disabled term would see, per episode, without changing anything: the same floor plane
(--floor, MoGe metres, converted by the alignment depth scale), the same convex-hull lowest point,
the same FLOOR_NEAR / FLOOR_FAR weights and the same camera-ray snap step (clipped to MAX_SNAP_M).

Reported per episode (floor_gap_report.json + stdout): floor share; signed height on floor frames
(negative = below the floor); |height| median (the gap the gravity term closes); the height and
snap at the first scored frame; and the snap the term would apply. Heights use the raw per-frame
rotation (physics_objects uses rest-medoid rotations when --rest is on).

Usage: measure_floor_gap.py --input DIR --floor FILE [--alignment DIR] [--scored FILE]
  --input       episodes dir (episode_XXXXXX.npz + episode_XXXXXX_object.*)
  --floor       outputs/physics/kappa_floor.json
  --alignment   outputs/alignment_sam (depth_scale per episode)
  --scored      JSON {episode: [frames]}; if absent, frame 0 and all frames are used
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
FLOOR_NEAR_M, FLOOR_FAR_M, MAX_SNAP_M = 0.06, 0.12, 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--floor", type=Path, default=ROOT / "outputs/physics/kappa_floor.json")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs/alignment_sam")
    ap.add_argument("--scored", type=Path, default=None)
    args = ap.parse_args()
    planes = json.loads(args.floor.read_text())
    scored = json.loads(args.scored.read_text()) if args.scored and args.scored.exists() else {}
    rep = {}
    for f in sorted(args.input.glob("episode_??????.npz")):
        tag = f.stem
        if tag not in planes:
            print(f"{tag}: no floor plane, skipped", flush=True)
            continue
        d = dict(np.load(f))
        if "object_translation" not in d:
            continue
        t = d["object_translation"].astype(np.float64)
        R = d["object_rotation"].astype(np.float64)
        s_obj = float(np.asarray(d["object_scale"]).reshape(()))
        T = len(t)
        fl = planes[tag]
        n = np.asarray(fl["normal"], float)
        c = fl["c"] / json.loads((args.alignment / tag / "alignment.json").read_text())["depth_scale"]
        V = np.asarray(trimesh.load(next(f.parent.glob(f"{tag}_object*")), force="mesh").convex_hull.vertices, float)
        low = (s_obj * np.einsum("tij,vj->tvi", R, V) @ n).min(axis=1)
        height = low + t @ n - c
        w = np.clip((FLOOR_FAR_M - np.abs(height)) / (FLOOR_FAR_M - FLOOR_NEAR_M), 0.0, 1.0)
        w = np.array([np.median(w[max(0, i - 7):i + 8]) for i in range(T)])
        on = w > 0.5
        ray = t / np.linalg.norm(t, axis=1, keepdims=True)
        nr = ray @ n
        step = np.where(np.abs(nr) > 0.05, -height / np.where(np.abs(nr) > 0.05, nr, 1.0), 0.0)
        step = np.clip(step, -MAX_SNAP_M, MAX_SNAP_M)
        fr = scored.get(tag, [0])
        r = {"episode": tag, "frames": T, "floor_share": float(on.mean()),
             "height_cm_median": float(np.median(height[on]) * 100) if on.any() else None,
             "height_cm_mean": float(height[on].mean() * 100) if on.any() else None,
             "gap_cm_median": float(np.median(np.abs(height[on])) * 100) if on.any() else None,
             "gap_cm_p90": float(np.percentile(np.abs(height[on]), 90) * 100) if on.any() else None,
             "snap_cm_median": float(np.median(np.abs(step[on])) * 100) if on.any() else None,
             "snap_cm_max": float(np.abs(step[on]).max() * 100) if on.any() else None,
             "snap_capped_share": float((np.abs(step[on]) >= MAX_SNAP_M - 1e-9).mean()) if on.any() else None,
             "first_scored_frame": int(fr[0]),
             "height_cm_first_scored": float(height[fr[0]] * 100) if fr[0] < T else None,
             "snap_cm_first_scored": float(step[fr[0]] * 100) if fr[0] < T else None}
        rep[tag] = r
        print(f"{tag}: floor {r['floor_share']*100:3.0f} % | gap {r['gap_cm_median'] or -1:5.1f} cm"
              f" (signed {r['height_cm_median'] or -1:+5.1f}) | snap {r['snap_cm_median'] or -1:4.1f} cm"
              f" max {r['snap_cm_max'] or -1:5.1f} | first scored f{r['first_scored_frame']}:"
              f" {r['height_cm_first_scored'] or -1:+5.1f} cm", flush=True)
    out = args.input / "floor_gap_report.json"
    out.write_text(json.dumps(rep, indent=1))
    on_all = [r for r in rep.values() if (r["gap_cm_median"] or 0) is not None and r["floor_share"] > 0.05]
    if on_all:
        med = np.median([r["gap_cm_median"] for r in on_all])
        signed = np.median([r["height_cm_median"] for r in on_all])
        anchor = [r["height_cm_first_scored"] for r in on_all if r["height_cm_first_scored"] is not None]
        print(f"\nPASS: {len(on_all)} episodes with floor contact; |gap| median {med:.1f} cm "
              f"(signed {signed:+.1f} cm); first-scored-frame height median "
              f"{np.median(anchor):+.1f} cm -> {out}")
        print("verdict: gravity term would move resting objects by the snap column; compare with"
              " the 8-14 cm documented limit")


if __name__ == "__main__":
    main()
