#!/usr/bin/env python3
"""Push tracked objects out of the hands (the plan's SDF "plausibility glue" for PEN).

Runs in the submission venv (numpy, trimesh, fast_simplification, the kit's v2dlb). For each
assembled episode it measures penetration exactly as the scorer does: hand points (from
scripts/extract_mhr_hands.py) moved into the object's mesh frame by the inverse Sim(3) pose,
depth inside the budgeted object mesh by winding number. On every frame where a hand penetrates,
the object is translated away from the penetrating points by the smallest distance (bisection)
that clears them. The per-frame offsets are dilated (signed max-abs over a short window) and
Gaussian-smoothed so the correction does not inject object acceleration (ACC-O). Hands and the
human are untouched.

Reads <in>/episode_*.npz + meshes, writes <out>/ with updated object_translation and
<out>/penetration_report.json (self-measured PEN before/after, in cm with alignment scale 1).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import os

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
KIT = ROOT / "official" / "v2d_submission_kit"
sys.path.insert(0, str(KIT))
from tools.pack_reconstruction import budget_of, find_mesh, read_sample  # noqa: E402
from v2dlb import mhr_submission as ms  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

CLEAR_DEPTH_M = 0.002
# Below this self-measured PEN the push only adds object acceleration (episode 6: 0.002 cm PEN,
# ACC-O 0.019 -> 0.032 cm/frame^2), so such episodes pass through unchanged.
MIN_PEN_CM = 0.05
MAX_PUSH_M = 0.25
DILATE_FRAMES = 5
SIGMA_FRAMES = 2.0


def scored_frames_and_budget():
    sample = read_sample(KIT / "data" / "track_1_sample_submission.parquet")
    parts = sample["row_id"].str.split("_", expand=True)
    keys = {"episode": parts[1].astype(int), "frame": parts[2].astype(int),
            "array": parts[3].astype(int), "point": parts[4].astype(int)}
    import pandas as pd
    keys = pd.DataFrame(keys)
    order = ms.array_order()
    budget = budget_of(keys.drop(columns="frame"), order.index("object_mesh_faces"), order.index("object_mesh_vertices"))
    frames = {int(e): np.array(sorted(g.loc[g["frame"] != ms.EPISODE_FRAME, "frame"].unique()))
              for e, g in keys.groupby("episode")}
    return frames, budget


def depths(hands, rotation, translation, scale, vertices, faces):
    """[P] depth (m, scene units) of each hand point inside the posed object; 0 outside."""
    local = (hands - translation) @ rotation / scale
    return ms.penetration_depth(local, vertices, faces) * scale


def self_pen_cm(hands, rotation, translation, scale, vertices, faces, frames) -> float:
    per_frame = [depths(hands[f], rotation[f], translation[f], scale, vertices, faces).mean() for f in frames]
    return float(np.mean(per_frame) * 100.0) if per_frame else 0.0


def smooth_offsets(offsets: np.ndarray) -> np.ndarray:
    padded = np.pad(offsets, ((DILATE_FRAMES // 2, DILATE_FRAMES // 2), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, DILATE_FRAMES, axis=0)  # [T, 3, W]
    pick = np.abs(windows).argmax(axis=-1)
    dilated = np.take_along_axis(windows, pick[..., None], axis=-1)[..., 0]
    radius = int(3 * SIGMA_FRAMES)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / SIGMA_FRAMES) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(dilated, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1)


def acc_cm(t: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(t, n=2, axis=0), axis=1).mean() * 100.0) if len(t) > 2 else 0.0


def resolve_episode(task) -> dict:
    path, out_dir, frames, budget = task
    tag = path.stem
    data = dict(np.load(path))
    hands = np.load(ROOT / "outputs" / "hands" / f"{tag}.npz")["hands"].astype(np.float64)
    vertices, faces = budget_mesh(find_mesh(path.parent / f"{tag}_object"), *budget)
    rotation = data["object_rotation"].astype(np.float64)
    translation = data["object_translation"].astype(np.float64)
    scale = float(np.asarray(data["object_scale"]).reshape(()))
    center_local = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scored = frames[int(tag.split("_")[1])]
    before = self_pen_cm(hands, rotation, translation, scale, vertices, faces, scored)
    mesh_path = find_mesh(path.parent / f"{tag}_object")
    shutil.copy(mesh_path, out_dir / mesh_path.name)
    if before < MIN_PEN_CM:
        np.savez(out_dir / f"{tag}.npz", **data)
        return {"episode": tag, "skipped": True, "penetrating_frames": 0, "frames": len(translation),
                "self_pen_cm_before": before, "self_pen_cm_after": before, "max_offset_m": 0.0,
                "object_acc_cm_per_frame2_before": acc_cm(translation),
                "object_acc_cm_per_frame2_after": acc_cm(translation)}

    offsets = np.zeros_like(translation)
    penetrating = 0
    for f in range(len(translation)):
        d = depths(hands[f], rotation[f], translation[f], scale, vertices, faces)
        if d.max() <= CLEAR_DEPTH_M:
            continue
        penetrating += 1
        center = translation[f] + scale * rotation[f] @ center_local
        direction = center - hands[f][d > 0].mean(axis=0)
        direction /= np.linalg.norm(direction) + 1e-9
        lo, hi = 0.0, MAX_PUSH_M
        if depths(hands[f], rotation[f], translation[f] + hi * direction, scale, vertices, faces).max() > CLEAR_DEPTH_M:
            offsets[f] = hi * direction
            continue
        for _ in range(12):
            mid = (lo + hi) / 2
            clear = depths(hands[f], rotation[f], translation[f] + mid * direction, scale, vertices, faces).max() <= CLEAR_DEPTH_M
            lo, hi = (lo, mid) if clear else (mid, hi)
        offsets[f] = hi * direction

    smoothed = smooth_offsets(offsets) if penetrating else offsets
    resolved = translation + smoothed
    after = self_pen_cm(hands, rotation, resolved, scale, vertices, faces, scored)

    data["object_translation"] = resolved.astype(np.float32)
    np.savez(out_dir / f"{tag}.npz", **data)
    return {"episode": tag, "skipped": False, "penetrating_frames": penetrating, "frames": len(translation),
            "self_pen_cm_before": before, "self_pen_cm_after": after,
            "max_offset_m": float(np.linalg.norm(smoothed, axis=1).max()),
            "object_acc_cm_per_frame2_before": acc_cm(translation), "object_acc_cm_per_frame2_after": acc_cm(resolved)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frames, budget = scored_frames_and_budget()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(p, args.out, frames, budget) for p in sorted(args.input.glob("episode_*.npz"))]
    with ProcessPoolExecutor(args.workers) as pool:
        report = list(pool.map(resolve_episode, tasks))
    for item in report:
        print(f"{item['episode']}: penetrating {item['penetrating_frames']}/{item['frames']} frames, "
              f"self-PEN {item['self_pen_cm_before']:.3f} -> {item['self_pen_cm_after']:.3f} cm, "
              f"object acc {item['object_acc_cm_per_frame2_before']:.3f} -> {item['object_acc_cm_per_frame2_after']:.3f}")
    summary = {k: float(np.mean([r[k] for r in report])) for k in
               ("self_pen_cm_before", "self_pen_cm_after", "object_acc_cm_per_frame2_before", "object_acc_cm_per_frame2_after")}
    (args.out / "penetration_report.json").write_text(json.dumps({
        "schema": "v2d.track1.hand_penetration_resolution.v1",
        "hand_points": "approximate (scripts/extract_mhr_hands.py)",
        "clear_depth_m": CLEAR_DEPTH_M, "max_push_m": MAX_PUSH_M, "min_pen_cm": MIN_PEN_CM,
        "dilate_frames": DILATE_FRAMES, "gaussian_sigma_frames": SIGMA_FRAMES,
        "mean": summary, "episodes": report}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"PASS: wrote {len(report)} penetration-resolved episodes to {args.out}")


if __name__ == "__main__":
    main()
