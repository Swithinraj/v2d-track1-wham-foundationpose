#!/usr/bin/env python3
"""Clear every hand vertex out of the object on every scored frame: penetration (PEN) driven to zero (submission venv).

The scorer's PEN is the mean depth of the body's hand vertices (a host-side role list) inside the submitted object
mesh, by winding number, over the scored frames. The role list is not public, so this clears the DENSE hand vertices
(mhr_bodies.py --hands-only: every vertex whose dominant skinning joint lies in a wrist subtree, 4636) from the
budgeted object mesh (the mesh the packer writes), with a margin, which clears any subset of them.

Per episode, after the hand-side contact fix (only residual, mostly palm penetration is left):
  1. on every scored frame where a hand vertex is inside, the object is pushed away from the penetrating vertices
     (direction: object centre minus their mean) by the smallest distance that clears all of them, plus MARGIN;
  2. the per-frame push vectors are spread over a window (signed max-abs) and Gaussian-smoothed, so the object does
     not jerk (ACC-O);
  3. every scored frame is checked again; frames that still penetrate get their remaining push added and step 2 is
     repeated (ITERATIONS); whatever is left after that is cleared exactly on its frame.
The human is untouched (CD-H, ACC-H unchanged); objects move by the pushes only.
Writes <out>/episode_*.npz (+ meshes) and pen_zero_report.json.
Usage: pen_zero.py --input DIR --hands DIR --out DIR [--workers N] [--margin_m 0.001] [--sigma 3] [--cap_m M]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolve_hand_penetration as rp  # noqa: E402
from v2dlb import mhr_submission as ms  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

ARGS = None
ITERATIONS = 6
MAX_PUSH_M = 0.25


def frame_depths(hands, rotation, translation, scale, vertices, faces):
    local = (hands - translation) @ rotation / scale
    return ms.penetration_depth(local, vertices, faces) * scale


def clearing_push(hands, rotation, translation, scale, vertices, faces, center_local):
    """Smallest push (vector, m) along (object centre - penetrating vertices) that clears every hand vertex."""
    d = frame_depths(hands, rotation, translation, scale, vertices, faces)
    if d.max() <= 0.0:
        return np.zeros(3)
    direction = translation + scale * rotation @ center_local - hands[d > 0].mean(axis=0)
    direction /= np.linalg.norm(direction) + 1e-12
    if ARGS.ray:   # push along the camera ray (depth: the direction the image cannot see), on the side that clears first
        ray = translation / (np.linalg.norm(translation) + 1e-12)
        best = None
        for sign in (1.0, -1.0):
            v = sign * ray
            for step in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25):
                if frame_depths(hands, rotation, translation + step * v, scale, vertices, faces).max() <= 0.0:
                    if best is None or step < best[0]:
                        best = (step, v)
                    break
        if best is not None:
            direction = best[1]
    lo, hi = 0.0, ARGS.cap_m or MAX_PUSH_M
    if frame_depths(hands, rotation, translation + hi * direction, scale, vertices, faces).max() > 0.0:
        return np.zeros(3) if ARGS.cap_m else hi * direction   # with a cap, frames needing more are left as they are
    for _ in range(14):
        mid = (lo + hi) / 2
        if frame_depths(hands, rotation, translation + mid * direction, scale, vertices, faces).max() > 0.0:
            lo = mid
        else:
            hi = mid
    return (hi + ARGS.margin_m) * direction


def spread(offsets: np.ndarray, window: int, sigma: float) -> np.ndarray:
    half = window // 2
    padded = np.pad(offsets, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    pick = np.abs(windows).argmax(axis=-1)
    dilated = np.take_along_axis(windows, pick[..., None], axis=-1)[..., 0]
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(dilated, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1)


def episode(path: Path) -> dict:
    tag = path.stem
    data = dict(np.load(path))
    hands = np.load(ARGS.hands / f"{tag}.npz")["hands"].astype(np.float64)
    frames_by_ep, budget = rp.scored_frames_and_budget()
    mesh = rp.find_mesh(path.parent / f"{tag}_object")
    vertices, faces = budget_mesh(mesh, *budget)
    shutil.copy(mesh, ARGS.out / mesh.name)
    rotation = data["object_rotation"].astype(np.float64)
    translation = data["object_translation"].astype(np.float64)
    scale = float(np.asarray(data["object_scale"]).reshape(()))
    center_local = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scored = [f for f in frames_by_ep[int(tag.split("_")[1])] if f < len(translation)]

    def pen(trans):
        per = [frame_depths(hands[f], rotation[f], trans[f], scale, vertices, faces) for f in scored]
        return float(np.mean([p.mean() for p in per]) * 100), int(sum(p.max() > 0 for p in per))

    before, frames_before = pen(translation)
    required = np.zeros_like(translation)
    offsets = np.zeros_like(translation)
    iterations = 0
    if frames_before:
        for iterations in range(1, ARGS.iterations + 1):
            current = translation + offsets
            pushes = {f: clearing_push(hands[f], rotation[f], current[f], scale, vertices, faces, center_local)
                      for f in scored}
            left = [f for f, p in pushes.items() if np.linalg.norm(p) > 0]
            if not left:
                break
            for f in left:
                required[f] = offsets[f] + pushes[f]
            offsets = spread(required, ARGS.window, ARGS.sigma)
        current = translation + offsets
        for f in scored:   # exact clearing of anything the smoothed pushes still miss
            p = clearing_push(hands[f], rotation[f], current[f], scale, vertices, faces, center_local)
            offsets[f] += p
    resolved = translation + offsets
    after, frames_after = pen(resolved)
    data["object_translation"] = resolved.astype(np.float32)
    np.savez(ARGS.out / f"{tag}.npz", **data)
    sf = np.array(scored)
    inner = sf[(sf >= 1) & (sf < len(resolved) - 1)]
    acc = lambda t: float(np.linalg.norm(t[inner - 1] - 2 * t[inner] + t[inner + 1], axis=1).mean() * 100)  # noqa: E731
    return {"episode": tag, "dense_pen_cm": [before, after], "penetrating_frames": [frames_before, frames_after],
            "iterations": iterations, "max_push_cm": float(np.linalg.norm(offsets, axis=1).max() * 100),
            "object_acc_cm": [acc(translation), acc(resolved)]}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--hands", type=Path, required=True, help="mhr_bodies.py --hands-only output of --input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--margin_m", type=float, default=0.001)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--ray", action="store_true", help="push along the camera ray when that clears within 25 cm")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="correct only these episodes (others copied unchanged)")
    ap.add_argument("--cap_m", type=float, default=None, help="largest push; frames needing more keep their penetration")
    ARGS = ap.parse_args()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    for extra in ARGS.input.iterdir():
        if extra.is_file() and not extra.name.startswith("episode_"):
            shutil.copy(extra, ARGS.out / extra.name)
    paths = sorted(ARGS.input.glob("episode_??????.npz"))
    if ARGS.only:   # residual correction only in these episodes; the others are copied unchanged
        for p in paths:
            if int(p.stem[-6:]) not in ARGS.only:
                for f in ARGS.input.glob(f"{p.stem}*"):
                    shutil.copy2(f, ARGS.out / f.name)
        paths = [p for p in paths if int(p.stem[-6:]) in ARGS.only]
    with ProcessPoolExecutor(ARGS.workers) as pool:
        report = list(pool.map(episode, paths))
    for r in report:
        print(f"{r['episode']}: dense PEN {r['dense_pen_cm'][0]:.5f} -> {r['dense_pen_cm'][1]:.5f} cm, frames "
              f"{r['penetrating_frames'][0]} -> {r['penetrating_frames'][1]}, max push {r['max_push_cm']:.2f} cm, "
              f"object acc {r['object_acc_cm'][0]:.4f} -> {r['object_acc_cm'][1]:.4f}", flush=True)
    mean = {k: [float(np.mean([r[k][0] for r in report])), float(np.mean([r[k][1] for r in report]))]
            for k in ("dense_pen_cm", "object_acc_cm")}
    (ARGS.out / "pen_zero_report.json").write_text(json.dumps({"margin_m": ARGS.margin_m, "window": ARGS.window,
                                                               "sigma": ARGS.sigma, "mean": mean, "episodes": report}, indent=2))
    print(json.dumps(mean))
    print(f"PASS: penetration cleared in {sum(r['penetrating_frames'][1] == 0 for r in report)}/{len(report)} episodes")


if __name__ == "__main__":
    main()
