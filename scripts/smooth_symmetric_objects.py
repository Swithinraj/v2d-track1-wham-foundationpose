#!/usr/bin/env python3
"""Smooth the centre trajectory of symmetry-tracked objects (hula hoop) in assembled episodes (v2d_cari4d.sif).

The symmetry-aware tracker fits the ring to every frame's mask, which leaves broadband frame-to-frame
jitter on moving frames (episode 1: 2.7 cm/frame^2 mean acceleration on scored frames). Per episode
whose object uses the symmetric track, the world-frame centre x_t minimises

    sum_t w_t |x_t - p_t|^2 + LAM * sum_t |x_{t-1} - 2 x_t + x_{t+1}|^2

with p = the assembled track, w = 40 on frames where the object's mask is at rest (the tracker's own
resting_frames) and 1 elsewhere. Rotation, mesh and every other episode are unchanged. Coupling the
centre's velocity to the grasping hand instead was tested and lowered the silhouette IoU: the MHR hands
are noisier than the track.

Kaggle A/B against the same pipeline without this step (LAM 100): ACC-O 0.1729 -> 0.0769,
CD-O 29.91 -> 29.89 cm (LAM 10: 0.0868 / 29.91).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_symmetric_object import resting_frames  # noqa: E402

REST_WEIGHT = 40.0


def smooth(p: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    T = len(p)
    D2 = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    return np.linalg.solve(np.diag(w) + lam * D2.T @ D2, w[:, None] * p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lam", type=float, default=100.0)
    args = parser.parse_args()
    symmetric = {k for k in json.loads((ROOT / "config/symmetric_objects.json").read_text()) if not k.startswith("_")}
    objects = {r["episode_index"]: r["object"] for r in map(json.loads, (ROOT / "data/track_1/meta/episodes_metadata.jsonl").read_text().splitlines()) if r}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"lam": args.lam, "rest_weight": REST_WEIGHT, "episodes": {}}
    for path in sorted(args.input.glob("episode_*.npz")):
        tag = path.stem
        episode = int(tag.split("_")[1])
        data = dict(np.load(path))
        T = len(data["object_translation"])
        track = ROOT / "outputs/object" / tag / "foundationpose/symmetric"
        mesh = ROOT / "inputs/object_meshes" / f"{objects[episode]}.symmetric.glb"
        if objects[episode] in symmetric and mesh.exists() and len(list(track.glob("*.json"))) == T:
            p = data["object_translation"].astype(np.float64)
            w = np.where(resting_frames(episode)[:T], REST_WEIGHT, 1.0)
            x = smooth(p, w, args.lam)
            data["object_translation"] = x.astype(np.float32)
            acc = lambda y: float((np.linalg.norm(np.diff(y, n=2, axis=0), axis=1) * 100).mean())
            report["episodes"][tag] = {"acc_cm": [acc(p), acc(x)], "offset_max_cm": float(np.linalg.norm(x - p, axis=1).max() * 100)}
            print(tag, json.dumps(report["episodes"][tag]), flush=True)
        np.savez(args.out / f"{tag}.npz", **data)
        for m in args.input.glob(f"{tag}_object.*"):
            shutil.copy(m, args.out / m.name)
    (args.out / "smooth_symmetric_report.json").write_text(json.dumps(report, indent=2))
    print(f"symmetric-object smoothing: {len(report['episodes'])} episodes, lam {args.lam} -> {args.out}")


if __name__ == "__main__":
    main()
