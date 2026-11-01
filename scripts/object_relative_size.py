#!/usr/bin/env python3
"""Size constancy among table episodes: scale each object about the camera to its table group's median size (host, CPU).

Kaggle showed that moving table objects closer on average costs a lot (+5.6 CD-O per episode at 0.974), so a scale calibrated
on the floor must not be carried over to them. What can still hold is the difference between episodes: the same object has
one size, so among its table episodes the one estimated larger is the one placed too far. Taking the median within the group
keeps the group's mean depth where it is. Floor episodes are not touched, and an object alone in its group stays as it is.

Usage: object_relative_size.py --input DIR --floor REPORT --out DIR [--strength 1]
  --floor: object_floor_scale_report.json; an episode it accepted counts as a floor episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from object_true_size import object_kind


def relative_scales(kinds, sizes, on_floor, strength=1.):
    """kappa per episode: 1 on the floor or alone in its group, else (group median / size) ** strength."""
    kinds, sizes, on_floor = np.asarray(kinds), np.asarray(sizes, float), np.asarray(on_floor, bool)
    if not (len(kinds) == len(sizes) == len(on_floor)) or not (sizes > 0).all():
        raise ValueError("kinds, sizes and on_floor must match, and sizes must be positive")
    kappa = np.ones(len(sizes))
    for kind in set(kinds.tolist()):
        group = (kinds == kind) & ~on_floor
        if group.sum() >= 2:
            kappa[group] = (np.median(sizes[group]) / sizes[group]) ** strength
    return kappa


def main():
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--floor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--episodes_meta", type=Path, default=Path("data/track_1/meta/episodes.jsonl"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    floor = json.loads(args.floor.read_text())
    tasks = {json.loads(line)["episode_index"]: json.loads(line)["tasks"][0] for line in args.episodes_meta.read_text().splitlines()}
    paths = sorted(args.input.glob("episode_??????.npz"))
    kinds, sizes, on_floor = [], [], []
    for path in paths:
        with np.load(path) as archive:
            scale = float(archive["object_scale"])
        mesh = trimesh.load(args.input / f"{path.stem}_object.glb", force="mesh", process=False)
        kinds.append(object_kind(tasks[int(path.stem.split("_")[1])]))
        sizes.append(float(np.linalg.norm(mesh.extents)) * scale)
        on_floor.append(bool(floor[path.stem].get("accepted")))
    kappas = relative_scales(kinds, sizes, on_floor, args.strength)
    shutil.copytree(args.input, args.out)
    report = {}
    for path, kind, kappa in zip(paths, kinds, kappas):
        report[path.stem] = dict(object=kind, kappa=float(kappa))
        if kappa != 1.:
            with np.load(path) as archive:
                data = dict(archive)
            data["object_translation"] = (data["object_translation"].astype(np.float64) * kappa).astype(data["object_translation"].dtype)
            data["object_scale"] = np.asarray(float(data["object_scale"]) * kappa, data["object_scale"].dtype)
            np.savez(args.out / path.name, **data)
    (args.out / "object_relative_size_report.json").write_text(json.dumps(report, indent=1) + "\n")
    changed = [k for k in kappas if k != 1.]
    print(f"PASS: relative size, {len(changed)} objects, range {min(changed):.3f}-{max(changed):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
