#!/usr/bin/env python3
"""One physical object has one size: per-object depth from size constancy across episodes (host, CPU). No learned parameters.

Scaling an object about the camera leaves every pixel in place, so its estimated size and its estimated depth are tied: an
object estimated too large is also too far. The same object appears in several episodes. Episodes where it touches the
floor calibrate its true size (scripts/object_floor_scale.py), and that size then gives the depth scale for every other
episode of the object, including those where it only ever sits on a table, where the floor gives no evidence.

Within objects, this cue and the floor agree: correlation 0.55 over 15 comparisons (scripts/analysis/size_constancy.py).
An object with no floor evidence in any episode keeps its median size times DEFAULT_KAPPA, the mean scale of the objects
that do have evidence.

Usage: object_true_size.py --input DIR --floor REPORT --out DIR [--default_kappa 0.974] [--sizes_out JSON]
  --floor: object_floor_scale_report.json measured on --input. Writes OUT/episode_*.npz (+ other files of --input),
  object_true_size_report.json and, with --sizes_out, the calibrated size per object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np

KINDS = (("hula_hoop", r"hula hoop"), ("big_red_bowl", r"big red bowl"), ("iron", r"\biron\b"), ("white_desk", r"white desk"),
         ("black_pan", r"\bpan\b"), ("foam_grass_block", r"\bblock\b"), ("pink_foam_roll", r"foam roll"),
         ("paint_roller", r"\broller\b"), ("short_wood_stool", r"\bstool\b"), ("white_laptop_cart", r"laptop cart"))
MAX_CHANGE = 0.15


def object_kind(task):
    """The object name in a task description. The first match wins, so 'foam roll' is tested before 'roller'."""
    text = task.lower()
    for name, pattern in KINDS:
        if re.search(pattern, text):
            return name
    raise ValueError(f"no known object in task: {task}")


def calibrated_sizes(kinds, sizes, floor_kappas, default_kappa):
    """(size per object, kappa per episode). floor_kappas holds NaN where the floor gave no evidence."""
    kinds = np.asarray(kinds)
    sizes, floor_kappas = np.asarray(sizes, float), np.asarray(floor_kappas, float)
    if len(kinds) != len(sizes) or len(sizes) != len(floor_kappas) or not (sizes > 0).all():
        raise ValueError("kinds, sizes and floor kappas must match, and sizes must be positive")
    true_size = {}
    for kind in sorted(set(kinds.tolist())):
        members = kinds == kind
        evidence = np.isfinite(floor_kappas) & members
        if evidence.any():
            true_size[kind] = float(np.median(sizes[evidence] * floor_kappas[evidence]))
        else:
            true_size[kind] = float(np.median(sizes[members]) * default_kappa)
    kappa = np.array([true_size[kind] / size for kind, size in zip(kinds.tolist(), sizes)])
    return true_size, kappa


def main():
    import trimesh
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--floor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--default_kappa", type=float, default=0.974)
    parser.add_argument("--episodes_meta", type=Path, default=Path("data/track_1/meta/episodes.jsonl"))
    parser.add_argument("--sizes_out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    floor = json.loads(args.floor.read_text())
    tasks = {json.loads(line)["episode_index"]: json.loads(line)["tasks"][0] for line in args.episodes_meta.read_text().splitlines()}
    paths = sorted(args.input.glob("episode_??????.npz"))
    kinds, sizes, floors = [], [], []
    for path in paths:
        episode = int(path.stem.split("_")[1])
        with np.load(path) as archive:
            scale = float(archive["object_scale"])
        mesh = trimesh.load(args.input / f"{path.stem}_object.glb", force="mesh", process=False)
        kinds.append(object_kind(tasks[episode]))
        sizes.append(float(np.linalg.norm(mesh.extents)) * scale)
        entry = floor[path.stem]
        # A fixed or default scale recorded in the report is not evidence; only accepted floor contact is.
        floors.append(entry["kappa_proposed"] if entry.get("accepted") else np.nan)
    true_size, kappas = calibrated_sizes(kinds, sizes, floors, args.default_kappa)
    shutil.copytree(args.input, args.out)
    report = {}
    for path, kind, size, kappa in zip(paths, kinds, sizes, kappas):
        applied = abs(kappa - 1) <= MAX_CHANGE
        report[path.stem] = dict(object=kind, size_m=size, kappa=float(kappa if applied else 1.), applied=bool(applied))
        if applied:
            with np.load(path) as archive:
                data = dict(archive)
            data["object_translation"] = (data["object_translation"].astype(np.float64) * kappa).astype(data["object_translation"].dtype)
            data["object_scale"] = np.asarray(float(data["object_scale"]) * kappa, data["object_scale"].dtype)
            np.savez(args.out / path.name, **data)
    (args.out / "object_true_size_report.json").write_text(json.dumps(report, indent=1) + "\n")
    if args.sizes_out:
        args.sizes_out.parent.mkdir(parents=True, exist_ok=True)
        args.sizes_out.write_text(json.dumps(true_size, indent=1) + "\n")
    values = [r["kappa"] for r in report.values()]
    print(f"PASS: size constancy, {sum(r['applied'] for r in report.values())}/{len(report)} applied, kappa median {np.median(values):.3f} "
          f"range {min(values):.3f}-{max(values):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
