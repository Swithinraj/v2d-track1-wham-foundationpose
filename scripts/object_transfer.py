#!/usr/bin/env python3
"""Keep an object's placement relative to the person when the person changes (host, CPU). No learned parameters.

The scorer places the object through a similarity fitted on the person at the first scored frame, so CD-O sees only where the
object is relative to the person there. So a better person need not disturb the objects: fit the same similarity from the
source person to the target person at that frame, and carry the source's object through it. CD-O then stays what it was with
the source's person and objects, and CD-H is the target person's.

This separates the two boards' best estimates. On Kaggle the measured camera improves the person (CD-H -0.20) but costs the
objects (CD-O +0.80 on BMQ3KGM).

For the scorer's row-vector transform X' = s (X @ R) + t and the object pose X = s_o R_o v + t_o:
  R_o' = R^T R_o,  s_o' = s s_o,  t_o' = s R^T t_o + t.

Usage: object_transfer.py --people DIR --people_meshes DIR --objects DIR --objects_meshes DIR --out DIR
  --people: episodes whose person is kept; --objects: episodes whose object is kept (with its mesh).
  *_meshes: ensemble_meshes.py output of those directories [scored frames, V, 3]; only the first frame is used.
Writes OUT/episode_*.npz and OUT/episode_*_object.glb and object_transfer_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from human_root_mix import similarity


def transfer_object(rotation, translation, scale, similarity_transform):
    """Object pose after X' = s (X @ R) + t. rotation [T,3,3], translation [T,3], scale s_o."""
    s, R, t = similarity_transform
    rotation, translation = np.asarray(rotation, float), np.asarray(translation, float)
    return R.T @ rotation, s * (translation @ R) + t, s * float(scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--people", type=Path, required=True)
    parser.add_argument("--people_meshes", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--objects_meshes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    args.out.mkdir(parents=True)
    report = {}
    for path in sorted(args.people.glob("episode_??????.npz")):
        tag = path.stem
        with np.load(path) as archive:
            data = dict(archive)
        with np.load(args.objects / path.name) as archive:
            source = {key: archive[key] for key in ("object_rotation", "object_translation", "object_scale")}
        T = len(data["pose"])
        if len(source["object_rotation"]) < T:
            raise ValueError(f"{tag}: objects cover {len(source['object_rotation'])} frames, the person {T}")
        fit = similarity(np.load(args.objects_meshes / f"{tag}.npy", mmap_mode="r")[0].astype(float),
                         np.load(args.people_meshes / f"{tag}.npy", mmap_mode="r")[0].astype(float))
        rotation, translation, scale = transfer_object(source["object_rotation"][:T], source["object_translation"][:T],
                                                       float(source["object_scale"]), fit)
        data.update(object_rotation=rotation.astype(np.float64), object_translation=translation.astype(np.float64), object_scale=np.float64(scale))
        np.savez(args.out / path.name, **data)
        shutil.copy2(args.objects / f"{tag}_object.glb", args.out / f"{tag}_object.glb")
        report[tag] = dict(scale=float(fit[0]), rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(fit[1]) - 1) / 2, -1, 1)))))
    (args.out / "object_transfer_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"PASS: {len(report)} objects carried to the new person -> {args.out}")


if __name__ == "__main__":
    main()
