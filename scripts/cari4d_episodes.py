#!/usr/bin/env python3
"""Episode files from exported CARI4D stages and converted human parameters (submission venv, CPU).

Takes cari4d_export.py's object poses and the kit converter's human parameters, both over scored frames only, and writes the
layout the stages and the packer read, indexed by video frame. Frames between scored stretches are bridged as in
scripts/parquet_episodes.py: they are not scored, but smoothers and the contact stage read them.

Usage: cari4d_episodes.py --export DIR --params DIR --out DIR
  --params: DIR/episode_*.npz from mesh_to_mhr_params.py (pose [F,136], scales, shape).
Writes OUT/episode_*.npz, OUT/episode_*_object.glb and OUT/scored_frames.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from parquet_episodes import _fill_linear, _fill_rotation


def build_episode(frames, pose, scales, shape, rotation, translation):
    """Arrays over the video's frames from scored-frame arrays; the bundle's rotation and translation cover every frame."""
    frames = np.asarray(frames)
    pose = np.asarray(pose, float)
    if len(pose) != len(frames):
        raise ValueError(f"pose has {len(pose)} frames, expected {len(frames)} scored frames")
    T = int(frames[-1]) + 1
    if len(rotation) < T or len(translation) < T:
        raise ValueError("object poses must cover every frame up to the last scored one")
    rotation = np.asarray(rotation[:T], float)
    translation = np.asarray(translation[:T], float)
    return dict(pose=_fill_linear(pose, frames, T), scales=np.asarray(scales, float), shape=np.asarray(shape, float),
                object_rotation=_fill_rotation(rotation[frames], frames, T), object_translation=_fill_linear(translation[frames], frames, T),
                object_scale=np.float64(1.))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an episode directory")
    args.out.mkdir(parents=True)
    scored = {}
    for path in sorted(args.export.glob("episode_??????.object.npz")):
        tag = path.name.split(".")[0]
        with np.load(path) as archive:
            frames, rotation, translation = archive["frames"], archive["object_rotation"], archive["object_translation"]
        with np.load(args.params / f"{tag}.npz") as archive:
            data = build_episode(frames, archive["pose"], archive["scales"], archive["shape"], rotation, translation)
        np.savez(args.out / f"{tag}.npz", **data)
        shutil.copy2(args.export / f"{tag}_object.glb", args.out / f"{tag}_object.glb")
        scored[str(int(tag.split("_")[1]))] = [int(f) for f in frames]
    (args.out / "scored_frames.json").write_text(json.dumps(scored))
    print(f"PASS: {len(scored)} CARI4D episodes -> {args.out}")


if __name__ == "__main__":
    main()
