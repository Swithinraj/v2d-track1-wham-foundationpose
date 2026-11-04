#!/usr/bin/env python3
"""Episodes = the human of one build (pose, scales, shape) with the objects of another (submission venv, CPU).

Both builds share the camera world (same alignment), so a human solved separately can be paired with separately solved
objects. Object arrays and meshes come from OBJECTS unchanged.
Writes <out>/episode_*.npz (+ the object meshes and shared files of OBJECTS).
Usage: combine_human_objects.py HUMAN_DIR OBJECTS_DIR OUT_DIR
"""
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    hd, od, out = map(Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
    for f in od.iterdir():
        if f.is_file() and not f.name.endswith(".npz"):
            shutil.copy2(f, out / f.name)
    for f in sorted(od.glob("episode_*.npz")):
        o = dict(np.load(f)); h = np.load(hd / f.name)
        for k in ("pose", "scales", "shape"):
            o[k] = h[k]
        np.savez(out / f.name, **o)
    print("combined", len(list(out.glob("episode_*.npz"))), "episodes")


if __name__ == "__main__":
    main()
