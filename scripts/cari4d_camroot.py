#!/usr/bin/env python3
"""Camera files for scripts/true_cam_params.py, describing CARI4D's own camera (host, stdlib + numpy).

CARI4D runs on MoGe-2's camera for each video (principal point at the image centre), not the camera the rest of this repository
recorded. true_cam_params.py reads its old camera from a root directory, so this writes one that holds CARI4D's, plus a new camera
keeping CARI4D's focal with the rig's measured principal point (mode tcp, the variant that Kaggle confirmed).

Usage: cari4d_camroot.py --export DIR --k_measured outputs/camera_measured/k_new.json --out DIR
Writes OUT/outputs/camera/<tag>/intrinsics.json, OUT/outputs/object/<tag>/{prompt/reference_frame.txt,intrinsics/000000.json}
and OUT/k_new.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def camera_entries(matrix, measured):
    """(old, new) camera dictionaries from CARI4D's [3,3] matrix and the measured camera."""
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("expected a valid 3x3 camera matrix")
    old = dict(fx=float(matrix[0, 0]), fy=float(matrix[1, 1]), cx=float(matrix[0, 2]), cy=float(matrix[1, 2]))
    new = dict(old, cx=float(measured["cx"]), cy=float(measured["cy"]))
    return old, new


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--k_measured", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.k_measured.read_text())
    k_new = {}
    for path in sorted(args.export.glob("episode_??????.object.npz")):
        tag = path.name.split(".")[0]
        with np.load(path) as archive:
            old, new = camera_entries(archive["camera"], measured[tag])
        k_new[tag] = new
        camera = args.out / "outputs/camera" / tag
        camera.mkdir(parents=True, exist_ok=True)
        (camera / "intrinsics.json").write_text(json.dumps(old))
        objects = args.out / "outputs/object" / tag
        (objects / "prompt").mkdir(parents=True, exist_ok=True)
        (objects / "intrinsics").mkdir(parents=True, exist_ok=True)
        (objects / "prompt/reference_frame.txt").write_text("0\n")
        (objects / "intrinsics/000000.json").write_text(json.dumps(old))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "k_new.json").write_text(json.dumps(k_new, indent=1))
    print(f"PASS: {len(k_new)} CARI4D cameras -> {args.out}")


if __name__ == "__main__":
    main()
