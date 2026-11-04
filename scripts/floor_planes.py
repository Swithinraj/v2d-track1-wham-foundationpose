#!/usr/bin/env python3
"""Floor normals in the measured rig camera (host, stdlib + numpy; see pipeline/recipes/floor.md).

MoGe-2 gives each episode a floor normal in its own camera, whose principal point is wrong. Changing the camera keeps
every pixel and maps 3D points linearly (X' = A X, scripts/true_cam_params.py), so a plane stays a plane and its normal
maps by the inverse transpose of A. A per-episode MoGe normal also carries about 0.3 degrees of noise, and 1 degree is
about 4 cm of false depth per metre walked. So each normal is first averaged over the episodes of the same rig camera
whose normals agree within 1 degree, which is the same camera mounting.

Usage: floor_planes.py --moge config/floor/kappa_floor.json --out outputs/floor/planes_measured.json
Writes {episode: {normal, normal_moge, support, spread_deg}}. The normal points up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAME_MOUNTING_DEG = 1.0


def camera_map(old, new):
    """A in X' = A X, for the same pixels and depth reinterpreted from camera old to camera new."""
    return np.array([[1, 0, (old["cx"] - new["cx"]) / old["fx"]],
                     [0, 1, (old["cy"] - new["cy"]) / old["fx"]],
                     [0, 0, new["fx"] / old["fx"]]], float)


def map_normal(normal, matrix):
    mapped = np.linalg.inv(matrix).T @ np.asarray(normal, float)
    return mapped / np.linalg.norm(mapped)


def session_normals(normals, cameras):
    """Mean normal over the episodes of the same camera within SAME_MOUNTING_DEG, with its support and spread."""
    normals = np.asarray(normals, float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cameras = np.asarray(cameras)
    out = []
    for k in range(len(normals)):
        angle = np.degrees(np.arccos(np.clip(normals @ normals[k], -1, 1)))
        same = np.flatnonzero((cameras == cameras[k]) & (angle < SAME_MOUNTING_DEG))
        mean = normals[same].mean(0)
        mean /= np.linalg.norm(mean)
        spread = float(np.degrees(np.arccos(np.clip(normals[same] @ mean, -1, 1))).max())
        out.append((mean, len(same), spread))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--moge", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=Path("config/camera/episode_cameras.json"))
    parser.add_argument("--old_cameras", type=Path, default=Path("outputs/camera"))
    parser.add_argument("--k_new", type=Path, default=Path("outputs/camera_measured/k_new.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    planes = json.loads(args.moge.read_text())
    identity = json.loads(args.identity.read_text())
    new = json.loads(args.k_new.read_text())
    tags = sorted(planes)
    means = session_normals([planes[t]["normal"] for t in tags], [identity[t]["camera"] for t in tags])
    result = {}
    for tag, (mean, support, spread) in zip(tags, means):
        old = json.loads((args.old_cameras / tag / "intrinsics.json").read_text())
        result[tag] = dict(normal=map_normal(mean, camera_map(old, new[tag])).tolist(), normal_moge=mean.tolist(),
                           support=support, spread_deg=spread)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"PASS: {len(result)} floor normals -> {args.out}")


if __name__ == "__main__":
    main()
