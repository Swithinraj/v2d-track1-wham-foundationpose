#!/usr/bin/env python3
"""Signed distance field of the actor mask, the independent check of the lateral root fix (v2d_cari4d.sif: numpy, cv2).

Every STRIDE-th frame of outputs/person/<episode>/sam2_masks/0 (pipeline/stages/person.sh): distance in pixels to
the mask boundary, positive outside the actor, negative inside, area-downsampled by DOWN ->
outputs/person/<episode>/mask_sdf.npz (frames [N], sdf [N, H/DOWN, W/DOWN] float16, NaN for empty masks; down).
Usage: person_mask_sdf.py [--episodes E ...] [--stride 5] [--down 4]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/track1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--down", type=int, default=4)
    args = ap.parse_args()
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        files = sorted((ROOT / "outputs/person" / tag / "sam2_masks/0").glob("*.png"))
        frames, fields = [], []
        for path in files[::args.stride]:
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
            h, w = m.shape
            size = (w // args.down, h // args.down)
            if not m.any():
                sdf = np.full(size[::-1], np.nan, np.float32)
            else:
                outside = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
                inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                sdf = cv2.resize(outside - inside, size, interpolation=cv2.INTER_AREA)
            frames.append(int(path.stem)); fields.append(sdf.astype(np.float16))
        np.savez_compressed(ROOT / "outputs/person" / tag / "mask_sdf.npz", frames=np.array(frames), sdf=np.stack(fields),
                            down=args.down)
        print(f"{tag}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
