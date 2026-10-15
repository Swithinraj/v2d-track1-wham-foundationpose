#!/usr/bin/env python3
"""Check SAM2 object-mask completeness and basic temporal sanity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.masks)
    missing: list[int] = []
    fractions: list[float] = []
    ious: list[float] = []
    previous: np.ndarray | None = None

    for frame in range(args.frames):
        path = root / f"{frame:06d}.png"
        if not path.is_file():
            missing.append(frame)
            previous = None
            continue
        mask = np.asarray(Image.open(path).convert("L")) > 0
        fractions.append(float(mask.mean()))
        if previous is not None:
            union = np.logical_or(previous, mask).sum()
            intersection = np.logical_and(previous, mask).sum()
            ious.append(float(intersection / union) if union else 1.0)
        previous = mask

    values = np.asarray(fractions, dtype=np.float64)
    report = {
        "schema": "v2d.track1.object_mask_audit.v1",
        "expected_frames": args.frames,
        "mask_files": len(fractions),
        "missing_frames": missing,
        "empty_frames": int((values == 0).sum()) if len(values) else args.frames,
        "tiny_frames": int((values < 0.0001).sum()) if len(values) else args.frames,
        "huge_frames": int((values > 0.65).sum()) if len(values) else 0,
        "area_fraction_min": float(values.min()) if len(values) else None,
        "area_fraction_median": float(np.median(values)) if len(values) else None,
        "area_fraction_max": float(values.max()) if len(values) else None,
        "consecutive_iou_median": float(np.median(ious)) if ious else None,
        "requires_visual_review": bool(
            missing or not len(values) or (values == 0).any() or (values > 0.65).any()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} SAM2 mask frames are missing")


if __name__ == "__main__":
    main()
