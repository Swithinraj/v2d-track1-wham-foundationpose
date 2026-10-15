#!/usr/bin/env python3
"""Extract the canonical SMPL triangle array from the user's licensed model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    with model_path.open("rb") as handle:
        try:
            model = pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            model = pickle.load(handle)
    faces = model["f"] if isinstance(model, dict) else model.f
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 10_000:
        raise SystemExit(f"ERROR: invalid SMPL triangle array: {faces.shape}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, faces)
    print(f"PASS: extracted {len(faces)} SMPL faces to {output}")


if __name__ == "__main__":
    main()
