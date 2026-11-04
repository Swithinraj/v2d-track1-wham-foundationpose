#!/usr/bin/env python3
"""Largest absolute difference per array between two episode directories (reproduction checks).
Usage: compare_episode_dirs.py DIR_A DIR_B [DIR_A2 DIR_B2 ...]"""
import sys
from pathlib import Path

import numpy as np

pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
for a, b in pairs:
    worst, n = {}, 0
    for f in sorted(Path(a).glob("episode_??????.npz")):
        x, y = np.load(f), np.load(Path(b) / f.name); n += 1
        for k in x.files:
            worst[k] = max(worst.get(k, 0.0), float(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max()))
    print(f"{Path(a).name} vs {Path(b).name} ({n} episodes):", {k: round(v, 6) for k, v in worst.items()}, flush=True)
