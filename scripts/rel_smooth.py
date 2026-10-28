"""Smooth the depth-reliability module's object corrections again over time (host venv: numpy).

The module's per-frame object correction c_o (scripts/rel_train.py --dump) is Gaussian-smoothed with SIGMA frames
(edge-padded) and re-zeroed at the anchor frame; the original is kept as c_o_raw. SIGMA 20 was chosen on the FORM-HOI
validation group: CD-O nearly unchanged against the unsmoothed correction, object acceleration error below the base
(see pipeline/recipes/rel.md).
Usage: rel_smooth.py DUMP_DIR [--sigma 20]
"""
import argparse
from pathlib import Path

import numpy as np


def gsmooth(x, sigma):
    if sigma <= 0:
        return x
    k = int(3 * sigma)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    g /= g.sum()
    return np.convolve(np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])]), g, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--sigma", type=float, default=20.0)
    args = ap.parse_args()
    n = 0
    for f in sorted(args.dump.glob("*.npz")):
        z = dict(np.load(f))
        raw = z.get("c_o_raw", z["c_o"]).astype(np.float64)
        c = gsmooth(raw, args.sigma)
        a = int(z["anchor"])
        c = c - c[a]
        c[a] = 0.0
        z["c_o_raw"] = raw.astype(np.float32)
        z["c_o"] = c.astype(np.float32)
        np.savez(f, **z)
        n += 1
    print(f"PASS: object corrections smoothed (sigma {args.sigma:g} frames) for {n} episodes")


if __name__ == "__main__":
    main()
