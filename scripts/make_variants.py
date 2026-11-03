#!/usr/bin/env python3
"""Generate candidate variants for local pre-scoring.

Variants (all stretch-wise: scored frames may have gaps; smoothing never crosses them):
  smooth-pose  --sigma S [--col-start K]   zero-phase Gaussian on MHR pose columns
                                           (default cols 6:122 — root translation/rotation
                                           untouched; identity scales/shape never touched)
  smooth-trans --sigma S [--alpha A]       zero-phase Gaussian (+ optional global scale)
                                           on object_translation only
  frame0       --sigma S                   replace frame 0's pose with the smoothed pose
                                           (denoises the Sim(3) alignment anchor; all other
                                           frames bit-identical)

Object arrays and meshes are carried over from the source unchanged (mesh rows come from
the source npz; pack-time glbs are provided separately).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def stretches(frames):
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    edges = [0, *breaks.tolist(), len(frames)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


def smooth_stretchwise(x: np.ndarray, frames: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian along time, independently within each contiguous stretch."""
    out = x.copy()
    for a, b in stretches(frames):
        seg = x[a:b]
        s = sigma
        while s >= 1 and (b - a) < int(4 * s) + 1:
            s /= 2.0
        if s < 1:
            continue
        out[a:b] = gaussian_filter1d(seg, s, axis=0, mode="nearest")
    return out


def load_src(src: Path):
    eps = {}
    for p in sorted(Path(src).glob("episode_*.npz")):
        eps[int(p.stem.split("_")[1])] = dict(np.load(p))
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--src", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--episodes", type=int, nargs="*", default=None)

    p = sub.add_parser("smooth-pose")
    common(p)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--col-start", type=int, default=6)
    p.add_argument("--col-end", type=int, default=122)

    p = sub.add_parser("smooth-trans")
    common(p)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--alpha", type=float, default=1.0)

    p = sub.add_parser("frame0")
    common(p)
    p.add_argument("--sigma", type=float, default=3.0)

    a = ap.parse_args()
    eps = load_src(a.src)
    eps = {e: d for e, d in eps.items() if a.episodes is None or e in a.episodes}
    a.out.mkdir(parents=True, exist_ok=True)
    for ep, d in sorted(eps.items()):
        frames = d.get("frames")
        if frames is None:
            frames = np.arange(len(d["mhr_pose"]))
        d = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
        if a.cmd == "smooth-pose":
            pose = d["mhr_pose"].astype(np.float64)
            pose[:, a.col_start:a.col_end] = smooth_stretchwise(
                pose[:, a.col_start:a.col_end], frames, a.sigma)
            d["mhr_pose"] = pose
        elif a.cmd == "smooth-trans":
            t = d["object_translation"].astype(np.float64)
            t = smooth_stretchwise(t, frames, a.sigma)
            if a.alpha != 1.0:
                t = t * a.alpha
            d["object_translation"] = t
        elif a.cmd == "frame0":
            pose = d["mhr_pose"].astype(np.float64)
            sm = smooth_stretchwise(pose, frames, a.sigma)
            pose[0] = sm[0]
            d["mhr_pose"] = pose
        np.savez(a.out / f"episode_{ep:06d}.npz", **d)
    print(f"wrote {a.out} ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
