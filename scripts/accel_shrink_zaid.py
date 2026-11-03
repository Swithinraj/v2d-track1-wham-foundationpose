#!/usr/bin/env python3
"""Zaid's exact accel_shrink (vpq-vitpose/scripts/accel_shrink.py), operating on submission parquets.

The scored-stretch form makes this a pure function of the scored-frame translations in the
parquet: within each contiguous run of scored frames, out = whittaker(x, 1, lam) + k (x -
whittaker), with a constant offset pinning the stretch's first frame (the alignment frame).
Verification target: WF -> WFk60 (k=0.6, lam=1e4, Kaggle ACC-O 0.07051 -> 0.06971).

usage: accel_shrink_zaid.py --base WF.parquet --k 0.7 [--lam 1e4] [--out OUT.parquet]
                            [--verify WFk60.parquet] [--commit URL]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded

TRANS_IDX = 4  # array_order: pose, scales, shape, rotation, TRANSLATION, scale, verts, faces


def whittaker(values, weights, lam):
    """argmin sum w (x - y)^2 + lam |D2 x|^2 for [T] or [T, K] values; pentadiagonal banded solve."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    T = len(values)
    if T < 3:
        return values.copy()
    band = np.zeros((3, T))
    band[2] = weights + lam * np.convolve(np.ones(T - 2), [1.0, 4.0, 1.0])
    band[1, 1:] = lam * -2 * np.convolve(np.ones(T - 2), [1.0, 1.0])
    band[0, 2:] = lam
    right = weights[:, None] * values if values.ndim == 2 else weights * values
    return solveh_banded(band, right)


def stretches(frames):
    """Positional index arrays of maximal runs of consecutive video-frame ids."""
    frames = np.asarray(frames)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    return np.split(np.arange(len(frames)), breaks)


def shrink(track, frames, k, lam=1e4):
    """[T, D] scored-frame track (rows in scored order, `frames` = their video ids) ->
    copy whose stretches carry k times the residual from the Whittaker smooth;
    first scored frame pinned (alignment unchanged)."""
    track = np.asarray(track, np.float64)
    out = track.copy()
    for part in stretches(frames):
        x = track[part]
        smooth = whittaker(x, np.ones(len(x)), lam)
        out[part] = smooth + k * (x - smooth)
    out += track[0] - out[0]
    return out


def accel(track, frames):
    errs = []
    for part in stretches(frames):
        if len(part) < 3:
            continue
        x = np.asarray(track)[part]
        norm = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)
        errs.append(np.r_[norm[:1], norm, norm[-1:]])
    return float(np.concatenate(errs).mean()) if errs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--lam", type=float, default=1e4)
    ap.add_argument("--verify", type=Path, default=None,
                    help="compare against this parquet instead of writing (e.g. WFk60)")
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.base)
    parts = df["row_id"].str.split("_", expand=True)
    is_trans = (parts[3].astype(int) == TRANS_IDX).to_numpy()

    new = df.copy()
    report = {}
    for ep in sorted(parts[1][is_trans].astype(int).unique()):
        sel = ((parts[1].astype(int) == ep) & is_trans).to_numpy()
        rows = df[sel]
        frm = parts[2][sel].astype(int).to_numpy()
        order = np.argsort(frm, kind="stable")
        t = rows[["x", "y", "z"]].to_numpy(np.float64)[order]      # scored-frame order
        t2 = shrink(t, frm, a.k, a.lam)      # stretches = runs of consecutive video-frame ids
        if a.verify is not None:
            ref = pd.read_parquet(a.verify)
            rparts = ref["row_id"].str.split("_", expand=True)
            rsel = ((rparts[1].astype(int) == ep) & (rparts[3].astype(int) == TRANS_IDX)).to_numpy()
            rrows = ref[rsel]
            rfrm = rparts[2][rsel].astype(int).to_numpy()
            rorder = np.argsort(rfrm, kind="stable")
            tref = rrows[["x", "y", "z"]].to_numpy(np.float64)[rorder]
            report[ep] = {"max_diff": float(np.abs(t2 - tref).max()),
                          "k_achieved": accel(t2, frm) / max(accel(t, frm), 1e-12)}
        else:
            idx = rows.index[order]
            new.loc[idx, ["x", "y", "z"]] = t2
            report[ep] = {"k_achieved": accel(t2, frm) / max(accel(t, frm), 1e-12)}

    ratios = np.array([r["k_achieved"] for r in report.values()])
    print(f"k={a.k} lam={a.lam}: achieved ratio median {np.median(ratios):.3f} "
          f"range [{ratios.min():.3f}, {ratios.max():.3f}] over {len(report)} eps")
    if a.verify is not None:
        dmax = max(r["max_diff"] for r in report.values())
        print(f"verification vs {a.verify.name}: max |diff| = {dmax:.3e} m "
              f"({'REPRODUCED' if dmax < 1e-4 else 'MISMATCH'})")
    else:
        if a.commit:
            new["code_commit_url"] = a.commit
        new.to_parquet(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
