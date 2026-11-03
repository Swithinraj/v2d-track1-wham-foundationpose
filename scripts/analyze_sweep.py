#!/usr/bin/env python3
"""Analyze local sweep results: calibrate local->Kaggle deltas, rank candidates.

Inputs:
  --scores DIR   with <name>.json from score_sweep.py ({"mean": {...}, ...})
  --known JSON   {name: {cd_h_cm: <kaggle>, ...}} for already-scored files
Outputs a per-board table:
  * delta calibration  Kaggle_delta = c * local_delta + b  (all pairs of known files)
  * every candidate's predicted Kaggle score  = kaggle(anchor) + c*(local - local(anchor))
    with anchor = known file nearest in local space (robust to reference bias), plus the
    raw absolute affine fit as a cross-check, and the implied Kaggle rank per board.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

BOARDS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm")
#: thresholds from the 2026-11-03 leaderboards (score -> best achievable rank)
RANK_TABLES = {
    "cd_h_cm": [11.83099, 12.63992, 12.65809, 12.67493, 12.91789, 12.94868, 13.07690,
                13.21103, 13.26540, 13.32564],
    "cd_o_cm": [15.53654, 15.70233, 16.18839, 16.37847, 16.39103, 16.44846, 17.08369,
                17.23590, 17.25432, 17.41469],
    "acc_h_cm": [0.13197, 0.13314, 0.13384, 0.13437, 0.13503, 0.13518, 0.13542, 0.13549,
                 0.13571, 0.13590],
    "acc_o_cm": [0.06809, 0.06824, 0.06833, 0.06870, 0.06894, 0.06930, 0.06970, 0.06971,
                 0.07051, 0.07053],
}


def implied_rank(score, table):
    return 1 + sum(1 for s in table if s < score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("locscores"))
    ap.add_argument("--known", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    known = json.loads(Path(a.known).read_text())

    local = {}
    for p in sorted(Path(a.scores).glob("*.json")):
        d = json.loads(p.read_text())
        if "mean" in d:
            local[p.stem] = d["mean"]
        else:
            local[p.stem] = {"error": d.get("error", "?")}
    print(f"{len(local)} candidates scored locally\n")

    # ---------- delta calibration per board ----------
    calib = {}
    for metric in BOARDS:
        pts = []
        names = [n for n in known if metric in known[n] and n in local and metric in local[n]]
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                dl = local[n2][metric] - local[n1][metric]
                dk = known[n2][metric] - known[n1][metric]
                if np.isfinite(dl) and np.isfinite(dk):
                    pts.append((dl, dk, n1, n2))
        if len(pts) < 6:
            print(f"{metric}: only {len(pts)} pairs — skipping calibration")
            continue
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        keep = np.abs(x - np.median(x)) < 5 * (np.std(x) + 1e-9)
        x, y = x[keep], y[keep]
        c, b = np.polyfit(x, y, 1)
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        calib[metric] = {"c": float(c), "b": float(b), "corr": corr, "n": int(len(x))}
        print(f"{metric}: c={c:.4f} b={b:+.5f} corr={corr:.3f} n={len(x)}")

    # ---------- predictions ----------
    print("\n== predicted Kaggle scores (anchor = nearest known file in local space) ==")
    lines = []
    for cand, lv in sorted(local.items()):
        if "error" in lv:
            print(f"{cand:>24s}: ERROR {lv['error'][:60]}")
            continue
        row = {"cand": cand, "pred": {}, "local": lv}
        for metric in BOARDS:
            if metric not in calib:
                continue
            anchors = [(n, known[n][metric], local[n][metric]) for n in known
                       if metric in known[n] and n in local and metric in local[n]
                       and metric in local[n]]
            if not anchors:
                continue
            n, kag, loc = min(anchors, key=lambda t: abs(t[2] - lv[metric]))
            c, b = calib[metric]["c"], calib[metric]["b"]
            pred = kag + c * (lv[metric] - loc) + 0.0
            row["pred"][metric] = {"pred": float(pred), "anchor": n,
                                   "rank": implied_rank(pred, RANK_TABLES[metric])}
        if row["pred"]:
            desc = "  ".join(f"{m.split('_')[0].upper()}={v['pred']:.5f}(r{v['rank']},a={v['anchor']})"
                             for m, v in row["pred"].items())
            print(f"{cand:>24s}: {desc}")
            lines.append(row)
    if a.out:
        a.out.write_text(json.dumps({"calib": calib, "rows": lines}, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
