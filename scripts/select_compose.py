#!/usr/bin/env python3
"""Compose a per-episode-best candidate dir from sweep results (CD-H selection).

Reads locscores/*.json, picks the best chain per episode for one metric (default
cd_h_cm) among an allow-list of chains (to avoid pseudo-GT constituents), and writes
a new episode npz dir taking the HUMAN arrays (mhr_pose/scales/shape + frames) from
the winning chain and the OBJECT arrays from a fixed object chain. Object glbs are
copied from the object chain's glb dir for packing.

usage: select_compose.py --scores DIR --metric cd_h_cm --allow P3 P4 V5 ... \
    --human-root /path/calib_npz --object-chain P4tab102 --out DIR [--glbs DIR]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

HUMAN_KEYS = ("mhr_pose", "mhr_scales", "mhr_shape", "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--metric", default="cd_h_cm")
    ap.add_argument("--allow", nargs="+", required=True)
    ap.add_argument("--human-root", type=Path, required=True)
    ap.add_argument("--object-chain", default="P4tab102")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--glbs", type=Path, default=None, help="dir with episode_*_object.glb")
    a = ap.parse_args()

    per_ep = {}   # ep -> {chain: metric}
    for p in Path(a.scores).glob("*.json"):
        d = json.loads(p.read_text())
        if "per_episode" not in d or p.stem not in a.allow:
            continue
        for ep, m in d["per_episode"].items():
            if a.metric in m:
                per_ep.setdefault(int(ep), {})[p.stem] = m[a.metric]
    if not per_ep:
        raise SystemExit("no per-episode scores found")
    a.out.mkdir(parents=True, exist_ok=True)
    obj_root = Path(a.human_root) / a.object_chain
    picks = {}
    for ep in sorted(per_ep):
        cand = {c: v for c, v in per_ep[ep].items() if (Path(a.human_root) / c).exists()}
        if not cand:
            print(f"ep{ep}: no allowed chain with local data, fallback {a.object_chain}")
            winner, chain = a.object_chain, a.object_chain
        else:
            chain = min(cand, key=cand.get)
            winner = chain
        picks[ep] = (winner, cand.get(winner, float("nan")))
        h = dict(np.load(Path(a.human_root) / chain / f"episode_{ep:06d}.npz"))
        o = dict(np.load(obj_root / f"episode_{ep:06d}.npz"))
        out = {k: h[k] for k in HUMAN_KEYS if k in h}
        out.update({k: o[k] for k in o if k.startswith("object")})
        np.savez(a.out / f"episode_{ep:06d}.npz", **out)
    print("per-episode picks:")
    for ep, (w, v) in picks.items():
        print(f"  ep{ep:02d}: {w:>16s} {v:.4f}")
    from collections import Counter
    print("chain usage:", dict(Counter(w for w, _ in picks.values())))
    if a.glbs:
        for ep in picks:
            g = Path(a.glbs) / f"episode_{ep:06d}_object.glb"
            if g.exists():
                shutil.copy2(g, a.out / g.name)
        print(f"copied glbs from {a.glbs}")


if __name__ == "__main__":
    main()
