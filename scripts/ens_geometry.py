#!/usr/bin/env python3
"""Pairwise scorer-style human distances between chains (research for ensemble weighting).

For every unordered pair (i, j), i before j in the given order, and every episode: fit the kit's similarity transform from
chain i's first scored frame to chain j's (exactly as the scorer aligns a submission to the GT),
apply it to all of i's frames, and measure
  cd   the kit's symmetric Chamfer (cm, all vertices as the surface, every `step`-th frame),
  msv  the mean squared vertex distance (cm^2; same MHR topology; all frames),
  mvd  the mean vertex distance (cm; all frames).
Per-episode means are written; the scorer's aggregate is the mean over episodes.

Use: if chamfer distances behave like a Euclidean geometry, the error of a weighted average of
members is  sum_i w_i CD_i^2 - sum_{i<j} w_i w_j d_ij^2  (ambiguity decomposition), with CD_i the
members' Kaggle CD-H and d_ij measured here. A known ensemble's Kaggle score validates that.

usage: ens_geometry.py --chain NAME=DIR [--chain ...] --out DIR [--step 6]
                       [--shard I --nshards N]
One JSON per episode is written to DIR as soon as it is done (resumable: existing files are skipped).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("LOCALSCORE_DIR", str(Path(__file__).resolve().parent)))
import localscore as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", action="append", required=True,
                    help="NAME=DIR, an npz dir with frames, mhr_pose, mhr_scales, mhr_shape")
    ap.add_argument("--out", type=Path, required=True, help="output dir, one JSON per episode")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chains = dict(c.split("=", 1) for c in a.chain)
    names = list(chains)
    allep = a.episodes if a.episodes else list(range(30))
    sizes = {e: len(np.load(Path(chains[names[0]]) / f"episode_{e:06d}.npz")["frames"]) for e in allep}
    loads, owner = [0] * a.nshards, {}
    for e in sorted(allep, key=lambda e: -sizes[e]):          # longest-first balance by frame count
        k = int(np.argmin(loads)); owner[e] = k; loads[k] += sizes[e]
    eps = sorted(e for e in allep if owner[e] == a.shard)
    a.out.mkdir(parents=True, exist_ok=True)
    m = L.kit()
    for ep in eps:
        dst = a.out / f"episode_{ep:06d}.json"
        if dst.exists():
            print(f"ep{ep}: exists, skip", flush=True)
            continue
        t0 = time.time()
        V, F = {}, {}
        for n, p in chains.items():
            d = dict(np.load(Path(p) / f"episode_{ep:06d}.npz"))
            V[n], _ = L.forward_chain(d, dev)
            F[n] = np.asarray(d["frames"])
        f0 = F[names[0]]
        assert all(np.array_equal(F[n], f0) for n in names), f"ep{ep}: frame sets differ"
        idx = torch.arange(0, len(f0), a.step, device=V[names[0]].device)
        r = {}
        with torch.no_grad():
            for ii, i in enumerate(names):
                for j in names[ii + 1:]:
                    tr = m.fit_similarity_transform(V[i][0], V[j][0])
                    vi = m.apply_similarity_transform(V[i], tr)
                    cd = m.symmetric_chamfer_distance(vi[idx], V[j][idx], chunk_size=4096) * 100.0
                    diff = vi - V[j]
                    r[f"{i}|{j}"] = {"cd": float(cd.mean()),
                                     "msv": float((diff ** 2).sum(-1).mean() * 1e4),
                                     "mvd": float(diff.norm(dim=-1).mean() * 100.0)}
        dst.write_text(json.dumps({"T": int(len(f0)), "pairs": r}))
        print(f"ep{ep}: T={len(f0)} chains={len(names)} {time.time() - t0:.0f}s", flush=True)
    print(f"DONE shard {a.shard}: episodes {eps} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
