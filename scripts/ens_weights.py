#!/usr/bin/env python3
"""Validate the ensemble-geometry model on a known ensemble, then choose member weights.

Model (ambiguity decomposition, exact for squared Euclidean errors): the error of a weighted
average of members is
    E(w) = sum_i w_i CD_i^2 - sum_{i<j} w_i w_j d_ij^2,     CD_ens = sqrt(E(w))
with CD_i the members' Kaggle CD-H and d_ij their pairwise scorer-style Chamfer distance
(ens_geometry.py; first-frame similarity alignment, symmetrised).

Validation (no fitted parameter): ENS3 = equal-weight average of P2, MF, C1 has a Kaggle score;
its predicted score, and the predicted vs measured distances from ENS3 to each member, test the
model before any weight is chosen.

usage: ens_weights.py --geom DIR --scores kaggle_known_scores.json --pool P4 P2 C1 ...
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def load_geom(geom_dir, key):
    """Per-episode JSONs from ens_geometry.py -> symmetric distance table (mean over episodes)."""
    per = {int(p.stem.split("_")[1]): json.loads(p.read_text())["pairs"]
           for p in sorted(Path(geom_dir).glob("episode_*.json"))}
    eps = sorted(per)
    pairs = per[eps[0]].keys()
    agg = {k: float(np.mean([per[e][k][key] for e in eps])) for k in pairs}
    names = sorted({x for k in agg for x in k.split("|")})
    D = {}
    for k, v in agg.items():
        i, j = k.split("|")
        D[i, j] = D[j, i] = v
    return names, D, len(eps)


def predict(w, cd2, D2):
    n = len(w)
    amb = sum(w[i] * w[j] * D2[i, j] for i in range(n) for j in range(i + 1, n))
    return float(np.dot(w, cd2) - amb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geom", type=Path, required=True, help="dir of per-episode JSONs")
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--known-ens", default="ENS3")
    ap.add_argument("--known-members", nargs="+", default=["P2", "MF", "C1"])
    ap.add_argument("--key", default="cd", choices=("cd", "mvd"))
    a = ap.parse_args()
    K = json.loads(a.scores.read_text())
    names, D, neps = load_geom(a.geom, a.key)
    print(f"geometry: {len(names)} chains, {neps} episodes, metric={a.key}")
    print("pairwise distance (cm):")
    print("       " + " ".join(f"{n:>6}" for n in names))
    for i in names:
        print(f"{i:>6} " + " ".join(f"{D[i, j]:6.2f}" if i != j else "     -" for j in names))

    # --- validation 1: Euclidean consistency of the distances themselves
    mem = a.known_members
    n = len(mem)
    s_all = sum(D[x, y] ** 2 for x, y in itertools.combinations(mem, 2))
    print(f"\n[check 1] distance from {a.known_ens} to each member: measured vs Euclidean prediction")
    for x in mem:
        pred = np.sqrt(sum(D[x, y] ** 2 for y in mem if y != x) / n - s_all / n ** 2)
        print(f"  {x:>4}: measured {D[a.known_ens, x]:.3f}   predicted {pred:.3f}")

    # --- validation 2: zero-parameter prediction of the known ensemble's Kaggle CD-H
    cd2 = np.array([K[x]["cd_h_cm"] ** 2 for x in mem])
    D2 = {(i, j): D[mem[i], mem[j]] ** 2 for i in range(n) for j in range(n) if i != j}
    w = np.full(n, 1.0 / n)
    e = predict(w, cd2, D2)
    print(f"\n[check 2] {a.known_ens} Kaggle CD-H: predicted {np.sqrt(e):.4f}   actual {K[a.known_ens]['cd_h_cm']:.4f}"
          f"   (members {', '.join(f'{x} {K[x]['cd_h_cm']:.3f}' for x in mem)})")

    # --- weights over the pool
    pool = [p for p in a.pool if p in names and p in K and "cd_h_cm" in K[p]]
    m = len(pool)
    cd2 = np.array([K[p]["cd_h_cm"] ** 2 for p in pool])
    D2 = {(i, j): D[pool[i], pool[j]] ** 2 for i in range(m) for j in range(m) if i != j}
    res = minimize(lambda w: predict(w, cd2, D2), np.full(m, 1.0 / m), method="SLSQP",
                   bounds=[(0, 1)] * m, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    wopt = np.where(res.x < 1e-4, 0, res.x)
    wopt /= wopt.sum()
    print(f"\n[optimum over {pool}]")
    for p, wi in sorted(zip(pool, wopt), key=lambda t: -t[1]):
        if wi > 0:
            print(f"  {p:>5} w={wi:.3f}  (Kaggle {K[p]['cd_h_cm']:.4f})")
    print(f"  predicted CD-H {np.sqrt(predict(wopt, cd2, D2)):.4f}")
    print("\n[simple ensembles]")
    best = pool[int(np.argmin(cd2))]
    for k in range(2, min(m, 4) + 1):
        for combo in itertools.combinations(pool, k):
            if best not in combo:
                continue
            idx = [pool.index(c) for c in combo]
            sub2 = {(x, y): D2[idx[x], idx[y]] for x in range(k) for y in range(k) if x != y}
            r = minimize(lambda w: predict(w, cd2[idx], sub2), np.full(k, 1.0 / k), method="SLSQP",
                         bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
            eq = np.sqrt(predict(np.full(k, 1.0 / k), cd2[idx], sub2))
            print(f"  {'+'.join(combo):24} equal {eq:.4f}  | opt {np.sqrt(r.fun):.4f} w={np.round(r.x, 2).tolist()}")


if __name__ == "__main__":
    main()
