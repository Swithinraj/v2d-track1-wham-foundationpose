#!/usr/bin/env python3
"""Patch ONLY the object_translation rows of an existing submission parquet from a
variant episode-npz dir (arrays indexed by scored-frame order, as unpack_parquet emits).
Everything else — human rows, meshes, rotations, scales, row_id set — stays bit-identical.
usage: patch_translations.py --base WF.parquet --variant DIR --out OUT.parquet [--commit URL]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRANS_IDX = 4  # array_order: mhr_pose, scales, shape, rotation, TRANSLATION, scale, verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.base)
    parts = df["row_id"].str.split("_", expand=True)
    is_trans = parts[3].astype(int) == TRANS_IDX
    patched = 0
    for ep in sorted(parts[1][is_trans].astype(int).unique()):
        z = np.load(Path(a.variant) / f"episode_{ep:06d}.npz")
        t = np.asarray(z["object_translation"], dtype=np.float64)
        sel = ((parts[1].astype(int) == ep) & is_trans).to_numpy()
        rows = df[sel]
        assert len(rows) == len(t), (len(rows), len(t))
        frm = parts[2][sel].astype(int).to_numpy()
        order = np.argsort(frm, kind="stable")          # rows sorted by frame ordinal
        df.loc[rows.index[order], ["x", "y", "z"]] = t
        patched += int(sel.sum())
    if a.commit:
        df["code_commit_url"] = a.commit
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {patched} translation rows patched "
          f"({len(df)} rows total)")


if __name__ == "__main__":
    main()
