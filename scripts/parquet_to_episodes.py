#!/usr/bin/env python3
"""Unpack a Track-1 submission parquet into a packer-ready episodes dir (video-frame npz + glbs).

Inverse of tools/pack_reconstruction.py for the fields the packer consumes, mirroring the
verified unpack in localscore.unpack_parquet: per-frame scalars are packed across x/y/z
(3 scalars per point row), and arrays are re-indexed from scored frames to video frames
using the frame ids in row_id. Feeds the container's outputs/<label>/reconstructions/.

usage: parquet_to_episodes.py PARQUET OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

KEYMAP = {"mhr_pose": "pose", "mhr_scales": "scales", "mhr_shape": "shape"}
PER_FRAME = {"mhr_pose": 136, "object_rotation": 9, "object_translation": 3}
EPISODE_SCALARS = {"mhr_scales": 68, "mhr_shape": 45}
NAMES = {0: "mhr_pose", 1: "mhr_scales", 2: "mhr_shape", 3: "object_rotation",
         4: "object_translation", 5: "object_scale", 6: "verts", 7: "faces"}


def unpack(parquet: Path):
    df = pd.read_parquet(parquet, columns=["row_id", "x", "y", "z"])
    parts = df["row_id"].str.split("_", expand=True)
    df = df.assign(ep=parts[1].astype(int), frm=parts[2].astype(int),
                   arr=parts[3].astype(int), pt=parts[4].astype(int))
    scored_by_ep = {int(ep): np.sort(g["frm"].astype(int).unique())
                    for ep, g in df[df["arr"] == 0].groupby("ep")}
    eps = {}
    for ep, block in df.groupby("ep"):
        d = {}
        for arr, sub in block.groupby("arr"):
            name = NAMES[int(arr)]
            sub = sub.sort_values(["frm", "pt"])
            if name in PER_FRAME:
                rows = int(sub["pt"].max()) + 1
                T = len(scored_by_ep[int(ep)])
                vals = sub[["x", "y", "z"]].to_numpy(np.float64).reshape(T, rows, 3)
                d[name] = vals.reshape(T, -1)[:, : PER_FRAME[name]]
                if name == "object_rotation":
                    d[name] = d[name].reshape(T, 3, 3)
            elif name in EPISODE_SCALARS:
                rows = int(sub["pt"].max()) + 1
                vals = sub[["x", "y", "z"]].to_numpy(np.float64).reshape(rows, 3)
                d[name] = vals.reshape(-1)[: EPISODE_SCALARS[name]]
            elif name == "object_scale":
                d[name] = float(sub["x"].iloc[0])
            elif name == "verts":
                d[name] = sub[["x", "y", "z"]].to_numpy(np.float64)
            elif name == "faces":
                d[name] = sub[["x", "y", "z"]].to_numpy(np.int64)
        eps[int(ep)] = d
    return eps, scored_by_ep


def main(parquet: Path, out: Path):
    eps, scored_by_ep = unpack(parquet)
    out.mkdir(parents=True, exist_ok=True)
    for ep, d in sorted(eps.items()):
        scored = scored_by_ep[ep]
        T = int(scored[-1]) + 1
        vid = np.arange(T)
        mask = np.isin(vid, scored)
        assert mask.sum() == len(scored)
        out_ep = {}
        for k, v in d.items():
            k2 = KEYMAP.get(k, k)
            if k in ("mhr_pose", "object_rotation", "object_translation"):
                full = np.zeros((T,) + v.shape[1:], dtype=np.float64)
                full[mask] = v
                out_ep[k2] = full
            else:
                out_ep[k2] = v
        np.savez(out / f"episode_{ep:06d}.npz", **out_ep)
        if "verts" in d and "faces" in d:
            m = trimesh.Trimesh(vertices=d["verts"], faces=d["faces"], process=False)
            m.export(out / f"episode_{ep:06d}_object.glb")
    print(f"wrote {out}: {len(eps)} episodes, {len(list(out.glob('*_object.glb')))} meshes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
