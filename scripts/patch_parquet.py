#!/usr/bin/env python3
"""Patch selected arrays of a packed Track-1 parquet; every other row stays bit-identical.

Subcommands
  npz       --variant DIR --arrays NAME [NAME ...]
            Copy the named arrays from a variant episode-npz dir. Both npz layouts are accepted:
            scored-frames-only (localscore/make_variants: `frames` + `mhr_pose`, rows in
            scored-frame order) and full-length (unpack_submission/kit: `pose` indexed by video
            frame id, unscored frames NaN). Per-frame arrays: mhr_pose, object_rotation,
            object_translation. Per-episode: mhr_scales, mhr_shape, object_scale.
  camscale  --scale S --episodes E [E ...]
            Camera-centred object scale: object_translation *= S and object_scale *= S on the
            listed episodes. The image projection of every object point is unchanged (an object
            S times larger at S times the depth projects identically); only its metric size and
            depth relative to the human move. This is the operation behind P4tab101/P4tab102
            (S = 1.01/1.02 on the 13 'table' episodes).

usage:
  patch_parquet.py --base B.parquet --out O.parquet [--commit URL] npz --variant DIR --arrays mhr_pose
  patch_parquet.py --base B.parquet --out O.parquet [--commit URL] camscale --scale 1.02 --episodes 0 2 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[1] / "official/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb import mhr_submission as ms  # noqa: E402

ORDER = ms.array_order()
EP_FRAME = ms.EPISODE_FRAME
COUNTS = {"mhr_pose": 136, "mhr_scales": 68, "mhr_shape": 45}
PER_FRAME = ("mhr_pose", "object_rotation", "object_translation")
PER_EPISODE = ("mhr_scales", "mhr_shape", "object_scale")


def pack3(values, count):
    flat = np.zeros(3 * -(-count // 3))
    flat[:count] = np.asarray(values, dtype=float).reshape(count)
    return flat.reshape(-1, 3)


def keys_of(df):
    p = df["row_id"].str.split("_", expand=True)
    return pd.DataFrame({"episode": p[1].astype(int).to_numpy(), "frame": p[2].astype(int).to_numpy(),
                         "array": p[3].astype(int).to_numpy(), "point": p[4].astype(int).to_numpy()},
                        index=df.index)


def frame_getter(z, name):
    """Return f(frame_id) -> per-frame value for either npz layout."""
    if "frames" in z:
        frames = np.asarray(z["frames"]).astype(int)
        pos = {int(f): i for i, f in enumerate(frames)}
        arr = np.asarray(z[name], dtype=np.float64)
        return lambda f: arr[pos[f]]
    key = "pose" if name == "mhr_pose" and "pose" in z else name
    arr = np.asarray(z[key], dtype=np.float64)
    return lambda f: arr[f]


def block(name, value):
    if name in COUNTS:
        return pack3(value, COUNTS[name])
    if name == "object_rotation":
        return np.asarray(value, dtype=float).reshape(3, 3)
    if name == "object_translation":
        return np.asarray(value, dtype=float).reshape(1, 3)
    if name == "object_scale":
        return np.array([[float(np.asarray(value).reshape(-1)[0]), 0.0, 0.0]])
    raise SystemExit(f"unsupported array {name}")


def patch_npz(df, keys, variant: Path, arrays):
    vals = df[["x", "y", "z"]].to_numpy(float).copy()
    changed = 0
    for ep, g in keys.groupby("episode"):
        z = dict(np.load(variant / f"episode_{ep:06d}.npz"))
        if "frames" not in z and "mhr_pose" in z and "pose" not in z:
            raise SystemExit(f"episode {ep}: npz has mhr_pose but no frames; ambiguous layout")
        for name in arrays:
            sel = g[g["array"] == ORDER.index(name)]
            if name in PER_EPISODE:
                key = name if name in z else {"mhr_scales": "scales", "mhr_shape": "shape"}.get(name, name)
                b = block(name, z[key])
                idx = sel.index.to_numpy()
                new = b[sel["point"].to_numpy()]
            else:
                get = frame_getter(z, name)
                idx, new = [], []
                for f, rows in sel.groupby("frame"):
                    b = block(name, get(int(f)))
                    idx.append(rows.index.to_numpy())
                    new.append(b[rows["point"].to_numpy()])
                idx, new = np.concatenate(idx), np.concatenate(new)
            pos = df.index.get_indexer(idx)
            if not np.isfinite(new).all():
                raise SystemExit(f"episode {ep} {name}: non-finite values in variant")
            changed += int((~np.isclose(vals[pos], new, rtol=0, atol=0)).any(axis=1).sum())
            vals[pos] = new
    return vals, changed


def patch_camscale(df, keys, scale, episodes):
    vals = df[["x", "y", "z"]].to_numpy(float).copy()
    ep_mask = keys["episode"].isin(episodes).to_numpy()
    t = ep_mask & (keys["array"] == ORDER.index("object_translation")).to_numpy()
    s = ep_mask & (keys["array"] == ORDER.index("object_scale")).to_numpy() & (keys["point"] == 0).to_numpy()
    vals[t] *= scale
    vals[s, 0] *= scale
    return vals, int(t.sum() + s.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", default=None, help="stamp code_commit_url (default: keep base's)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("npz")
    p.add_argument("--variant", type=Path, required=True)
    p.add_argument("--arrays", nargs="+", required=True, choices=PER_FRAME + PER_EPISODE)
    p = sub.add_parser("camscale")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.base)
    keys = keys_of(df)
    if a.cmd == "npz":
        vals, n = patch_npz(df, keys, a.variant, a.arrays)
        what = f"{n} rows changed in {a.arrays}"
    else:
        vals, n = patch_camscale(df, keys, a.scale, a.episodes)
        what = f"{n} rows scaled x{a.scale} on episodes {a.episodes}"
    out = df.copy()
    out[["x", "y", "z"]] = vals
    if a.commit:
        out["code_commit_url"] = a.commit
    assert list(out.columns) == list(df.columns) and len(out) == len(df)
    assert out["row_id"].equals(df["row_id"])
    assert np.isfinite(out[["x", "y", "z"]].to_numpy(float)).all()
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {what} ({len(out)} rows)")


if __name__ == "__main__":
    main()
