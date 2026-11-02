#!/usr/bin/env python3
"""Rebuild per-episode npz files from a packed Kaggle Track-1 submission parquet (lossless inverse).

The pipeline's outputs/ and submissions/ were lost; the packed BMQ3KGMtp parquet contains the
complete episode state (official/v2d_submission_kit/tools/pack_reconstruction.py, pack_track1):
per episode mhr_scales [68], mhr_shape [45], object_scale, the budgeted object mesh (vertices
and faces), and per scored frame mhr_pose [136] + object_rotation [3x3] + object_translation.
This script unpacks those into <out>/episode_XXXXXX.npz (+ the mesh as _object.ply), with the
same keys the downstream tools (physics_objects.py, measure_floor_gap.py) read.

Verification is built in and decisive: the script re-packs its own reconstruction with the SAME
layout and requires an exact match against the input parquet — these are the values the scorer
saw, so a mismatch would mean we rebuilt a different candidate.

Coverage: the submitted state only (object motion + human MHR parameters on scored frames, in
the pipeline's own frame). Image-derived per-frame artifacts (masks, visibility, MoGe depth,
floor planes) are NOT in the parquet; object-side A/Bs (gravity, penetration, ACC-O
self-checks) are possible, re-running image stages is not.

Usage: unpack_submission.py --parquet FILE --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "official/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb import mhr_submission as ms  # noqa: E402

ARRAY_NAMES = ("mhr_pose", "mhr_scales", "mhr_shape", "object_scale",
               "object_mesh_vertices", "object_mesh_faces", "object_rotation", "object_translation")
PER_EPISODE = ("mhr_scales", "mhr_shape", "object_scale", "object_mesh_vertices", "object_mesh_faces")


def ms_pack(values, count):
    flat = np.zeros(3 * -(-count // 3))
    flat[:count] = np.asarray(values, dtype=float).reshape(count)
    return flat.reshape(-1, 3)


def blocks(data, mesh_v, mesh_f):
    per_episode = {
        "mhr_scales": ms_pack(data["mhr_scales"], 68),
        "mhr_shape": ms_pack(data["mhr_shape"], 45),
        "object_scale": np.array([[float(data["object_scale"]), 0.0, 0.0]]),
        "object_mesh_vertices": mesh_v,
        "object_mesh_faces": mesh_f.astype(float),
    }
    per_frame = {
        "mhr_pose": lambda f: ms_pack(data["pose"][f], 136),
        "object_rotation": lambda f: data["object_rotation"][f],
        "object_translation": lambda f: data["object_translation"][f][None],
    }
    return per_episode, per_frame


def repack_values(keys, order, episodes_dir):
    """Rebuild the x/y/z column exactly as pack_reconstruction.pack_track1 would."""
    values = np.full((len(keys), 3), np.nan)
    for episode, group in keys.groupby("episode"):
        data = dict(np.load(episodes_dir / f"episode_{episode:06d}.npz"))
        per_episode, per_frame = blocks(data, data["object_mesh_vertices"], data["object_mesh_faces"])
        for (frame, array), rows in group.groupby(["frame", "array"]):
            name = order[array]
            block = per_episode[name] if frame == ms.EPISODE_FRAME else per_frame[name](int(frame))
            values[rows.index] = block[rows["point"].to_numpy()]
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet)
    parts = df["row_id"].str.split("_", expand=True)
    keys = pd.DataFrame({"episode": parts[1].astype(int), "frame": parts[2].astype(int),
                         "array": parts[3].astype(int), "point": parts[4].astype(int)})
    keys = keys.join(df[["x", "y", "z"]])
    order = ms.array_order()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for episode, group in keys.groupby("episode"):
        ep_rows = group[group["frame"] == ms.EPISODE_FRAME]
        frames = sorted(group.loc[group["frame"] != ms.EPISODE_FRAME, "frame"].unique())
        T = frames[-1] + 1

        def episode_array(name):
            rows = ep_rows[ep_rows["array"] == order.index(name)].sort_values("point")
            return rows[["x", "y", "z"]].to_numpy(float)

        data = {"object_scale": np.array(episode_array("object_scale").reshape(-1)[0])}
        for name, count in (("mhr_scales", 68), ("mhr_shape", 45)):
            flat = episode_array(name).reshape(-1)
            data[name] = flat[:count]
        data["pose"] = np.full((T, 136), np.nan)
        data["object_rotation"] = np.full((T, 3, 3), np.nan)
        data["object_translation"] = np.full((T, 3), np.nan)
        for f in frames:
            fr = group[group["frame"] == f]
            data["pose"][f] = fr[fr["array"] == order.index("mhr_pose")].sort_values("point")[
                ["x", "y", "z"]].to_numpy(float).reshape(-1)[:136]
            data["object_rotation"][f] = fr[fr["array"] == order.index("object_rotation")].sort_values("point")[
                ["x", "y", "z"]].to_numpy(float).reshape(3, 3)
            data["object_translation"][f] = fr[fr["array"] == order.index("object_translation")].sort_values("point")[
                ["x", "y", "z"]].to_numpy(float)[0]
        v_rows = episode_array("object_mesh_vertices")
        f_rows = episode_array("object_mesh_faces")
        # unscored frames stay NaN in the arrays; the round trip only writes scored frames.
        # The mesh is stored in the npz too: PLY would round vertices to float32 and break the
        # exact round trip; the .ply copy is only for trimesh-consuming tools.
        data["object_mesh_vertices"], data["object_mesh_faces"] = v_rows, f_rows.astype(np.int64)
        import trimesh
        mesh = trimesh.Trimesh(vertices=v_rows, faces=f_rows.astype(np.int64), process=False)
        mesh.export(a.out / f"episode_{episode:06d}_object.ply")
        np.savez(a.out / f"episode_{episode:06d}.npz", **data)
        report[f"episode_{episode:06d}"] = {"scored_frames": len(frames),
                                            "mesh_vertices": int(len(v_rows)), "mesh_faces": int(len(f_rows))}
        print(f"  episode_{episode:06d}: {len(frames)} scored frames, "
              f"mesh {len(v_rows)}v/{len(f_rows)}f", flush=True)

    # decisive check: re-pack and compare exactly against the input parquet
    values = repack_values(keys, order, a.out)
    src = df[["x", "y", "z"]].to_numpy(float)
    bad = ~np.isclose(values, src, rtol=0, atol=1e-9) & ~(np.isnan(values) & np.isnan(src))
    n_bad = int(bad.sum())
    if n_bad or np.isnan(values).any():
        print(f"FAIL: {n_bad} value(s) differ on the round trip "
              f"({int(np.isnan(values).sum())} unpacked as NaN) — reconstruction is not exact")
        raise SystemExit(1)
    (a.out / "unpack_report.json").write_text(json.dumps(report, indent=1))
    print(f"PASS: round-trip exact; {len(report)} episodes, "
          f"{sum(r['scored_frames'] for r in report.values())} scored frames -> {a.out}")


if __name__ == "__main__":
    main()
