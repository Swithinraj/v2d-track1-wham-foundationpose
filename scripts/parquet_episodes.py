#!/usr/bin/env python3
"""Episode files from a Track 1 submission Parquet (submission venv, CPU).

A submission file is the only exact record of what Kaggle scored, and several scored candidates' intermediate
directories no longer exist. This turns such a file back into the episode layout every stage reads (NPZ indexed by video
frame plus the object mesh), so a change can be measured against a known score.

A submission only carries scored frames. Frames between scored stretches are bridged linearly (pose parameters and
translation) or by slerp (object rotation), and frames before the first or after the last scored frame hold the nearest
one. They are not scored, but smoothers and the contact stage read them. Budget padding (zero-area faces and vertices
they no longer use) is removed; the packer pads again.

Usage: parquet_episodes.py SUBMISSION.parquet OUT_DIR
Writes OUT_DIR/episode_*.npz, OUT_DIR/episode_*_object.glb and OUT_DIR/scored_frames.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

ARRAYS = ["mhr_pose", "mhr_scales", "mhr_shape", "object_rotation", "object_translation", "object_scale",
          "object_mesh_vertices", "object_mesh_faces"]
EPISODE_FRAME = 999999
PER_FRAME_POINTS = {"mhr_pose": 46, "object_rotation": 3, "object_translation": 1}
COUNTS = {"mhr_pose": 136, "mhr_scales": 68, "mhr_shape": 45}


def _once(block, name, episode):
    rows = block[block.array == ARRAYS.index(name)].sort_values("point")
    if rows.empty or not np.array_equal(rows.point.to_numpy(), np.arange(len(rows))):
        raise ValueError(f"episode {episode}: {name} rows are missing or out of order")
    return rows[["x", "y", "z"]].to_numpy(float)


def _per_frame(block, name, frames, episode):
    points = PER_FRAME_POINTS[name]
    rows = block[block.array == ARRAYS.index(name)].sort_values(["frame", "point"])
    if len(rows) != len(frames) * points:
        raise ValueError(f"episode {episode}: {name} has {len(rows)} rows, expected {len(frames) * points}")
    return rows[["x", "y", "z"]].to_numpy(float).reshape(len(frames), points, 3)


def _fill_linear(values, frames, T):
    """[len(frames), ...] -> [T, ...], linear between known frames, nearest beyond them."""
    flat = values.reshape(len(frames), -1)
    grid = np.arange(T)
    return np.stack([np.interp(grid, frames, flat[:, k]) for k in range(flat.shape[1])], axis=1).reshape(T, *values.shape[1:])


def _fill_rotation(rotation, frames, T):
    grid = np.clip(np.arange(T), frames[0], frames[-1])
    u, _, vt = np.linalg.svd(rotation)
    return Slerp(frames, Rotation.from_matrix(u @ vt))(grid).as_matrix()


def _clean_mesh(vertices, faces):
    faces = np.rint(faces).astype(np.int64)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 0
    faces = faces[keep]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def episodes_from_rows(frame):
    """{episode: arrays} from a submission's rows; a missing row raises rather than shifts later values."""
    parts = frame["row_id"].astype(str).str.split("_", expand=True)
    table = pd.DataFrame({"episode": parts[1].astype(int), "frame": parts[2].astype(int), "array": parts[3].astype(int),
                          "point": parts[4].astype(int), "x": frame["x"].to_numpy(float), "y": frame["y"].to_numpy(float),
                          "z": frame["z"].to_numpy(float)})
    out = {}
    for episode, block in table.groupby("episode", sort=True):
        per_frame = block[block.array == ARRAYS.index("mhr_pose")]
        frames = np.unique(per_frame.frame.to_numpy())
        if len(frames) == 0:
            raise ValueError(f"episode {episode}: no scored frames")
        pose = _per_frame(block, "mhr_pose", frames, episode).reshape(len(frames), -1)[:, :COUNTS["mhr_pose"]]
        rotation = _per_frame(block, "object_rotation", frames, episode)
        translation = _per_frame(block, "object_translation", frames, episode)[:, 0]
        T = int(frames[-1]) + 1
        vertices, faces = _clean_mesh(_once(block, "object_mesh_vertices", episode),
                                      _once(block, "object_mesh_faces", episode))
        out[int(episode)] = dict(
            frames=frames,
            pose=_fill_linear(pose, frames, T),
            scales=_once(block, "mhr_scales", episode).ravel()[:COUNTS["mhr_scales"]],
            shape=_once(block, "mhr_shape", episode).ravel()[:COUNTS["mhr_shape"]],
            object_rotation=_fill_rotation(rotation, frames, T),
            object_translation=_fill_linear(translation, frames, T),
            object_scale=float(_once(block, "object_scale", episode)[0, 0]),
            mesh_vertices=vertices,
            mesh_faces=faces,
        )
    return out


def write_episodes(episodes, out):
    import trimesh
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; refusing to overwrite an episode directory")
    out.mkdir(parents=True)
    scored = {}
    for episode, data in sorted(episodes.items()):
        tag = f"episode_{episode:06d}"
        np.savez(out / f"{tag}.npz", pose=data["pose"], scales=data["scales"], shape=data["shape"],
                 object_rotation=data["object_rotation"], object_translation=data["object_translation"],
                 object_scale=np.float64(data["object_scale"]))
        trimesh.Trimesh(data["mesh_vertices"], data["mesh_faces"], process=False).export(out / f"{tag}_object.glb")
        scored[str(episode)] = [int(f) for f in data["frames"]]
    (out / "scored_frames.json").write_text(json.dumps(scored))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    episodes = episodes_from_rows(pd.read_parquet(args.submission))
    write_episodes(episodes, args.out)
    print(f"wrote {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
