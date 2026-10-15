#!/usr/bin/env python3
"""Assemble per-episode Track 1 NPZ files for tools/pack_reconstruction.py (eval_reconstruction.py).

Joint alignment step of the WHAM + FoundationPose baseline. For every episode:

  human   outputs/human/episode_X/mhr_human.npz (WHAM world metres, MHR parameters), with the
          smoothed WHAM drift from outputs/alignment/episode_X/drift.npz added to the root
          translation. pose[:, :3] is in MHR units: +diag(1,-1,-1) * d * 10 moves the body by d m.
  object  outputs/object/episode_X/foundationpose/smoothed/*.json (object-to-camera, MoGe-2 metres)
          mapped into the drift-corrected WHAM world:
              p_world = R_cw^T (p_camera / depth_scale - t_cw)
          so object_rotation = R_cw^T R_co, object_translation = R_cw^T (t_co / depth_scale - t_cw),
          object_scale = s_co / depth_scale (the mesh stays in its own metric units).
  mesh    inputs/object_meshes/<object>.glb, copied as episode_X_object.glb.

Writes <out>/episode_XXXXXX.npz + episode_XXXXXX_object.glb and <out>/build_report.json.
Runs with numpy only (the submission venv or any container).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/work/smoses2s/nvidea_chal/TRACK1")
MHR_FLIP = np.array([1.0, -1.0, -1.0])
MHR_TRANSLATION_UNITS_PER_METRE = 10.0


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def episode_rows() -> dict[int, dict]:
    meta = ROOT / "data" / "track_1" / "meta"
    rows = {r["episode_index"]: r for r in map(json.loads, (meta / "episodes_metadata.jsonl").read_text().splitlines()) if r}
    for r in map(json.loads, (meta / "episodes.jsonl").read_text().splitlines()):
        if r:
            rows[r["episode_index"]]["length"] = r["length"]
    return rows


def build_episode(episode: int, row: dict, poses_subdir: str, out: Path) -> dict:
    tag = f"episode_{episode:06d}"
    frames = int(row["length"])
    human = np.load(ROOT / "outputs" / "human" / tag / "mhr_human.npz")
    alignment = json.loads((ROOT / "outputs" / "alignment" / tag / "alignment.json").read_text())
    drift_world = np.load(ROOT / "outputs" / "alignment" / tag / "drift.npz")["drift_world"]
    if len(human["pose"]) != frames or len(drift_world) != frames:
        raise SystemExit(f"{tag}: frame counts differ (video {frames}, human {len(human['pose'])}, drift {len(drift_world)})")

    pose = human["pose"].astype(np.float64).copy()
    pose[:, :3] += MHR_FLIP * drift_world * MHR_TRANSLATION_UNITS_PER_METRE

    r_cw = np.asarray(alignment["R_cw"], dtype=np.float64)
    t_cw = np.asarray(alignment["t_cw"], dtype=np.float64)
    depth_scale = float(alignment["depth_scale"])

    pose_dir = ROOT / "outputs" / "object" / tag / "foundationpose" / poses_subdir
    rotation = np.empty((frames, 3, 3))
    translation = np.empty((frames, 3))
    scales = np.empty(frames)
    for frame in range(frames):
        path = pose_dir / f"{frame:06d}.json"
        if not path.exists():
            raise SystemExit(f"{tag}: missing object pose {path}")
        record = json.loads(path.read_text())
        r_co = quaternion_wxyz_to_matrix(np.asarray(record["rotation"], dtype=np.float64))
        t_co = np.asarray(record["translation"], dtype=np.float64)
        s = np.asarray(record.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if np.ptp(s) > 1e-4 * np.abs(s).max():
            raise SystemExit(f"{tag}: anisotropic object scale {s} at frame {frame}")
        rotation[frame] = r_cw.T @ r_co
        translation[frame] = r_cw.T @ (t_co / depth_scale - t_cw)
        scales[frame] = s.mean()
    if np.ptp(scales) > 1e-4 * scales.max():
        raise SystemExit(f"{tag}: object scale varies over time ({scales.min()}..{scales.max()})")
    object_scale = float(scales.mean() / depth_scale)

    for name, value in (("pose", pose), ("object_rotation", rotation), ("object_translation", translation)):
        if not np.isfinite(value).all():
            raise SystemExit(f"{tag}: non-finite {name}")

    mesh = ROOT / "inputs" / "object_meshes" / f"{row['object']}.glb"
    shutil.copy(mesh, out / f"{tag}_object.glb")
    np.savez(out / f"{tag}.npz",
             pose=pose.astype(np.float32),
             scales=human["scales"].astype(np.float32),
             shape=human["shape"].astype(np.float32),
             object_rotation=rotation.astype(np.float32),
             object_translation=translation.astype(np.float32),
             object_scale=np.float32(object_scale))
    step = np.diff(translation, axis=0)
    return {
        "episode": episode, "object": row["object"], "frames": frames,
        "depth_scale": depth_scale, "object_scale": object_scale,
        "drift_max_m": float(np.linalg.norm(drift_world, axis=1).max()),
        "object_speed_p95_m_per_frame": float(np.percentile(np.linalg.norm(step, axis=1), 95)) if len(step) else 0.0,
        "reprojection_drift_corrected_median_px": alignment["pnp"]["reprojection_drift_corrected"]["median_px"],
        "mesh": str(mesh.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "submissions" / "episodes_wham_fp")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all 30")
    parser.add_argument("--poses", default="smoothed", choices=("smoothed", "raw"),
                        help="FoundationPose output subdirectory to use")
    args = parser.parse_args()

    rows = episode_rows()
    episodes = args.episodes if args.episodes is not None else sorted(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    report = [build_episode(e, rows[e], args.poses, args.out) for e in episodes]
    (args.out / "build_report.json").write_text(json.dumps({
        "schema": "v2d.track1.wham_fp_episodes.v1",
        "human": "WHAM -> MHR (official fitter), drift-corrected against static-camera keypoints",
        "object": f"TRELLIS generated mesh (not ground truth) + FoundationPose ({args.poses}) in MoGe-2 depth",
        "episodes": report,
    }, indent=2))
    for item in report:
        print(f"episode {item['episode']:2d} {item['object']:18s} depth_scale={item['depth_scale']:.3f} "
              f"object_scale={item['object_scale']:.3f} drift_max={item['drift_max_m']:.3f} m")
    print(f"PASS: wrote {len(report)} Track 1 episode files to {args.out}")


if __name__ == "__main__":
    main()
