#!/usr/bin/env python3
"""Assemble Track 1 episodes with the SAM 3D Body human in the static camera's frame (v2d_cari4d.sif: numpy + cv2).

World frame = the static camera, in SAM 3D Body's metric.
  human   outputs/sam3d_body/<episode>/refit.npz: pose refitted with one shared identity (scales 68 + shape 45)
  object  the FoundationPose camera-frame track (smoothed; the symmetry-aware track for symmetric objects) mapped
          with the episode's MoGe/SAM depth scale s (sam3d_body_identity.py): object_rotation = R_co,
          object_translation = t_co / s, object_scale = s_co / s (a camera alignment with R_cw = I, t_cw = 0,
          depth_scale = s). Symmetric objects (hula hoop) get the centre smoothing of
          smooth_symmetric_objects.py (lambda 100, weight 40 on frames where the mask rests).
  mesh    inputs/object_meshes/<object>.glb (<object>.symmetric.glb with the symmetric track).
Also (re)writes outputs/alignment_sam/<episode>/alignment.json (R_cw = I, t_cw = 0, depth_scale = s, the episode's
intrinsics, outputs/camera; first written by sam3d_body_hands.py) for the human refinement stage.
Usage: build_track1_episodes_sam.py --out DIR [--episodes ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_symmetric_object import resting_frames  # noqa: E402

REST_WEIGHT, HOOP_LAM = 40.0, 100.0


def smooth(p: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    T = len(p)
    D2 = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    return np.linalg.solve(np.diag(w) + lam * D2.T @ D2, w[:, None] * p)


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = ROOT / "data" / "track_1" / "meta" / "episodes_metadata.jsonl"
    objects = {r["episode_index"]: r["object"] for r in map(json.loads, meta.read_text().splitlines()) if r}
    report = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        work = ROOT / "outputs" / "sam3d_body" / tag
        refit = np.load(work / "refit.npz")
        s = float(np.load(work / "prepare.npz")["depth_scale"])
        T = len(refit["pose"])
        mesh = ROOT / "inputs" / "object_meshes" / f"{objects[ep]}.glb"
        sym = ROOT / "outputs" / "object" / tag / "foundationpose" / "symmetric"
        track = ROOT / "outputs" / "object" / tag / "foundationpose" / "smoothed"
        if sym.exists() and len(list(sym.glob("*.json"))) == T:
            track = sym
            mesh = mesh.with_name(f"{objects[ep]}.symmetric.glb")
        R = np.zeros((T, 3, 3))
        t = np.zeros((T, 3))
        sc = np.zeros(T)
        for f in range(T):
            rec = json.loads((track / f"{f:06d}.json").read_text())
            R[f] = quat_to_R(rec["rotation"])
            t[f] = rec["translation"]
            sc[f] = np.mean(rec.get("scale", [1, 1, 1]))
        trans = t / s
        if track.name == "symmetric":
            trans = smooth(trans, np.where(resting_frames(ep)[:T], REST_WEIGHT, 1.0), HOOP_LAM)
        np.savez(args.out / f"{tag}.npz", pose=refit["pose"].astype(np.float32),
                 scales=refit["identity"][:68].astype(np.float32), shape=refit["identity"][68:].astype(np.float32),
                 object_rotation=R.astype(np.float32), object_translation=trans.astype(np.float32),
                 object_scale=np.float32(sc.mean() / s))
        shutil.copy(mesh, args.out / f"{tag}_object.glb")
        intrinsics = json.loads((ROOT / "outputs" / "camera" / tag / "intrinsics.json").read_text())
        align = ROOT / "outputs" / "alignment_sam" / tag
        align.mkdir(parents=True, exist_ok=True)
        (align / "alignment.json").write_text(json.dumps({"R_cw": np.eye(3).tolist(), "t_cw": [0.0, 0.0, 0.0], "depth_scale": s,
                                                          "intrinsics": intrinsics, "note": "SAM 3D Body camera world"}))
        report[tag] = {"depth_scale": s, "track": track.name, "object_scale": float(sc.mean() / s), "mesh": str(mesh.relative_to(ROOT))}
        print(tag, json.dumps(report[tag]), flush=True)
    (args.out / "build_report.json").write_text(json.dumps({"human": "SAM 3D Body, shared identity, camera world",
                                                            "episodes": report}, indent=2))


if __name__ == "__main__":
    main()
