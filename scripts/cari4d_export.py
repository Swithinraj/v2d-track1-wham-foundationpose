#!/usr/bin/env python3
"""Decode a CARI4D bundle's human and object for the scored frames (v2d_cari4d.sif, GPU).

Stage choices (--stage):
  init      the initial estimate (SAM 3D Body, FoundationPose)
  coconet   CoCoNet's prediction, before the final optimiser
  refined   after the contact-guided optimiser
In CARI4D's own ablation the optimiser made CD-H worse while making motion smoother, so each stage is worth scoring.

The human leaves as MHR-topology vertices, which the kit's mesh_to_mhr_params.py turns into the scored parameters, because
the bundle's continuous parameters are not the scorer's pose parameters. The object leaves as rotation and translation for
the pipeline's aligned mesh, with scale 1.

Usage: cari4d_export.py --root OUT_DIR --episode E --stage coconet --out DIR
Writes OUT/episode_*.verts.npy [scored frames, 18439, 3] in metres (Track 1 frame), OUT/episode_*.object.npz and
OUT/episode_*_object.glb (a copy of the aligned mesh).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

STAGES = {"init": ("coconet.pth", "in"), "coconet": ("coconet.pth", "pr_initial"), "refined": ("refined.pth", "pr")}


def split_pose(pose_abs):
    """[T,4,4] rigid transforms -> rotation [T,3,3], translation [T,3]; rejects anything but a proper rigid transform."""
    pose_abs = np.asarray(pose_abs, float)
    if pose_abs.ndim != 3 or pose_abs.shape[1:] != (4, 4) or not np.isfinite(pose_abs).all():
        raise ValueError("object poses must be finite [T,4,4]")
    rotation = pose_abs[:, :3, :3]
    if not np.allclose(rotation @ rotation.swapaxes(1, 2), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("object poses must be rigid")
    return rotation, pose_abs[:, :3, 3]


def main():
    import torch
    from lib_mhr import MHRLayer
    from lib_mhr.schema import MHR_PARAM_DIMS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=Path("/track1/outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    tag = f"episode_{args.episode:06d}"
    file_name, block_name = STAGES[args.stage]
    bundle = torch.load(args.root / tag / "inference" / file_name, map_location="cpu", weights_only=False)
    frames = np.asarray(json.loads(args.scored.read_text())[str(args.episode)])
    block = bundle[block_name]
    T = len(bundle["frames"])
    if frames[-1] >= T:
        raise ValueError(f"{tag}: scored frame {frames[-1]} is outside the {T} frames of the bundle")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT"), device=device)
    kept = []
    for start in range(0, len(frames), args.batch):
        index = frames[start:start + args.batch]
        params = {key: torch.as_tensor(np.asarray(block[key])[index], device=device, dtype=torch.float) for key in MHR_PARAM_DIMS}
        with torch.inference_mode():
            kept.append(layer.mhr_forward(params).vertices.detach().cpu().numpy().astype(np.float32))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{tag}.verts.npy", np.concatenate(kept))
    if "pose_abs" in block:
        pose = np.asarray(block["pose_abs"])
    else:
        pose = np.eye(4)[None].repeat(T, 0)
        pose[:, :3, :3] = np.asarray(block["obj_rot"])
        pose[:, :3, 3] = np.asarray(block["obj_t"])
    rotation, translation = split_pose(pose)
    camera = np.asarray(bundle["observations"]["K_full"])[0]
    np.savez(args.out / f"{tag}.object.npz", object_rotation=rotation, object_translation=translation, frames=frames, camera=camera)
    shutil.copy2(args.root / tag / "export" / tag / "object_mesh/output_aligned.glb", args.out / f"{tag}_object.glb")
    print(f"PASS: {tag} {args.stage}: {len(frames)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
