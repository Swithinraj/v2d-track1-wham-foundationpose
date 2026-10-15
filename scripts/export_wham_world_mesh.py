#!/usr/bin/env python3
"""Export the longest WHAM track as a full-frame world-coordinate SMPL mesh sequence.

Run this inside the WHAM container from its runtime working directory. Missing detection
frames are filled from the nearest observed frame so downstream adapters receive a dense
episode sequence. This export is SMPL topology and is not directly submission-compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from configs import constants as _C
from lib.models.smpl import SMPL


def nearest_observation(frame_ids: np.ndarray, total_frames: int) -> np.ndarray:
    if len(frame_ids) == 0:
        raise ValueError("WHAM track contains no observed frames")
    frames = np.arange(total_frames, dtype=np.int64)
    insertion = np.searchsorted(frame_ids, frames)
    right = np.clip(insertion, 0, len(frame_ids) - 1)
    left = np.clip(insertion - 1, 0, len(frame_ids) - 1)
    choose_right = np.abs(frame_ids[right] - frames) < np.abs(frame_ids[left] - frames)
    return np.where(choose_right, right, left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WHAM wham_output.pkl")
    parser.add_argument("--output", required=True, help="Output NPZ")
    parser.add_argument("--frames", required=True, type=int, help="Full episode frame count")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()

    results = joblib.load(args.input)
    if not results:
        raise SystemExit("ERROR: WHAM output contains no person tracks")
    subject = max(results, key=lambda key: len(results[key]["frame_ids"]))
    track = results[subject]
    frame_ids = np.asarray(track["frame_ids"], dtype=np.int64)
    order = np.argsort(frame_ids)
    frame_ids = frame_ids[order]
    if frame_ids[0] < 0 or frame_ids[-1] >= args.frames:
        raise SystemExit(f"ERROR: frame IDs outside [0, {args.frames}): {frame_ids[[0, -1]]}")

    dense_index = nearest_observation(frame_ids, args.frames)
    pose = np.asarray(track["pose_world"], dtype=np.float32)[order][dense_index]
    trans = np.asarray(track["trans_world"], dtype=np.float32)[order][dense_index]
    betas = np.asarray(track["betas"], dtype=np.float32)[order][dense_index]

    device = torch.device(args.device)
    model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender="neutral",
        batch_size=min(args.chunk, args.frames),
        create_transl=False,
    ).to(device)

    vertices: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, args.frames, args.chunk):
            stop = min(start + args.chunk, args.frames)
            output = model.get_output(
                body_pose=torch.from_numpy(pose[start:stop, 3:]).to(device),
                global_orient=torch.from_numpy(pose[start:stop, :3]).to(device),
                betas=torch.from_numpy(betas[start:stop]).to(device),
                transl=torch.from_numpy(trans[start:stop]).to(device),
            )
            vertices.append(output.vertices.detach().cpu().numpy().astype(np.float32))

    dense_vertices = np.concatenate(vertices)
    if dense_vertices.shape != (args.frames, 6890, 3):
        raise SystemExit(f"ERROR: unexpected SMPL vertices shape: {dense_vertices.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v2d.track1.wham_smpl_world.v1",
        "subject": str(subject),
        "total_frames": args.frames,
        "observed_frames": int(len(frame_ids)),
        "filled_frames": int(args.frames - len(np.unique(frame_ids))),
        "topology": "SMPL-6890",
        "submission_compatible": False,
    }
    np.savez_compressed(
        output_path,
        vertices=dense_vertices,
        pose_world=pose,
        trans_world=trans,
        betas=betas,
        source_frame_ids=frame_ids,
        dense_source_frame_ids=frame_ids[dense_index],
        report=json.dumps(report, sort_keys=True),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
