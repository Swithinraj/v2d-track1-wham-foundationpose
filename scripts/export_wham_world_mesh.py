#!/usr/bin/env python3
"""Export selected WHAM track fragments as a full-frame world-coordinate SMPL mesh sequence.

Run this inside the WHAM container from its runtime working directory. Missing detection
frames are filled from the nearest observed frame so downstream adapters receive a dense
episode sequence. With multiple ordered subjects, later subjects win overlapping frames.
Without --subjects, the longest track remains the default. This export is SMPL topology and
is not directly submission-compatible.
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
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="ordered WHAM subject IDs; later tracks win overlapping frames")
    args = parser.parse_args()

    results = joblib.load(args.input)
    if not results:
        raise SystemExit("ERROR: WHAM output contains no person tracks")
    subjects = args.subjects or [max(results, key=lambda key: len(results[key]["frame_ids"]))]
    resolved_subjects = []
    chunks = []
    per_subject = []
    for priority, requested in enumerate(subjects):
        subject = requested if requested in results else str(requested)
        if subject not in results:
            raise SystemExit(f"ERROR: requested WHAM subject {requested} is absent; available={list(results)}")
        resolved_subjects.append(subject)
        track = results[subject]
        frames = np.asarray(track["frame_ids"], dtype=np.int64)
        order = np.argsort(frames)
        count = len(frames)
        chunk = {
            "frames": frames[order],
            "pose": np.asarray(track["pose_world"], dtype=np.float32)[order],
            "trans": np.asarray(track["trans_world"], dtype=np.float32)[order],
            "betas": np.asarray(track["betas"], dtype=np.float32)[order],
            "subjects": np.full(count, int(requested), dtype=np.int64),
            "priority": np.full(count, priority, dtype=np.int64),
        }
        translation_offset = np.zeros(3, dtype=np.float32)
        if chunks:
            previous = chunks[-1]
            common, previous_i, current_i = np.intersect1d(
                previous["frames"], chunk["frames"], assume_unique=False, return_indices=True)
            if len(common):
                translation_offset = np.median(
                    previous["trans"][previous_i] - chunk["trans"][current_i], axis=0).astype(np.float32)
            else:
                tail = previous["trans"][max(0, len(previous["trans"]) - 6):]
                velocity = np.median(np.diff(tail, axis=0), axis=0) if len(tail) > 1 else np.zeros(3)
                gap = int(chunk["frames"][0] - previous["frames"][-1])
                target = previous["trans"][-1] + velocity * max(gap, 1)
                translation_offset = (target - chunk["trans"][0]).astype(np.float32)
            chunk["trans"] += translation_offset
        chunks.append(chunk)
        per_subject.append({"subject": str(subject), "observed_frames": int(count),
                            "first_frame": int(frames.min()), "last_frame": int(frames.max()),
                            "translation_offset_world_m": translation_offset.tolist()})

    frame_ids = np.concatenate([chunk["frames"] for chunk in chunks])
    priority = np.concatenate([chunk["priority"] for chunk in chunks])
    pose_sparse = np.concatenate([chunk["pose"] for chunk in chunks])
    trans_sparse = np.concatenate([chunk["trans"] for chunk in chunks])
    betas_sparse = np.concatenate([chunk["betas"] for chunk in chunks])
    source_subjects = np.concatenate([chunk["subjects"] for chunk in chunks])
    order = np.lexsort((priority, frame_ids))
    frame_ids, pose_sparse, trans_sparse = frame_ids[order], pose_sparse[order], trans_sparse[order]
    betas_sparse, source_subjects = betas_sparse[order], source_subjects[order]
    # Higher-priority fragments sort last for duplicate frames; retain that observation.
    keep = np.r_[frame_ids[1:] != frame_ids[:-1], True]
    frame_ids, pose_sparse, trans_sparse = frame_ids[keep], pose_sparse[keep], trans_sparse[keep]
    betas_sparse, source_subjects = betas_sparse[keep], source_subjects[keep]
    if frame_ids[0] < 0 or frame_ids[-1] >= args.frames:
        raise SystemExit(f"ERROR: frame IDs outside [0, {args.frames}): {frame_ids[[0, -1]]}")

    dense_index = nearest_observation(frame_ids, args.frames)
    pose = pose_sparse[dense_index]
    trans = trans_sparse[dense_index]
    if len(resolved_subjects) > 1:
        # Stitched fragments of one person must not introduce a body-shape jump at the ID switch.
        identity_betas = np.median(betas_sparse, axis=0).astype(np.float32)
        betas = np.broadcast_to(identity_betas, (args.frames, len(identity_betas))).copy()
    else:
        betas = betas_sparse[dense_index]   # single track: unchanged legacy behaviour

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
        "subject": str(resolved_subjects[0]),
        "subjects": [str(subject) for subject in resolved_subjects],
        "selection": "configured_fragments" if args.subjects else "longest_track",
        "fragment_stitching": "translation_continuity" if len(resolved_subjects) > 1 else "not_needed",
        "per_subject": per_subject,
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
        source_subject_ids=source_subjects,
        dense_source_frame_ids=frame_ids[dense_index],
        dense_source_subject_ids=source_subjects[dense_index],
        report=json.dumps(report, sort_keys=True),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
