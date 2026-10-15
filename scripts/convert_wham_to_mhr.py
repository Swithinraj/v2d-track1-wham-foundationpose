#!/usr/bin/env python3
"""Convert dense WHAM SMPL meshes to official Track 1 MHR parameters.

Stage 1 applies Meta MHR's released SMPL-to-MHR barycentric surface map. Stage 2
uses the challenge submission kit's reference TorchScript fitter, which enforces
one identity over the episode and writes the exact 136/68/45 parameter schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def map_smpl_to_mhr(
    vertices: np.ndarray,
    faces_path: str,
    mapping_path: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    faces = np.asarray(np.load(faces_path), dtype=np.int64)
    mapping = np.load(mapping_path)
    triangle_ids = np.asarray(mapping["triangle_ids"], dtype=np.int64)
    bary = np.asarray(mapping["baryc_coords"], dtype=np.float32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected triangular SMPL faces [F,3], got {faces.shape}")
    if triangle_ids.shape != (18439,) or bary.shape != (18439, 3):
        raise ValueError(
            f"unexpected Meta SMPL->MHR mapping: triangles={triangle_ids.shape}, bary={bary.shape}"
        )
    if triangle_ids.min() < 0 or triangle_ids.max() >= len(faces):
        raise ValueError("Meta mapping references a triangle outside the SMPL face array")

    dev = torch.device(device)
    source_faces = torch.as_tensor(faces[triangle_ids], dtype=torch.long, device=dev)
    weights = torch.as_tensor(bary, dtype=torch.float32, device=dev)[None, :, :, None]
    output = np.empty((len(vertices), 18439, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(vertices), batch_size):
            stop = min(start + batch_size, len(vertices))
            batch = torch.as_tensor(vertices[start:stop], dtype=torch.float32, device=dev)
            triangles = batch[:, source_faces, :]
            output[start:stop] = (triangles * weights).sum(dim=2).cpu().numpy()
            print(f"mapped {stop}/{len(vertices)} SMPL frames to MHR topology", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="WHAM world mesh NPZ")
    parser.add_argument("--output", required=True, help="Track 1 human MHR NPZ")
    parser.add_argument("--smpl-faces", required=True, help="WHAM smpl_faces.npy")
    parser.add_argument("--mapping", required=True, help="Meta smpl2mhr_mapping.npz")
    parser.add_argument("--mhr-model", required=True, help="Apache-2.0 MHR mhr_model.pt")
    parser.add_argument("--official-converter", required=True, help="mesh_to_mhr_params.py")
    parser.add_argument("--batch-size", type=int, default=64, help="Topology-map GPU batch")
    parser.add_argument("--model-batch", type=int, default=256, help="MHR fitter model batch")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = np.load(args.input)
    vertices = np.asarray(source["vertices"], dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3):
        raise SystemExit(f"ERROR: expected WHAM SMPL vertices [T,6890,3], got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise SystemExit("ERROR: WHAM vertices contain NaN or Inf")

    mhr_topology = map_smpl_to_mhr(
        vertices,
        args.smpl_faces,
        args.mapping,
        args.device,
        args.batch_size,
    )

    converter_path = Path(args.official_converter).resolve()
    sys.path.insert(0, str(converter_path.parent))
    from mesh_to_mhr_params import convert  # noqa: E402

    converted = convert(
        mhr_topology,
        args.mhr_model,
        device=args.device,
        precision=args.precision,
        model_batch=args.model_batch,
        log=lambda *items, **kwargs: print(*items, **kwargs, file=sys.stderr, flush=True),
    )
    fit_report = converted.pop("report")
    pose = np.asarray(converted["pose"], dtype=np.float32)
    scales = np.asarray(converted["scales"], dtype=np.float32)
    shape = np.asarray(converted["shape"], dtype=np.float32)
    if pose.shape != (len(vertices), 136) or scales.shape != (68,) or shape.shape != (45,):
        raise SystemExit(
            f"ERROR: official converter returned pose={pose.shape}, scales={scales.shape}, shape={shape.shape}"
        )
    if not np.isfinite(pose).all() or not np.isfinite(scales).all() or not np.isfinite(shape).all():
        raise SystemExit("ERROR: MHR conversion produced NaN or Inf")

    report = {
        "schema": "v2d.track1.wham_to_mhr.v2",
        "frames": int(len(vertices)),
        "source_topology": "SMPL-6890",
        "target_topology": "MHR-lod1-18439",
        "topology_converter": "facebookresearch/MHR smpl2mhr_mapping.npz",
        "parameter_converter": "official v2d_submission_kit mesh_to_mhr_params.py",
        "coordinate_frame": "WHAM world metres; official fitter handles MHR native y/z and centimetres",
        "fit": fit_report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pose=pose,
        scales=scales,
        shape=shape,
        per_frame_vertex_error_mm=converted["per_frame_vertex_error_mm"],
        valid_input=converted["valid_input"],
        report=json.dumps(report, sort_keys=True),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
