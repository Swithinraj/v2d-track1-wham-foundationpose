#!/usr/bin/env python3
"""Validate persistent object-branch assets and list every missing metric mesh."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise SystemExit(f"ERROR: missing or truncated file: {path}")


def main() -> None:
    for image in ("v2d_cari4d.sif", "v2d_sam2.sif", "v2d_grounding_dino.sif"):
        require_file(ROOT / "containers" / image, 1_000_000)

    require_file(ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt", 1_000_000)
    require_file(ROOT / "weights" / "object" / "moge2" / "model.pt", 1_000_000)
    require_file(
        ROOT / "weights" / "object" / "grounding_dino" / "groundingdino_swint_ogc.pth",
        1_000_000,
    )
    fp = ROOT / "weights" / "object" / "foundationpose" / "nvlabs_pytorch"
    for run in ("2024-01-11-20-02-45", "2023-10-28-18-33-37"):
        require_file(fp / run / "config.yml", 100)
        require_file(fp / run / "model_best.pth", 1_000_000)

    metadata = ROOT / "data" / "track_1" / "meta" / "episodes_metadata.jsonl"
    objects = sorted(
        {
            json.loads(line)["object"]
            for line in metadata.read_text().splitlines()
            if line.strip()
        }
    )
    mesh_root = ROOT / "inputs" / "object_meshes"
    missing = [mesh_root / f"{name}.glb" for name in objects if not (mesh_root / f"{name}.glb").is_file()]
    if missing:
        print("PASS: object containers and public checkpoints are ready")
        print("BLOCKED: supply these metric meshes before FoundationPose:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(2)

    for name in objects:
        require_file(mesh_root / f"{name}.glb", 1_000)
    print(f"PASS: object runtime, public checkpoints, and all {len(objects)} metric meshes are ready")


if __name__ == "__main__":
    main()
