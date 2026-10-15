#!/usr/bin/env python3
"""Validate the pinned WHAM checkout, container, public assets, and licensed SMPL input."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("/work/smoses2s/nvidea_chal/TRACK1")
WHAM = ROOT / "third_party" / "WHAM"
ASSETS = ROOT / "weights" / "wham"
EXPECTED_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise SystemExit(f"ERROR: missing or truncated file: {path}")


def main() -> None:
    require_file(ROOT / "containers" / "wham.sif", 1_000_000)
    require_file(WHAM / "demo.py")
    commit = subprocess.run(
        ["git", "-C", str(WHAM), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"ERROR: WHAM commit mismatch: expected {EXPECTED_COMMIT}, found {commit}")

    for name in (
        "wham_vit_bedlam_w_3dpw.pth.tar",
        "hmr2a.ckpt",
        "dpvo.pth",
        "yolov8x.pt",
        "vitpose-h-multi-coco.pth",
    ):
        require_file(ASSETS / "checkpoints" / name, 1_000_000)

    body = ASSETS / "dataset" / "body_models"
    for name in (
        "smpl_mean_params.npz",
        "J_regressor_wham.npy",
        "J_regressor_h36m.npy",
        "J_regressor_feet.npy",
    ):
        require_file(body / name, 1_000)

    # This file cannot legally be redistributed with WHAM. The user must obtain it after
    # accepting the SMPL/SMPLify terms and place it at this exact path.
    require_file(body / "smpl" / "SMPL_NEUTRAL.pkl", 1_000_000)
    # Derived locally from the licensed model by submit_human_pipeline.sh. It is not part of
    # WHAM's public body_models.tar.gz archive.
    require_file(body / "smpl_faces.npy", 1_000)

    print(f"PASS: WHAM runtime, commit {commit[:12]}, checkpoints, and SMPL are ready")


if __name__ == "__main__":
    main()
