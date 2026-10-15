#!/usr/bin/env python3
"""Validate the pinned Meta map, converter image, and reference MHR model."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path("/work/smoses2s/nvidea_chal/TRACK1")
MHR = ROOT / "third_party" / "MHR"
EXPECTED_COMMIT = "d96fafa33bbf018647c70c3525e91f53e79d2a14"
MODEL_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise SystemExit(f"ERROR: missing or truncated file: {path}")


def main() -> None:
    require_file(ROOT / "containers" / "mhr_converter.sif", 1_000_000)
    commit = subprocess.run(
        ["git", "-C", str(MHR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"ERROR: MHR commit mismatch: expected {EXPECTED_COMMIT}, found {commit}")

    mapping = MHR / "tools" / "mhr_smpl_conversion" / "assets" / "smpl2mhr_mapping.npz"
    require_file(mapping, 1_000)
    assets = ROOT / "weights" / "mhr" / "assets"
    model = assets / "mhr_model.pt"
    require_file(model, 1_000_000)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"ERROR: MHR TorchScript checksum mismatch: {digest}")

    require_file(ROOT / "official" / "v2d_submission_kit" / "tools" / "track1" / "mesh_to_mhr_params.py")
    print(f"PASS: Meta MHR map at commit {commit[:12]}, reference model, and official fitter are ready")


if __name__ == "__main__":
    main()
