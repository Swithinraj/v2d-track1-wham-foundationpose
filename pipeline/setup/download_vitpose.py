"""Download the one ViTPose-H checkpoint used by the best Track 1 recipe.

The model comes from a pinned Hugging Face model mirror and is checksum-verified.
No training or validation dataset is downloaded.
"""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "weights" / "vitpose" / "vitpose-h-multi-coco.pth"
EXPECTED_SHA256 = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"


def valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == EXPECTED_SHA256


if not valid(DEST):
    downloaded = Path(
        hf_hub_download(
            repo_id="camenduru/GVHMR",
            filename="vitpose/vitpose-h-multi-coco.pth",
            revision="21b32d5389e2e59c0737d4c4095bbc0b8c23f66b",
            local_dir=ROOT / "weights",
        )
    )
    if downloaded != DEST:
        raise SystemExit(f"Unexpected download location: {downloaded}")
    if not valid(DEST):
        raise SystemExit("ViTPose checkpoint checksum mismatch")

print("PASS: ViTPose-H checkpoint is present and SHA256-verified")
