"""Fetch only the pinned V2D Track 1 competition release.

Run with the submission virtual environment on Linux or Windows. The optional
token stays outside the repository and is never copied or printed.
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
pins = dict(
    line.split("=", 1)
    for line in (ROOT / "config" / "upstream.env").read_text(encoding="utf-8").splitlines()
    if line.startswith(("V2D_DATASET_ID=", "V2D_DATASET_REVISION="))
)
TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", ROOT.parent / "huggingface.txt"))
token = os.environ.get("HF_TOKEN")
if not token and TOKEN_FILE.exists():
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()

snapshot_download(
    repo_id=pins["V2D_DATASET_ID"],
    repo_type="dataset",
    revision=pins["V2D_DATASET_REVISION"],
    allow_patterns=["track_1/**"],
    local_dir=ROOT / "data",
    cache_dir=ROOT / ".cache" / "huggingface",
    token=token,
)
