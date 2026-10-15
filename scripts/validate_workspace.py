#!/usr/bin/env python3
"""Fast, dependency-free validation of the Track 1 workspace and public dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/work/smoses2s/nvidea_chal/TRACK1")
DATA = ROOT / "data" / "track_1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def validate_data() -> None:
    meta = DATA / "meta"
    videos = DATA / "videos" / "chunk-000" / "observation.images.exo_camera"
    parquet = DATA / "data" / "chunk-000"
    require((DATA / "README.md").is_file(), f"missing {DATA / 'README.md'}")
    for name in ("info.json", "episodes.jsonl", "episodes_metadata.jsonl", "tasks.jsonl"):
        require((meta / name).is_file(), f"missing metadata file {meta / name}")

    video_files = sorted(videos.glob("episode_*.mp4"))
    parquet_files = sorted(parquet.glob("episode_*.parquet"))
    require(len(video_files) == 30, f"expected 30 videos, found {len(video_files)}")
    require(len(parquet_files) == 30, f"expected 30 parquet indexes, found {len(parquet_files)}")
    require(all(path.stat().st_size > 1_000_000 for path in video_files), "one or more videos are truncated")
    require(all(path.stat().st_size > 1_000 for path in parquet_files), "one or more parquet files are truncated")

    with (meta / "info.json").open() as handle:
        info = json.load(handle)
    with (meta / "episodes.jsonl").open() as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    require(len(episodes) == 30, f"expected 30 episode records, found {len(episodes)}")
    total_frames = sum(int(item["length"]) for item in episodes)
    require(total_frames == 16_563, f"expected 16,563 frames, metadata reports {total_frames:,}")

    first = video_files[0]
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-of", "json", str(first),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("WARN: ffprobe unavailable; skipped video codec check")
    else:
        stream = json.loads(probe.stdout)["streams"][0]
        require(stream["width"] == 1536 and stream["height"] == 1152, f"unexpected video resolution: {stream}")
        require(stream["codec_name"] == "h264" and stream["r_frame_rate"] == "30/1", f"unexpected video encoding: {stream}")

    print(f"PASS data: 30 videos, 30 parquet files, {total_frames:,} frames, schema {info.get('codebase_version', 'unknown')}")


def validate_workspace() -> None:
    require(ROOT.is_dir(), f"missing workspace {ROOT}")
    repo = ROOT / "third_party" / "video_to_data"
    kit = ROOT / "official" / "v2d_submission_kit"
    require((repo / ".git").is_dir(), "official toolkit checkout missing")
    require((repo / "reconstruction" / "modules" / "v2d_cari4d" / "README.md").is_file(), "CARI4D module missing")
    require((kit / "eval_reconstruction.py").is_file(), "extracted submission kit missing")
    require((kit / "data" / "track_1_sample_submission.parquet").is_file(), "Track 1 sample submission missing")
    print("PASS workspace: toolkit, CARI4D module, and Track 1 submission kit are present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if not args.data_only:
        validate_workspace()
    validate_data()


if __name__ == "__main__":
    main()

