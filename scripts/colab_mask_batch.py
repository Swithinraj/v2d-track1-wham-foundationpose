#!/usr/bin/env python3
"""Run the repo's object and actor mask stages natively on a Colab GPU.

This calls the same GroundingDINO prompt scripts and NVIDIA SAM 2 propagation
function as object_masks.sh/person.sh, but keeps both models loaded across
episodes and resumes at completed episode boundaries. It expects their Python
packages and the two checkpoints to be installed separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))


def call_main(module, *arguments: str) -> None:
    saved = sys.argv
    sys.argv = [module.__file__, *arguments]
    try:
        module.main()
    finally:
        sys.argv = saved


def complete(directory: Path, frames: int) -> bool:
    return (directory.is_dir()
            and all((directory / f"{frame:06d}.png").is_file() for frame in range(frames)))


def run_episode(ep: int, frames: int, object_script, person_script, audit_script, video_to_masks) -> None:
    tag = f"episode_{ep:06d}"
    video = ROOT / "data/track_1/videos/chunk-000/observation.images.exo_camera" / f"{tag}.mp4"
    obj_root = ROOT / "outputs/object" / tag
    person_root = ROOT / "outputs/person" / tag
    obj_masks = obj_root / "sam2_masks/1"
    actor_masks = person_root / "sam2_masks/0"
    sam2_weights = ROOT / "weights/sam2"
    dino_weights = ROOT / "weights/object/grounding_dino"

    if not complete(obj_masks, frames):
        prompt = obj_root / "prompt/sam2_prompts.json"
        if not prompt.is_file():
            call_main(
                object_script,
                "--episode", str(ep), "--video", str(video),
                "--model-dir", str(dino_weights),
                "--output-dir", str(prompt.parent),
            )
        video_to_masks(str(video), str(prompt), str(obj_root / "sam2_masks"), str(sam2_weights))
        if not complete(obj_masks, frames):
            raise RuntimeError(f"{tag}: object masks incomplete")
    audit_path = obj_root / "mask_audit.json"
    if not audit_path.is_file():
        call_main(
            audit_script, "--masks", str(obj_masks), "--frames", str(frames),
            "--output", str(audit_path),
        )

    if not complete(actor_masks, frames):
        prompt = person_root / "prompt/sam2_prompts.json"
        if not prompt.is_file():
            call_main(
                person_script,
                "--episode", str(ep), "--video", str(video),
                "--model-dir", str(dino_weights),
                "--output-dir", str(prompt.parent),
            )
        video_to_masks(str(video), str(prompt), str(person_root / "sam2_masks"), str(sam2_weights))
        if not complete(actor_masks, frames):
            raise RuntimeError(f"{tag}: actor masks incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", type=int, nargs="*", default=list(range(30)))
    args = parser.parse_args()
    import torch
    from scripts import audit_object_masks, detect_object_prompts, detect_person_prompts
    from v2d.sam2.lib.video_to_masks import video_to_masks

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SAM 2 video propagation")
    metadata = ROOT / "data/track_1/meta/episodes.jsonl"
    expected = {int(row["episode_index"]): int(row["length"])
                for line in metadata.read_text().splitlines() if (row := json.loads(line))}
    for ep in args.episodes:
        if ep not in expected:
            raise ValueError(f"Episode {ep} is absent from Track 1 metadata")
        started = time.monotonic()
        run_episode(
            ep, expected[ep], detect_object_prompts, detect_person_prompts,
            audit_object_masks, video_to_masks,
        )
        print(f"PASS episode_{ep:06d}: object and actor masks, {time.monotonic() - started:.1f}s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
