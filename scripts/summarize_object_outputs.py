#!/usr/bin/env python3
"""Validate per-frame MoGe/SAM2 products and write a Track 1 object status report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/work/smoses2s/nvidea_chal/TRACK1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_root = args.root / "outputs" / "object"
    report_path = args.output or output_root / "BASELINE_STATUS.json"

    lengths = {}
    with (args.root / "data" / "track_1" / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            lengths[int(item["episode_index"])] = int(item["length"])

    episodes = []
    incomplete = []
    review = []
    for index in range(30):
        expected = lengths[index]
        base = output_root / f"episode_{index:06d}"
        depth = len(list((base / "depth").glob("*.png")))
        intrinsics = len(list((base / "intrinsics").glob("*.json")))
        masks = len(list((base / "sam2_masks" / "1").glob("*.png")))
        prompt_path = base / "prompt" / "sam2_prompts.json"
        audit_path = base / "mask_audit.json"
        prompt = json.loads(prompt_path.read_text()) if prompt_path.is_file() else {}
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
        complete = depth == expected and intrinsics == expected and masks == expected
        if not complete:
            incomplete.append(index)
        requires_review = bool(audit.get("requires_visual_review", True))
        if requires_review:
            review.append(index)
        episodes.append({
            "episode_index": index,
            "expected_frames": expected,
            "depth_frames": depth,
            "intrinsics_frames": intrinsics,
            "mask_frames": masks,
            "prompt_source": prompt.get("provenance", {}).get("source", "legacy_grounding_dino"),
            "mask_requires_temporal_review": requires_review,
            "empty_mask_frames": audit.get("empty_frames"),
            "complete": complete,
        })

    report = {
        "schema": "v2d.track1.object_baseline_status.v1",
        "episodes": episodes,
        "complete_episode_count": 30 - len(incomplete),
        "incomplete_episodes": incomplete,
        "temporal_review_episodes": review,
        "foundationpose_blocker": (
            "Public Track 1 release contains no object CAD meshes; place the ten metric meshes "
            "listed in docs/OBJECT_BASELINE.md before submitting FoundationPose."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
