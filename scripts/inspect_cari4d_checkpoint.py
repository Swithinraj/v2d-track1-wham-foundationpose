#!/usr/bin/env python3
"""Write a shape-only inventory of the accepted CARI4D checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


INTERESTING = re.compile(
    r"mhr|body[_\.]?model|head[_\.]?pose|decoder|shape[_\.]?comp|scale[_\.]?comp|joint[_\.]?rotation",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    top_level = list(payload) if isinstance(payload, dict) else [type(payload).__name__]
    matches: list[dict] = []
    tensor_count = 0
    tensor_elements = 0

    def visit(value: object, prefix: str) -> None:
        nonlocal tensor_count, tensor_elements
        if torch.is_tensor(value):
            tensor_count += 1
            tensor_elements += int(value.numel())
            if INTERESTING.search(prefix) and len(matches) < 5000:
                matches.append({
                    "key": prefix,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                })
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child, child_prefix)
        elif isinstance(value, (list, tuple)) and len(value) <= 1000:
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(payload, "")
    report = {
        "schema": "v2d.track1.cari4d_checkpoint_inventory.v1",
        "checkpoint": str(checkpoint),
        "top_level_keys": top_level,
        "tensor_count": tensor_count,
        "tensor_elements": tensor_elements,
        "interesting_tensor_count": len(matches),
        "interesting_tensors": matches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in report.items() if key != "interesting_tensors"}, indent=2))


if __name__ == "__main__":
    main()
