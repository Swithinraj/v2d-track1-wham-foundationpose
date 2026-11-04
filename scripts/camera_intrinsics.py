#!/usr/bin/env python3
"""One pinhole camera per episode: the median of MoGe-2's per-frame intrinsics (host, stdlib only).

The camera is static, so a single intrinsics estimate serves the whole episode (SAM 3D Body prompts, projections,
the human refinement). Reads outputs/object/<episode>/intrinsics/*.json (object_depth.sh), writes
outputs/camera/<episode>/intrinsics.json with fx, fy, cx, cy, width, height.
Usage: camera_intrinsics.py EPISODE
"""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def main() -> None:
    tag = f"episode_{int(sys.argv[1]):06d}"
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/object" / tag / "intrinsics").glob("*.json"))]
    if not rows:
        raise SystemExit(f"{tag}: no MoGe intrinsics; run pipeline/stages/object_depth.sh")
    k = {key: float(statistics.median(r[key] for r in rows)) for key in ("fx", "fy", "cx", "cy", "width", "height")}
    out = ROOT / "outputs/camera" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "intrinsics.json").write_text(json.dumps(k, indent=2))
    print(tag, json.dumps(k))


if __name__ == "__main__":
    main()
