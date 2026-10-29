#!/usr/bin/env python3
"""Camera consistency check (10-29, owner: "is their container configured properly for our use case?").
FoundationPose tracks each object with the MoGe intrinsics of ONE frame (outputs/object/<tag>/intrinsics/<ref>.json), the
human (SAM 3D Body, refinement, scorer export) uses outputs/camera/<tag>/intrinsics.json, and the build
(scripts/build_track1_episodes_sam.py: object_translation = t_co / s) does not convert between them. A pixel seen by
FoundationPose at (u - c) = f_obj x / z lands, in the human's camera, at x' = x f_obj / f_cam: the fix scales the object's
lateral coordinates by f_obj / f_cam (depth unchanged; principal points agree). Writes ROOT/submissions/<IN>_ffix with the
object translation fixed (human unchanged), for scoring with fh_score.py. Usage: object_camera_replay.py ROOT [IN]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    root = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "episodes_cand_BMGM"
    src, dst = root / "submissions" / name, root / "submissions" / f"{name}_ffix"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst); rs = {}
    for f in sorted(dst.glob("episode_??????.npz")):
        tag = f.stem; od = root / "outputs/object" / tag
        ref = int((od / "prompt/reference_frame.txt").read_text().strip())
        fo = json.loads((od / "intrinsics" / f"{ref:06d}.json").read_text())["fx"]
        fc = json.loads((root / "outputs/camera" / tag / "intrinsics.json").read_text())["fx"]
        r = float(fo) / float(fc); z = dict(np.load(f)); t = z["object_translation"].astype(np.float64)
        t[:, :2] *= r; z["object_translation"] = t.astype(np.float32); np.savez(f, **z); rs[tag] = r
    (dst / "focal_fix.json").write_text(json.dumps(rs, indent=1))
    v = np.array(list(rs.values()))
    print(f"PASS: {len(v)} episodes, f_obj / f_cam median {np.median(v):.4f} (range {v.min():.4f}-{v.max():.4f}) -> {dst.name}")


if __name__ == "__main__":
    main()
