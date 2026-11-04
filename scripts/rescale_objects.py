#!/usr/bin/env python3
"""Re-express an episode directory's objects with the depth scale of the human they are combined with (host, numpy).

The episode builder (build_track1_episodes_sam.py) stores the objects as object_translation = t_co / s and
object_scale = s_co / s, where s is the episode's MoGe/SAM depth scale (outputs/sam3d_body/<ep>/prepare.npz) of the
SAM 3D Body run the human came from. When the human is re-inferred and s re-derived (s_new), objects built with the
old scale (s_old) must follow: translation and scale x s_old / s_new, a scaling about the camera that leaves every
object's image projection unchanged. The human (pose, scales, shape) and the object rotations are copied unchanged.
Usage: rescale_objects.py --input DIR --out DIR --s_old SAM3D_BODY_DIR --s_new SAM3D_BODY_DIR
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def depth_scale(sam_dir, tag):
    return float(np.load(Path(sam_dir) / tag / "prepare.npz")["depth_scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s_old", type=Path, required=True, help="outputs/sam3d_body of the run the objects were built with")
    ap.add_argument("--s_new", type=Path, required=True, help="outputs/sam3d_body of the run the human comes from")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for src in sorted(args.input.iterdir()):
        if not src.is_file():
            continue
        if not (src.name.startswith("episode_") and src.suffix == ".npz"):
            shutil.copy2(src, args.out / src.name)
            continue
        tag = src.stem
        d = dict(np.load(src))
        f = depth_scale(args.s_old, tag) / depth_scale(args.s_new, tag)
        d["object_translation"] = (d["object_translation"].astype(np.float64) * f).astype(d["object_translation"].dtype)
        d["object_scale"] = np.asarray(float(d["object_scale"]) * f, dtype=np.asarray(d["object_scale"]).dtype)
        np.savez(args.out / src.name, **d)
        report[tag] = {"factor": f}
        print(tag, f"objects x {f:.5f}", flush=True)
    fs = np.array([r["factor"] for r in report.values()])
    print(f"factor median {np.median(fs):.5f} range {fs.min():.5f}..{fs.max():.5f}")
    (args.out / "rescale_objects_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
