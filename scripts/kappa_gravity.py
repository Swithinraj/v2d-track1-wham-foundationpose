#!/usr/bin/env python3
"""Gravity: the person stands on the floor; scale the objects so that the scene agrees (mhr_converter.sif, GPU or CPU).

The human is SAM 3D Body in its own metric; the objects are placed from MoGe depth with the body-pixel ratio s. In
standing frames the person's soles float above MoGe's floor plane (median ~8 cm, up to 14 cm): person and scene
disagree in scale by ~5 %. CD-O scores the object relative to the person (Sim(3) fitted on the human), so the
consistent fix is to scale every object about the camera by kappa (translation and scale; the projection is
unchanged; the person is untouched) such that the floor, scaled with them, passes under the person's soles:
  sole height h_t = min over body vertices of n.X - c/s   (floor n.X = c in MoGe metres, --floor; s = depth scale)
  h = median of h_t over the standing half of the frames (lowest soles)      kappa = 1 + h / (c / s)
Run after the depth trend and BEFORE the human refinement (the refinement anchors grasping hands to the objects).
Writes <out>/episode_*.npz (+ meshes, shared files) and kappa_gravity_report.json.
Usage: kappa_gravity.py --input DIR --out DIR --floor FILE [--alignment DIR] [--episodes ...]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

T1 = Path("/track1")
sys.path.insert(0, str(T1 / "official/v2d_submission_kit/tools/track1"))
from mesh_to_mhr_params import MHR  # noqa: E402

MODEL = T1 / "weights/mhr/assets/mhr_model.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True, help="JSON {episode: {normal, c}}: floor n.X = c, MoGe metres")
    ap.add_argument("--alignment", type=Path, default=T1 / "outputs/alignment_sam")
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args()
    planes = json.loads(args.floor.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for extra in args.input.iterdir():
        if extra.is_file() and not extra.name.startswith("episode_"):
            shutil.copy2(extra, args.out / extra.name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mhr = MHR(str(MODEL), dev, chunk=64, precision="float32")
    report = {}
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        d = dict(np.load(args.input / f"{tag}.npz"))
        s = json.loads((args.alignment / tag / "alignment.json").read_text())["depth_scale"]
        n, cs = np.asarray(planes[tag]["normal"], float), planes[tag]["c"] / s
        ident = torch.as_tensor(np.concatenate([d["scales"], d["shape"]])[None], device=dev)
        with torch.no_grad():
            v, j = mhr.run(torch.as_tensor(d["pose"], device=dev), ident)
        sole = (v.double().cpu().numpy() / 1000.0 @ n).min(1) - cs
        foot = (j.double().cpu().numpy()[:, [8, 24]] @ n).min(1) - cs - 0.03
        h = float(np.median(sole[sole <= np.median(sole)]))
        h_joint = float(np.median(foot[foot <= np.median(foot)]))
        kappa = 1.0 + h / cs
        d["object_translation"] = (d["object_translation"].astype(np.float64) * kappa).astype(np.float32)
        d["object_scale"] = np.float32(float(d["object_scale"]) * kappa)
        np.savez(args.out / f"{tag}.npz", **d)
        for extra in args.input.glob(f"{tag}_object*"):
            shutil.copy2(extra, args.out / extra.name)
        report[tag] = {"sole_height_cm": h * 100, "kappa": kappa, "kappa_from_foot_joints": 1.0 + h_joint / cs,
                       "camera_height_m": -planes[tag]["c"], "depth_scale": s}
        print(tag, json.dumps({k: round(x, 4) for k, x in report[tag].items()}), flush=True)
    k = [r["kappa"] for r in report.values()]
    (args.out / "kappa_gravity_report.json").write_text(json.dumps(
        {"kappa_median": float(np.median(k)), "kappa_range": [min(k), max(k)], "episodes": report}, indent=1))
    print(f"PASS: kappa median {np.median(k):.3f} range {min(k):.3f}-{max(k):.3f}")


if __name__ == "__main__":
    main()
