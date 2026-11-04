#!/usr/bin/env python3
"""MediaPipe-free 2D body landmarks in MediaPipe's layout, from the tracked ViTPose-H keypoints (submission venv, CPU).

The human refinement (wrist_fusion.py, root_lateral.py; HUMAN_KP2D_DIR) and the contact solve (human_contact_solver.py
--kp2d_dir) read MediaPipe's layout: pose [T,3,33,5] (x, y, z, visibility, presence for up to 3 people), hands
[T,3,2,21,3], hand_score [T,3,2]. Here slot 0 = the actor's ViTPose-H keypoints (vitpose_track.py: top-down on the
actor's SAM 2 track, so the actor by construction), mapped COCO-17 -> MediaPipe-33; a joint is present where its
confidence > 0.5 (visibility = presence = confidence). Slots 1-2 and the hands are empty (hand_score 0), so no hand
landmark term is used downstream.
COCO and MediaPipe hips sit ~10 px apart and the refinement pairs the MHR hip joints with MediaPipe-defined hips, so
each ViTPose hip is shifted by one constant per episode: the median of (projected MHR hip joint - ViTPose hip) over its
confident frames, measured on a human refined without any 2D landmarks (--bodies: mhr_bodies.py of the
HUMAN_KP2D_DIR=none refinement). The frame-to-frame signal stays ViTPose's own.
Writes <out>/<episode>.npz.
Usage: kp2d_vitpose.py --vitpose DIR --bodies DIR --out DIR [--alignment DIR] [--episodes ...]
"""
import argparse
import os
from pathlib import Path
import json

import numpy as np

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
COCO2MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
HIPS = {23: 2, 24: 18}                         # MediaPipe hip landmark -> MHR joint (human_refine_common.PAIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitpose", type=Path, default=ROOT / "outputs/vitpose", help="vitpose_track.py output")
    ap.add_argument("--bodies", type=Path, required=True, help="mhr_bodies.py output (joints) of a human refined without 2D landmarks")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs/alignment_sam")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(30)))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        vp = np.load(args.vitpose / f"{tag}.npz")["kps2d"].astype(np.float64); Tv = len(vp)
        J = np.load(args.bodies / f"{tag}.npz")["joints"].astype(np.float64); T = len(J)
        a = json.loads((args.alignment / tag / "alignment.json").read_text()); K, ds = a["intrinsics"], float(a["depth_scale"])
        c = ds * J; uv = np.stack([K["fx"] * c[..., 0] / c[..., 2] + K["cx"], K["fy"] * c[..., 1] / c[..., 2] + K["cy"]], -1)
        pose = np.full((T, 3, 33, 5), np.nan, np.float32); pose[..., 3:] = 0.0
        hands = np.full((T, 3, 2, 21, 3), np.nan, np.float32); hscore = np.zeros((T, 3, 2), np.float32)
        n = min(T, Tv); offs = {}
        for cidx, l in COCO2MP.items():
            ok = np.zeros(T, bool); ok[:n] = vp[:n, cidx, 2] > 0.5
            shift = np.zeros(2)
            if l in HIPS and ok.sum() >= 10:
                shift = np.median(uv[ok, HIPS[l]] - vp[np.flatnonzero(ok), cidx, :2], 0)
            offs[l] = np.round(shift, 1).tolist()
            fi = np.flatnonzero(ok)
            pose[fi, 0, l, :2] = vp[fi, cidx, :2] + shift; pose[fi, 0, l, 2] = 0.0
            pose[fi, 0, l, 3] = vp[fi, cidx, 2]; pose[fi, 0, l, 4] = vp[fi, cidx, 2]
        np.savez(args.out / f"{tag}.npz", pose=pose, hands=hands, hand_score=hscore)
        print(tag, "frames", T, "| ViTPose frames", Tv, "| wrists L/R visible", int((pose[:, 0, 15, 3] > 0.5).sum()),
              int((pose[:, 0, 16, 3] > 0.5).sum()), "| hip offsets px", offs[23], offs[24], flush=True)


if __name__ == "__main__":
    main()
