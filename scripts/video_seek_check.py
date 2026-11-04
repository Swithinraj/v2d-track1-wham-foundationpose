#!/usr/bin/env python3
"""Wiring check (10-29): FoundationPose's run_video_to_poses reads every RGB frame with cap.set(CAP_PROP_POS_FRAMES, idx)
(forward from the reference frame, then BACKWARD to 0), while MoGe-2 / SAM 2 / SAM 3D Body read the video sequentially
and name their outputs by that count. If seeking is inexact, FoundationPose pairs an RGB frame with another frame's
depth and mask. For each video: decode sequentially, then fetch frames the way FoundationPose does (forward + backward
from a mid reference) and compare pixels. Usage: video_seek_check.py VIDEO [VIDEO ...]
"""
import sys

import cv2
import numpy as np

for path in sys.argv[1:]:
    cap = cv2.VideoCapture(path); seq = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        seq.append(cv2.resize(fr, (160, 90)).astype(np.int16))
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    cap = cv2.VideoCapture(path); ref = len(seq) // 2; bad, off = 0, []
    order = list(range(ref, len(seq))) + list(range(ref - 1, -1, -1))
    for idx in order[::3]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
        if not ok:
            bad += 1; continue
        g = cv2.resize(fr, (160, 90)).astype(np.int16)
        d = np.abs(g - seq[idx]).mean()
        if d > 0.5:
            lo, hi = max(0, idx - 5), min(len(seq), idx + 6)
            best = min(range(lo, hi), key=lambda j: np.abs(g - seq[j]).mean()); off.append(best - idx); bad += 1
    print(f"{path.split('/')[-1]}: frames seq {len(seq)} meta {n_meta}; seek mismatches {bad}/{len(order[::3])}"
          f"{'' if not off else f', offsets {sorted(set(off))}'}", flush=True)
