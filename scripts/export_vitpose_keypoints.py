#!/usr/bin/env python3
"""WHAM's ViTPose keypoints of the exported actor track(s), per video frame (wham.sif, CPU).

The exported actor fragments (subjects) and per-frame sources of wham_smpl_world.npz pick the keypoints.
Used as the held-out 2D check of the human refinement. Writes outputs/vitpose/episode_XXXXXX.npz:
kps [T, 17, 3] (x_px, y_px, confidence; NaN where missing).
Usage: export_vitpose_keypoints.py EPISODE [...]
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

T1 = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
out = T1 / "outputs/vitpose"
out.mkdir(parents=True, exist_ok=True)
for ep in [int(a) for a in sys.argv[1:]] or range(30):
    tag = f"episode_{ep:06d}"
    z = np.load(T1 / "outputs/wham" / tag / "wham_smpl_world.npz")
    rep = json.loads(str(z["report"]))
    subjects = rep.get("subjects", [rep["subject"]])
    observed = z["source_frame_ids"].astype(int)
    src = z["source_subject_ids"].astype(int) if "source_subject_ids" in z.files else np.full(len(observed), int(subjects[0]))
    tracking = joblib.load(T1 / "outputs/wham" / tag / "tracking_results.pth")
    lookup = {}
    for sel in subjects:
        tr = tracking[int(sel)] if int(sel) in tracking else tracking[sel]
        for fr, kp in zip(np.asarray(tr["frame_id"]), np.asarray(tr["keypoints"])):
            lookup[(int(sel), int(fr))] = kp
    T = len(z["vertices"])
    kps = np.full((T, 17, 3), np.nan, np.float32)
    for fr, s in zip(observed, src):                      # export rows are indexed by video frame (round 2: V[frames])
        if (int(s), int(fr)) in lookup and int(fr) < T:
            kps[int(fr)] = lookup[(int(s), int(fr))]
    np.savez_compressed(out / f"{tag}.npz", kps=kps)
    print(tag, f"frames with keypoints {np.isfinite(kps[:, 0, 0]).mean():.0%}", flush=True)
