#!/usr/bin/env python3
"""Which rig camera recorded each challenge episode? (10-29 formula audit; the four FORM-HOI rig cameras each have one
calibration over all recording days, scripts/camera_truth_check.py). Per video: a small grey background = per-pixel temporal
median of 40 uniformly spaced frames (people and objects move, the room does not), 96 x 72, zero-mean / unit-norm.
FORM-HOI episodes give labelled backgrounds (episodes_metadata.jsonl camera); each challenge episode takes the camera of
its best-correlated FORM-HOI backgrounds (top-5 vote) and reports the correlation, also of the best OTHER camera, so a
weak or ambiguous match is visible. Writes OUT/camera_id.json {episode: {camera, corr, corr_other, votes}}.
Usage: camera_identify.py FORMHOI_DIR CHALLENGE_ROOT OUT [MAX_PER_CAMERA]
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def background(path, n=40):
    cap = cv2.VideoCapture(str(path)); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for i in np.linspace(0, max(T - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok:
            fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 72), interpolation=cv2.INTER_AREA))
    cap.release()
    b = np.median(np.stack(fr), 0).astype(np.float64); b -= b.mean()
    return b.ravel() / max(np.linalg.norm(b), 1e-9)


def main():
    F, C, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); cap_n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    lab, seen = [], Counter()
    for root in sorted(F.glob("root*")):
        meta = root / "data/track_1/meta/episodes_metadata.jsonl"
        if root.is_symlink() or not meta.exists():
            continue
        for m in map(json.loads, filter(str.strip, meta.read_text().splitlines())):
            v = root / "data/track_1/videos/chunk-000/observation.images.exo_camera" / f"episode_{m['episode_index']:06d}.mp4"
            if v.exists() and seen[m["camera"]] < cap_n:
                lab.append((m["camera"], v)); seen[m["camera"]] += 1
    ch = sorted((C / "data/track_1/videos/chunk-000/observation.images.exo_camera").glob("episode_*.mp4"))
    with ProcessPoolExecutor(16) as ex:
        LB = np.stack(list(ex.map(background, [v for _, v in lab]))); CB = np.stack(list(ex.map(background, ch)))
    cams = np.array([c for c, _ in lab]); S = CB @ LB.T; rep = {}
    for i, v in enumerate(ch):
        top = np.argsort(-S[i])[:5]; vote = Counter(cams[top]).most_common(); cam = vote[0][0]
        other = S[i][cams != cam].max() if (cams != cam).any() else float("nan")
        rep[v.stem] = {"camera": cam, "corr": float(S[i][cams == cam].max()), "corr_other": float(other), "votes": dict(vote)}
        print(f"{v.stem}: {cam:26s} corr {rep[v.stem]['corr']:.3f} (best other {other:.3f}) votes {dict(vote)}", flush=True)
    within = [float(np.sort(S_)[-2]) for S_ in (LB @ LB.T)]
    print(f"labelled backgrounds: {dict(seen)}; FORM-HOI self-match (2nd best) median {np.median(within):.3f}")
    out.mkdir(parents=True, exist_ok=True); (out / "camera_id.json").write_text(json.dumps(rep, indent=1))
    print("PASS:", Counter(r["camera"] for r in rep.values()))


if __name__ == "__main__":
    main()
