#!/usr/bin/env python3
"""SAM 2 prompts for the ACTOR of an episode from GroundingDINO person detections (v2d_grounding_dino.sif).

The actor is the person who handles the object. On every STRIDE-th frame GroundingDINO detects "a person."; the
detections are linked into tracks (greedy IoU > 0.3 with the track's last box, gaps of up to 3 samples bridged).
Each track is scored by its median distance to the object's SAM 2 mask (outputs/object/<episode>/sam2_masks/1;
distance of the mask's box centre to the person box, 0 when inside, in person-box heights) over the samples where
both exist; the actor is the track with the lowest score among tracks covering >= 20 % of the samples (ties: the
longer track). Prompts: the actor's box every PROMPT_EVERY frames where it was detected with confidence >= 0.35,
as SAM 2 box prompts for object id 0. Reviewed overrides in config/manual_person_prompts.json replace the
automatic prompts (same schema as config/manual_object_prompts.json).
Writes outputs/person/<episode>/prompt/{sam2_prompts.json, person_tracks.json}.
Usage: detect_person_prompts.py --episode E --video PATH --model-dir DIR --output-dir DIR [--stride 10]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from v2d.grounding_dino.lib.video_to_object_bboxes import _detect_frame, _get_model

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
PROMPT_EVERY = 30


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def object_box(episode: int, frame: int):
    path = ROOT / "outputs/object" / f"episode_{episode:06d}" / "sam2_masks/1" / f"{frame:06d}.png"
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None or not (m > 127).any():
        return None
    ys, xs = np.nonzero(m > 127)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def contact_distance(person, obj):
    cx, cy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
    dx = max(person[0] - cx, 0.0, cx - person[2])
    dy = max(person[1] - cy, 0.0, cy - person[3])
    return float(np.hypot(dx, dy) / max(person[3] - person[1], 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--manual-overrides", type=Path, default=ROOT / "config/manual_person_prompts.json")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manual = json.loads(args.manual_overrides.read_text()).get(str(args.episode)) if args.manual_overrides.is_file() else None
    if manual:
        prompts = [{"frame_index": int(p["frame_index"]), "object_id": 0, "points": None, "point_labels": None,
                    "box": dict(zip(("x0", "y0", "x1", "y1"), map(float, p["box"]))), "mask_path": None} for p in manual]
        (args.output_dir / "sam2_prompts.json").write_text(json.dumps({"prompts": prompts, "provenance": {
            "episode": args.episode, "source": "manual_visual_review"}}, indent=2))
        print(f"manual person prompts for episode {args.episode}: {len(prompts)}")
        return
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = _get_model(args.model_dir)
    samples = list(range(0, total, args.stride))
    dets = {}
    for f in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        found = _detect_frame(model, frame, "a person.", args.box_threshold, args.text_threshold)
        dets[f] = [([d["box"]["x0"], d["box"]["y0"], d["box"]["x1"], d["box"]["y1"]], float(d["confidence"])) for d in found
                   if (d["box"]["y1"] - d["box"]["y0"]) > 0.15 * frame.shape[0]]
    cap.release()
    tracks: list[dict] = []
    for f in samples:
        used = set()
        for tr in tracks:
            if f - tr["frames"][-1] > 4 * args.stride:
                continue
            best, bi = 0.3, None
            for i, (box, conf) in enumerate(dets.get(f, [])):
                if i not in used and iou(box, tr["boxes"][-1]) > best:
                    best, bi = iou(box, tr["boxes"][-1]), i
            if bi is not None:
                used.add(bi)
                tr["frames"].append(f); tr["boxes"].append(dets[f][bi][0]); tr["conf"].append(dets[f][bi][1])
        for i, (box, conf) in enumerate(dets.get(f, [])):
            if i not in used:
                tracks.append({"frames": [f], "boxes": [box], "conf": [conf]})
    for tr in tracks:
        dist = [contact_distance(b, o) for f, b in zip(tr["frames"], tr["boxes"]) if (o := object_box(args.episode, f)) is not None]
        tr["score"] = float(np.median(dist)) if dist else float("inf")
        tr["coverage"] = len(tr["frames"]) / max(len(samples), 1)
    eligible = [t for t in tracks if t["coverage"] >= 0.2] or tracks
    actor = min(eligible, key=lambda t: (t["score"], -len(t["frames"])))
    prompts, last = [], -10 ** 9
    for f, box, conf in zip(actor["frames"], actor["boxes"], actor["conf"]):
        if conf >= 0.35 and f - last >= PROMPT_EVERY:
            prompts.append({"frame_index": int(f), "object_id": 0, "points": None, "point_labels": None,
                            "box": dict(zip(("x0", "y0", "x1", "y1"), map(float, box))), "mask_path": None})
            last = f
    if not prompts:
        raise SystemExit(f"episode {args.episode}: no confident actor detection; add config/manual_person_prompts.json")
    (args.output_dir / "sam2_prompts.json").write_text(json.dumps({"prompts": prompts, "provenance": {
        "episode": args.episode, "source": "groundingdino_person_tracks", "stride": args.stride,
        "actor_score": actor["score"], "actor_coverage": actor["coverage"]}}, indent=2))
    (args.output_dir / "person_tracks.json").write_text(json.dumps(
        [{k: t[k] for k in ("frames", "score", "coverage")} for t in sorted(tracks, key=lambda t: t["score"])[:6]], indent=1))
    print(f"episode {args.episode}: {len(tracks)} person tracks, actor score {actor['score']:.2f}, coverage "
          f"{actor['coverage']:.2f}, {len(prompts)} prompts")


if __name__ == "__main__":
    main()
