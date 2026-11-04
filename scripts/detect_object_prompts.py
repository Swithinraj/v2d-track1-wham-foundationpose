#!/usr/bin/env python3
"""Create auditable SAM2 object box prompts from sparse GroundingDINO detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import os

from v2d.grounding_dino.lib.video_to_object_bboxes import _detect_frame, _get_model


ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])


def episode_metadata(index: int) -> dict:
    path = ROOT / "data" / "track_1" / "meta" / "episodes_metadata.jsonl"
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if int(item["episode_index"]) == index:
                return item
    raise ValueError(f"episode {index} is absent from {path}")


def load_manual_override(path: Path, episode: int) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload.get(str(episode))


def write_prompt_artifacts(
    *,
    output: Path,
    video: str,
    metadata: dict,
    prompts: list[dict],
    provenance: dict,
) -> None:
    payload = {"prompts": prompts, "provenance": provenance}
    (output / "sam2_prompts.json").write_text(json.dumps(payload, indent=2))
    primary = prompts[0]
    primary_frame = int(primary["frame_index"])
    (output / "reference_frame.txt").write_text(f"{primary_frame}\n")

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, primary_frame)
    ok, preview = cap.read()
    cap.release()
    if ok:
        box = primary.get("box")
        points = primary.get("points") or []
        labels = primary.get("point_labels") or []
        if box is not None:
            p0 = (round(box["x0"]), round(box["y0"]))
            p1 = (round(box["x1"]), round(box["y1"]))
            cv2.rectangle(preview, p0, p1, (0, 255, 0), 4)
            label_origin = (max(0, p0[0]), max(30, p0[1] - 10))
        else:
            for point, label in zip(points, labels):
                center = (round(point["x"]), round(point["y"]))
                color = (0, 255, 0) if int(label) == 1 else (0, 0, 255)
                cv2.circle(preview, center, 8, color, -1)
            first = points[0]
            label_origin = (max(0, round(first["x"])), max(30, round(first["y"]) - 14))
        source = provenance.get("source", "grounding_dino")
        cv2.putText(
            preview,
            f"{metadata['object']} {source} frame={primary_frame}",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(output / "primary_prompt.jpg"), preview)

    print(json.dumps(provenance, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--min-frame-gap", type=int, default=120)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--manual-overrides",
        default=str(ROOT / "config" / "manual_object_prompts.json"),
    )
    args = parser.parse_args()

    if not 0 <= args.episode <= 29:
        raise SystemExit("ERROR: --episode must be between 0 and 29")
    metadata = episode_metadata(args.episode)
    prompt_text = metadata["object_prompt"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or width <= 0 or height <= 0:
        raise SystemExit(f"ERROR: cannot read video metadata: {args.video}")

    manual = load_manual_override(Path(args.manual_overrides), args.episode)
    if manual is not None:
        frame_index = int(manual["frame_index"])
        box_values = manual.get("box")
        point_values = manual.get("points")
        if (box_values is None) == (point_values is None):
            raise SystemExit("ERROR: manual prompt must provide exactly one of box or points")
        box = None
        points = None
        point_labels = None
        if box_values is not None:
            box_values = [float(value) for value in box_values]
            if len(box_values) != 4:
                raise SystemExit("ERROR: manual box must contain [x0, y0, x1, y1]")
            x0, y0, x1, y1 = box_values
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise SystemExit(f"ERROR: invalid manual box {box_values} for image {width}x{height}")
            box = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            points = [{"x": float(value[0]), "y": float(value[1])} for value in point_values]
            point_labels = [int(value) for value in manual.get("point_labels", [1] * len(points))]
            if not points or len(points) != len(point_labels):
                raise SystemExit("ERROR: manual points and point_labels must be non-empty and equal length")
            if any(not (0 <= p["x"] < width and 0 <= p["y"] < height) for p in points):
                raise SystemExit(f"ERROR: manual point outside image {width}x{height}: {points}")
        if not 0 <= frame_index < total:
            raise SystemExit(f"ERROR: invalid manual frame {frame_index}/{total}")
        prompts = [{
            "frame_index": frame_index,
            "object_id": 1,
            "points": points,
            "point_labels": point_labels,
            "box": box,
            "mask_path": None,
        }]
        provenance = {
            "episode": args.episode,
            "object": metadata["object"],
            "text": prompt_text,
            "primary_frame": frame_index,
            "primary_confidence": None,
            "sample_stride": None,
            "source": "manual_visual_review",
            "note": manual.get("note", ""),
        }
        (output / "sparse_detections.json").write_text("{}\n")
        write_prompt_artifacts(
            output=output,
            video=args.video,
            metadata=metadata,
            prompts=prompts,
            provenance=provenance,
        )
        return

    model = _get_model(args.model_dir)
    sampled = sorted(set(range(0, total, args.stride)) | {total - 1})
    detections: dict[str, list[dict]] = {}
    candidates: list[tuple[float, int, dict]] = []
    image_area = width * height

    for count, frame_idx in enumerate(sampled, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        found = _detect_frame(
            model,
            frame,
            prompt_text,
            args.box_threshold,
            args.text_threshold,
        )
        detections[str(frame_idx)] = found
        for detection in found:
            box = detection["box"]
            area = max(0.0, box["x1"] - box["x0"]) * max(0.0, box["y1"] - box["y0"])
            fraction = area / image_area
            if 0.0001 <= fraction <= 0.65:
                candidates.append((float(detection["confidence"]), frame_idx, detection))
        if count % 10 == 0 or count == len(sampled):
            print(f"sampled {count}/{len(sampled)} frames")

    cap.release()
    (output / "sparse_detections.json").write_text(json.dumps(detections, indent=2))
    if not candidates:
        raise SystemExit(
            f"ERROR: GroundingDINO found no usable detection for {prompt_text!r}; "
            f"create a manual SAM2 prompt for episode {args.episode}"
        )

    selected: list[tuple[float, int, dict]] = []
    for candidate in sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True):   # ties: never compare dicts
        if all(abs(candidate[1] - previous[1]) >= args.min_frame_gap for previous in selected):
            selected.append(candidate)
            if len(selected) == args.max_prompts:
                break

    prompts = []
    for confidence, frame_idx, detection in selected:
        prompts.append({
            "frame_index": frame_idx,
            "object_id": 1,
            "points": None,
            "point_labels": None,
            "box": detection["box"],
            "mask_path": None,
        })

    primary_confidence, primary_frame, primary_detection = selected[0]
    provenance = {
        "episode": args.episode,
        "object": metadata["object"],
        "text": prompt_text,
        "primary_frame": primary_frame,
        "primary_confidence": primary_confidence,
        "sample_stride": args.stride,
        "source": "grounding_dino",
    }
    write_prompt_artifacts(
        output=output,
        video=args.video,
        metadata=metadata,
        prompts=prompts,
        provenance=provenance,
    )


if __name__ == "__main__":
    main()
