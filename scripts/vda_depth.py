"""Metric Video Depth Anything (ViT-L) depth for whole episodes, in the pipeline's depth encoding (GPU; an interpreter
with torch, OpenCV, PIL and the Video-Depth-Anything code). Used only for the human depth trend's qn series
(pipeline/stages/vda_prep.sh, recipes/trend.md); every other stage keeps MoGe-2's depth.
Each video is resized to 1280 px on its long side and run through VDA at input size 518 (fp16); the output is rescaled
per episode by one constant, c = median over every 10th frame of median(MoGe-2 / VDA) on background pixels (person and
object masks dilated by 41 px and excluded), which puts it on MoGe-2's absolute scale without touching its temporal
structure (a per-episode constant cancels in the trend's normalised qn anyway). Writes
outputs/object_vda/<episode>/depth/%06d.png (uint16 raw = 65535 / (depth_m + 1), full resolution) and scale.json.
Usage: vda_depth.py --root TRACK1_ROOT --vda_repo DIR --weights FILE --episodes E [E ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def moge_depth(root, tag, f):
    raw = np.array(Image.open(root / f"outputs/object/{tag}/depth/{f:06d}.png")).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(raw > 0, 65535.0 / raw - 1.0, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--vda_repo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.vda_repo))
    from video_depth_anything.video_depth import VideoDepthAnything
    m = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    m.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
    m = m.cuda().eval()
    R0 = args.root
    for ep in args.episodes:
        tag = f"episode_{ep:06d}"
        out = R0 / f"outputs/object_vda/{tag}/depth"
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(R0 / f"data/track_1/videos/chunk-000/observation.images.exo_camera/{tag}.mp4"))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        H, W = frames[0].shape[:2]
        sc = 1280 / max(H, W)
        fr = np.stack([cv2.resize(f, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) for f in frames])
        t = time.time()
        depths, _ = m.infer_video_depth(fr, 30, input_size=518, device="cuda", fp32=False)
        sec = (time.time() - t) / len(fr)
        k = np.ones((41, 41), np.uint8)
        ratios = []
        for f in range(0, len(depths), 10):
            d = cv2.resize(depths[f].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            p = cv2.imread(str(R0 / f"outputs/person/{tag}/sam2_masks/0/{f:06d}.png"), 0)
            o = cv2.imread(str(R0 / f"outputs/object/{tag}/sam2_masks/1/{f:06d}.png"), 0)
            bg = cv2.dilate(((p > 127) | (o > 127)).astype(np.uint8), k) == 0
            mg = moge_depth(R0, tag, f)
            sel = bg & np.isfinite(mg) & (mg > 0.3) & (d > 0.3)
            ratios.append(np.median(mg[sel] / d[sel]))
        c = float(np.median(ratios))
        for f, d in enumerate(depths):
            dm = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * c
            raw = np.clip(np.round(65535.0 / (np.maximum(dm, 1e-3) + 1.0)), 1, 65535).astype(np.uint16)
            Image.fromarray(raw).save(out / f"{f:06d}.png")
        (out.parent / "scale.json").write_text(json.dumps({"scale_to_moge": c, "frames": len(depths), "sec_per_frame": sec,
                                                           "ratio_spread": float(np.std(ratios) / np.mean(ratios))}))
        print(f"{tag}: {len(depths)} frames, {sec:.3f} s/frame, scale to MoGe {c:.4f}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
