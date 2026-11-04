#!/usr/bin/env python3
"""MediaPipe 2D body and hand landmarks for one episode (.venvs/mediapipe, CPU; pipeline/stages/keypoints2d.sh).

PoseLandmarker (heavy, up to 3 people, VIDEO mode) on every frame. Hands are too small in the full frame
for MediaPipe's palm detector, so HandLandmarker (IMAGE mode) runs on an upscaled square crop around each
detected person's pose wrist (side 2.5x the elbow-wrist length, >= 96 px, resized to 320 px).
Everything detected is stored raw; the actor is chosen later against the projected MHR body, so a
background person cannot leak in. Pixel coordinates in the original 1536x1152 image.
Models: weights/mediapipe/{pose_landmarker_heavy,hand_landmarker}.task (pipeline/setup/setup_mediapipe.sh).
Writes outputs/mediapipe/episode_XXXXXX.npz:
  pose [T, 3, 33, 5]     x_px, y_px, z (relative), visibility, presence  (NaN where no person)
  hands [T, 3, 2, 21, 3] per person and side (0 = pose left wrist, 1 = right): x_px, y_px, z
  hand_score [T, 3, 2]   HandLandmarker handedness score of the crop's best hand
Usage: mediapipe_extract.py EPISODE
"""
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
CROP = 320
VIDEOS = ROOT / "data/track_1/videos/chunk-000/observation.images.exo_camera"
MODELS = ROOT / "weights/mediapipe"


def main():
    ep = int(sys.argv[1])
    tag = f"episode_{ep:06d}"
    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_heavy.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=3, min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))
    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2))
    cap = cv2.VideoCapture(str(VIDEOS / f"{tag}.mp4"))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose = np.full((T, 3, 33, 5), np.nan, np.float32)
    hands = np.full((T, 3, 2, 21, 3), np.nan, np.float32)
    hand_score = np.full((T, 3, 2), np.nan, np.float32)
    t0 = time.time()
    for f in range(T):
        ok, bgr = cap.read()
        if not ok:
            print(f"{tag}: video ended at frame {f} of {T}", flush=True)
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(round(f * 1000 / 30))
        res = pose_lm.detect_for_video(image, ts)
        for i, lms in enumerate(res.pose_landmarks[:3]):
            pose[f, i] = [[l.x * W, l.y * H, l.z, l.visibility, l.presence] for l in lms]
            for side, (elbow, wrist) in enumerate(((13, 15), (14, 16))):
                e, w = pose[f, i, elbow, :2], pose[f, i, wrist, :2]
                half = max(48.0, 1.25 * float(np.linalg.norm(w - e)))
                x0, y0 = int(round(w[0] - half)), int(round(w[1] - half))
                x1, y1 = int(round(w[0] + half)), int(round(w[1] + half))
                if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                    continue
                crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
                sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
                crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
                k = CROP / crop.shape[0]
                crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
                hres = hand_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                if hres.hand_landmarks:
                    hands[f, i, side] = [[x0 + l.x * CROP / k, y0 + l.y * CROP / k, l.z] for l in hres.hand_landmarks[0]]
                    hand_score[f, i, side] = hres.handedness[0][0].score
    out = ROOT / "outputs/mediapipe"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", pose=pose, hands=hands, hand_score=hand_score)
    print(f"{tag}: {T} frames in {time.time() - t0:.0f} s | frames with a person {np.isfinite(pose[:, 0, 0, 0]).mean():.0%}, "
          f"person-0 hands found: left {np.isfinite(hands[:, 0, 0, 0, 0]).mean():.0%}, right {np.isfinite(hands[:, 0, 1, 0, 0]).mean():.0%}", flush=True)


if __name__ == "__main__":
    main()
