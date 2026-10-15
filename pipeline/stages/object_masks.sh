#!/bin/bash
# GroundingDINO prompts + SAM 2 propagation for one episode -> outputs/object/episode_X/{prompt,sam2_masks}.
# Reviewed prompt overrides in config/manual_object_prompts.json are applied automatically.
# Usage: object_masks.sh EPISODE   (DINO_STRIDE, default 15)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/object/${TAG}"
require_file "${VIDEOS}/${TAG}.mp4"
require_file "${TRACK1_ROOT}/weights/object/grounding_dino/groundingdino_swint_ogc.pth" "run pipeline/setup/download_weights.sh"
require_file "${TRACK1_ROOT}/weights/sam2/sam2.1_hiera_large.pt" "run pipeline/setup/download_weights.sh"
mkdir -p "${OUT}/prompt" "${OUT}/sam2_masks"

in_container v2d_grounding_dino.sif python /track1/scripts/detect_object_prompts.py \
  --episode "${EPISODE}" \
  --video "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --model-dir /track1/weights/object/grounding_dino \
  --output-dir "/track1/outputs/object/${TAG}/prompt" \
  --stride "${DINO_STRIDE:-15}"

in_container v2d_sam2.sif python -m v2d.sam2.lib.video_to_masks \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --prompts_path "/track1/outputs/object/${TAG}/prompt/sam2_prompts.json" \
  --masks_dir "/track1/outputs/object/${TAG}/sam2_masks" \
  --weights_dir /track1/weights/sam2

in_container v2d_sam2.sif python /track1/scripts/audit_object_masks.py \
  --masks "/track1/outputs/object/${TAG}/sam2_masks/1" \
  --frames "$(episode_field "${EPISODE}" length)" \
  --output "/track1/outputs/object/${TAG}/mask_audit.json"
echo "PASS: GroundingDINO + SAM 2 masks ${TAG}"
