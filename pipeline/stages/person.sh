#!/bin/bash
# The actor (the person handling the object) for one episode -> outputs/person/episode_X/{prompt,sam2_masks/0}:
#   scripts/detect_person_prompts.py   GroundingDINO person tracks; the actor is the track closest to the object mask
#   SAM 2 video propagation            person mask on every frame (object id 0)
# Needs the episode's object masks (object_masks.sh). Usage: person.sh EPISODE   (PERSON_STRIDE, default 10)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/person/${TAG}"
require_file "${TRACK1_ROOT}/outputs/object/${TAG}/mask_audit.json" "run pipeline/stages/object_masks.sh ${EPISODE}"
mkdir -p "${OUT}/prompt" "${OUT}/sam2_masks"
in_container v2d_grounding_dino.sif python /track1/scripts/detect_person_prompts.py --episode "${EPISODE}" \
  --video "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --model-dir /track1/weights/object/grounding_dino --output-dir "/track1/outputs/person/${TAG}/prompt" \
  --stride "${PERSON_STRIDE:-10}"
in_container v2d_sam2.sif python -m v2d.sam2.lib.video_to_masks \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --prompts_path "/track1/outputs/person/${TAG}/prompt/sam2_prompts.json" \
  --masks_dir "/track1/outputs/person/${TAG}/sam2_masks" --weights_dir /track1/weights/sam2
FRAMES="$(episode_field "${EPISODE}" length)"
COUNT="$(find "${OUT}/sam2_masks/0" -maxdepth 1 -name '*.png' | wc -l)"
[[ "${COUNT}" -eq "${FRAMES}" ]] || { echo "Expected ${FRAMES} person masks, found ${COUNT}" >&2; exit 4; }
echo "PASS: actor masks ${TAG}"
