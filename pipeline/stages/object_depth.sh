#!/bin/bash
# MoGe-2 metric depth + per-frame intrinsics for one episode -> outputs/object/episode_X/{depth,intrinsics}.
# Usage: object_depth.sh EPISODE   (MOGE_BATCH_SIZE, default 4)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/object/${TAG}"
require_file "${VIDEOS}/${TAG}.mp4"
require_file "${TRACK1_ROOT}/weights/object/moge2/model.pt" "run pipeline/setup/download_weights.sh"
mkdir -p "${OUT}/depth" "${OUT}/intrinsics"

in_container v2d_cari4d.sif python -m v2d.moge.lib.video_to_depth \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --depth_folder "/track1/outputs/object/${TAG}/depth" \
  --intrinsics_folder "/track1/outputs/object/${TAG}/intrinsics" \
  --weights_path /track1/weights/object/moge2 \
  --batch_size "${MOGE_BATCH_SIZE:-4}"

EXPECTED="$(episode_field "${EPISODE}" length)"
DEPTH="$(find "${OUT}/depth" -maxdepth 1 -name '*.png' | wc -l)"
INTRINSICS="$(find "${OUT}/intrinsics" -maxdepth 1 -name '*.json' | wc -l)"
[[ "${DEPTH}" -eq "${EXPECTED}" && "${INTRINSICS}" -eq "${EXPECTED}" ]] || {
  echo "Depth output incomplete: expected=${EXPECTED} depth=${DEPTH} intrinsics=${INTRINSICS}" >&2
  exit 4
}
echo "PASS: MoGe-2 depth ${TAG}, ${EXPECTED} frames"
