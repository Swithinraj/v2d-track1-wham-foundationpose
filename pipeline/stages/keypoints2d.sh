#!/bin/bash
# 2D keypoints of one episode for the human refinement stage:
#   MediaPipe pose (<= 3 people) + hand landmarks on wrist crops -> outputs/mediapipe/episode_X.npz (CPU)
#   WHAM's own ViTPose keypoints of the exported actor track     -> outputs/vitpose/episode_X.npz (held-out check)
# Usage: keypoints2d.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
MP_PYTHON="${TRACK1_ROOT}/.venvs/mediapipe/bin/python"
require_file "${MP_PYTHON}" "run pipeline/setup/setup_mediapipe.sh"
require_file "${TRACK1_ROOT}/outputs/wham/${TAG}/wham_smpl_world.npz" "run pipeline/stages/wham.sh ${EPISODE}"

"${MP_PYTHON}" "${TRACK1_ROOT}/scripts/mediapipe_landmarks.py" "${EPISODE}"
in_container wham.sif python /track1/scripts/export_vitpose_keypoints.py "${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/mediapipe/${TAG}.npz"
require_file "${TRACK1_ROOT}/outputs/vitpose/${TAG}.npz"
