#!/bin/bash
# 2D keypoints of one episode for the human refinement stage:
#   MediaPipe pose (<= 3 people) + hand landmarks on wrist crops -> outputs/mediapipe/episode_X.npz (CPU)
# The actor's landmarks are chosen later against the projected SAM 3D Body human. The held-out 2D check of the
# refinement uses SAM 3D Body's own raw keypoints (outputs/keypoints2d_sam, sam3d_body.sh).
# Usage: keypoints2d.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
MP_PYTHON="${TRACK1_ROOT}/.venvs/mediapipe/bin/python"
require_file "${MP_PYTHON}" "run pipeline/setup/setup_mediapipe.sh"
"${MP_PYTHON}" "${TRACK1_ROOT}/scripts/mediapipe_landmarks.py" "${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/mediapipe/${TAG}.npz"
echo "PASS: MediaPipe landmarks ${TAG}"
