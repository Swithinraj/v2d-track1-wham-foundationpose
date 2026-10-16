#!/bin/bash
# Host virtual environment with MediaPipe (pose + hand landmarkers) and its two models, for
# pipeline/stages/keypoints2d.sh. Uses PYTHON (default python3, must be >= 3.10). Pins: config/upstream.env.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

PYTHON="${PYTHON:-python3}"
"${PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || {
  echo "Python >= 3.10 is required for MediaPipe (set PYTHON=...)." >&2
  exit 2
}
VENV="${TRACK1_ROOT}/.venvs/mediapipe"
[[ -x "${VENV}/bin/python" ]] || "${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
# shellcheck disable=SC2086
"${VENV}/bin/python" -m pip install --quiet ${MEDIAPIPE_PIP}

MODELS="${TRACK1_ROOT}/weights/mediapipe"
mkdir -p "${MODELS}"
fetch() {  # fetch URL SHA256
  local file="${MODELS}/$(basename "$1")"
  [[ -s "${file}" ]] || curl -sSfL -o "${file}" "$1"
  echo "$2  ${file}" | sha256sum -c --quiet || { echo "checksum mismatch: ${file}" >&2; exit 3; }
}
fetch "${MEDIAPIPE_POSE_URL}" "${MEDIAPIPE_POSE_SHA256}"
fetch "${MEDIAPIPE_HAND_URL}" "${MEDIAPIPE_HAND_SHA256}"
"${VENV}/bin/python" -c "import mediapipe, cv2; print('mediapipe', mediapipe.__version__, 'opencv', cv2.__version__)"
echo "PASS: MediaPipe venv at .venvs/mediapipe, models in weights/mediapipe"
