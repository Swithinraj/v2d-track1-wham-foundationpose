#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

V2D_REPO="${TRACK1_ROOT}/third_party/video_to_data"
KIT_ZIP="${V2D_REPO}/docs/v2d_challenge/assets/v2d_submission_kit.zip"
KIT_ROOT="${TRACK1_ROOT}/official/v2d_submission_kit"

if [[ ! -d "${V2D_REPO}/.git" ]]; then
  echo "Official toolkit checkout is missing at ${V2D_REPO}" >&2
  exit 1
fi
if [[ ! -f "${KIT_ZIP}" ]]; then
  echo "Submission kit archive is missing at ${KIT_ZIP}" >&2
  exit 1
fi

mkdir -p \
  "${TRACK1_ROOT}/.cache" \
  "${TRACK1_ROOT}/.cache/pip" \
  "${TRACK1_ROOT}/.cache/tmp" \
  "${TRACK1_ROOT}/.venvs" \
  "${TRACK1_ROOT}/containers" \
  "${TRACK1_ROOT}/data" \
  "${TRACK1_ROOT}/inputs" \
  "${TRACK1_ROOT}/logs" \
  "${TRACK1_ROOT}/official" \
  "${TRACK1_ROOT}/outputs" \
  "${TRACK1_ROOT}/submissions" \
  "${TRACK1_ROOT}/weights"

if [[ ! -f "${KIT_ROOT}/KIT_MANIFEST.json" ]]; then
  unzip -q "${KIT_ZIP}" -d "${TRACK1_ROOT}/official"
fi

load_python
python --version

SUBMISSION_VENV="${TRACK1_ROOT}/.venvs/submission"
if [[ ! -x "${SUBMISSION_VENV}/bin/python" ]]; then
  python -m venv "${SUBMISSION_VENV}"
fi
"${SUBMISSION_VENV}/bin/python" -m pip install --upgrade pip
"${SUBMISSION_VENV}/bin/python" -m pip install -r "${KIT_ROOT}/requirements.txt"
"${SUBMISSION_VENV}/bin/python" "${KIT_ROOT}/eval_reconstruction.py" --help >/dev/null

V2D_HOST_VENV="${TRACK1_ROOT}/.venvs/v2d-host"
if [[ ! -x "${V2D_HOST_VENV}/bin/python" ]]; then
  python -m venv "${V2D_HOST_VENV}"
fi
source "${V2D_HOST_VENV}/bin/activate"
python -m pip install --upgrade pip
bash "${V2D_REPO}/reconstruction/scripts/install_packages.sh"
python -m pip install pytest==9.1.1
python -m v2d.cari4d.docker.run_inference --help >/dev/null

echo "Workspace bootstrap complete: ${TRACK1_ROOT}"

