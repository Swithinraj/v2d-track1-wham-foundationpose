#!/bin/bash
# Host virtual environment for assembly and packing (the official kit's requirements).
# Uses PYTHON (default python3, must be >= 3.10).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

PYTHON="${PYTHON:-python3}"
"${PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || {
  echo "Python >= 3.10 is required for the submission kit (set PYTHON=...)." >&2
  exit 2
}
KIT="${TRACK1_ROOT}/official/v2d_submission_kit"
require_file "${KIT}/requirements.txt" "run pipeline/setup/fetch_sources.sh first"

VENV="${TRACK1_ROOT}/.venvs/submission"
[[ -x "${VENV}/bin/python" ]] || "${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
"${VENV}/bin/python" -m pip install --quiet -r "${KIT}/requirements.txt"
(cd "${KIT}" && "${VENV}/bin/python" eval_reconstruction.py --help >/dev/null)
echo "PASS: submission venv at .venvs/submission"
