#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export HF_HOME="${TRACK1_ROOT}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${TRACK1_ROOT}/data"

HF_BIN="$(command -v hf || true)"
if [[ -z "${HF_BIN}" ]]; then
  echo "The Hugging Face 'hf' CLI is not available." >&2
  exit 1
fi

"${HF_BIN}" download \
  --repo-type dataset \
  nvidia/video_to_data_challenge \
  --revision 5f68335f3acc802033d1e80728c1633197521de8 \
  --include 'track_1/**' \
  --local-dir "${TRACK1_ROOT}/data"

python3 "${SCRIPT_DIR}/validate_workspace.py" --data-only

