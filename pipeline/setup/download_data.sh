#!/bin/bash
# Download the public Track 1 release (videos + metadata) at the pinned dataset revision.
# A Hugging Face token is used if HF_TOKEN or HF_TOKEN_FILE is set.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

require_file "${CONTAINERS}/v2d_cari4d.sif" "run pipeline/setup/build_containers.sh cari4d"
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
fi
[[ -n "${HF_TOKEN:-}" ]] && export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
mkdir -p "${TRACK1_ROOT}/data" "${TRACK1_ROOT}/.cache/huggingface"

in_container v2d_cari4d.sif HF_HOME=/track1/.cache/huggingface HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download('${V2D_DATASET_ID}', repo_type='dataset', revision='${V2D_DATASET_REVISION}',
                  allow_patterns=['track_1/**'], local_dir='/track1/data')
"
unset APPTAINERENV_HF_TOKEN
"${HOST_PYTHON}" "${TRACK1_ROOT}/scripts/validate_workspace.py" --data-only
