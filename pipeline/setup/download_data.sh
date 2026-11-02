#!/bin/bash
# Download only the public Track 1 release (videos + metadata) at the pinned
# dataset revision, using the host submission virtual environment.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

require_file "${VENV_PYTHON}" "run pipeline/setup/setup_venv.sh first"
"${VENV_PYTHON}" "${TRACK1_ROOT}/pipeline/setup/download_track1.py"
"${HOST_PYTHON}" "${TRACK1_ROOT}/scripts/validate_workspace.py" --data-only
