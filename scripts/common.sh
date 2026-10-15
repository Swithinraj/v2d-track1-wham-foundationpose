#!/bin/bash
set -euo pipefail

TRACK1_ROOT=/work/smoses2s/nvidea_chal/TRACK1
WORKSPACE_ROOT=/work/smoses2s/nvidea_chal

if [[ ! -d "${TRACK1_ROOT}" ]]; then
  echo "Missing ${TRACK1_ROOT}" >&2
  exit 1
fi

assert_workspace_path() {
  local candidate
  candidate="$(realpath -m "$1")"
  case "${candidate}" in
    "${WORKSPACE_ROOT}"|"${WORKSPACE_ROOT}"/*) ;;
    *)
      echo "Refusing path outside ${WORKSPACE_ROOT}: ${candidate}" >&2
      exit 2
      ;;
  esac
}

load_python() {
  module load python3/3.12.1
}

export TRACK1_ROOT WORKSPACE_ROOT
export PIP_CACHE_DIR="${TRACK1_ROOT}/.cache/pip"
export TMPDIR="${TRACK1_ROOT}/.cache/tmp"

