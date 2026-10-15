#!/bin/bash
# Shared settings for every pipeline script. Source it; do not execute it.
#
# Everything lives under the repository root (override with TRACK1_ROOT). Heavy work runs
# inside Apptainer images with the repository mounted at /track1, so the host only needs
# bash, Apptainer, an NVIDIA driver, git, and Python >= 3.10 (stdlib) for small helpers.

set -euo pipefail

TRACK1_ROOT="${TRACK1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRACK1_ROOT
CONTAINERS="${TRACK1_ROOT}/containers"
DATA="${TRACK1_ROOT}/data/track_1"
VIDEOS="${DATA}/videos/chunk-000/observation.images.exo_camera"
VENV_PYTHON="${TRACK1_ROOT}/.venvs/submission/bin/python"
HOST_PYTHON="${TRACK1_HOST_PYTHON:-python3}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CONTAINERS}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${CONTAINERS}/apptainer-tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# shellcheck source=../config/upstream.env
source "${TRACK1_ROOT}/config/upstream.env"

# in_container IMAGE COMMAND...: run COMMAND in containers/IMAGE with the GPU, the repository at
# /track1 and a writable HOME. (Apptainer refuses --env HOME=..., which would leave HOME on the
# unmounted host home; warp and matplotlib then fail to create their caches.)
in_container() {
  local image="$1"
  shift
  apptainer exec --nv --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" --pwd /track1 \
    "${CONTAINERS}/${image}" env HOME=/tmp MPLCONFIGDIR=/tmp/mpl "$@"
}

require_file() {
  [[ -s "$1" ]] || { echo "Missing $1${2:+ ($2)}" >&2; exit 3; }
}

require_episode() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && (($1 >= 0 && $1 <= 29)) || {
    echo "Episode must be an integer from 0 through 29." >&2
    exit 2
  }
}

episode_tag() { printf 'episode_%06d' "$1"; }

episode_field() {  # episode_field EPISODE FIELD (episodes.jsonl or episodes_metadata.jsonl)
  "${HOST_PYTHON}" - "${DATA}/meta" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
meta, episode, field = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for name in ("episodes.jsonl", "episodes_metadata.jsonl"):
    for line in (meta / name).read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("episode_index") == episode and field in row:
            print(row[field])
            raise SystemExit
raise SystemExit(f"episode {episode} has no field {field!r}")
PY
}

all_objects() {
  "${HOST_PYTHON}" - "${DATA}/meta/episodes_metadata.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print("\n".join(sorted({r["object"] for r in rows})))
PY
}

object_episodes() {
  "${HOST_PYTHON}" - "${DATA}/meta/episodes_metadata.jsonl" "$1" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(" ".join(str(r["episode_index"]) for r in rows if r["object"] == sys.argv[2]))
PY
}
