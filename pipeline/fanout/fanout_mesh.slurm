#!/bin/bash
#SBATCH --job-name=prep-mesh
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep-mesh-%A_%a.out
#SBATCH --error=logs/prep-mesh-%A_%a.err

# Fan-out task: rebuild the metric mesh for ONE object (index SLURM_ARRAY_TASK_ID into the
# sorted distinct object list, 10 objects x 3 episodes). object_mesh.sh needs all of the
# object's episodes' depth + masks, so the task polls for them (5-min interval, up to 3 h)
# before running. Skips objects whose mesh is already installed (e.g. hula_hoop from the
# smoke test). Resumable: resubmit failed tasks unchanged.
# Submit: mkdir -p logs && sbatch pipeline/fanout/fanout_mesh.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"

IDX="${SLURM_ARRAY_TASK_ID:?}"
mapfile -t OBJECTS < <("${HOST_PYTHON}" - "${DATA}/meta/episodes_metadata.jsonl" <<'PY'
import json, sys
objs = sorted({json.loads(l)["object"] for l in open(sys.argv[1])})
print("\n".join(objs))
PY
)
OBJECT="${OBJECTS[${IDX}]:?}"
[[ -n "${OBJECT}" ]] || { echo "no object at index ${IDX}" >&2; exit 2; }
mapfile -t EPS < <("${HOST_PYTHON}" - "${DATA}/meta/episodes_metadata.jsonl" "${OBJECT}" <<'PY'
import json, sys
eps = sorted(json.loads(l)["episode_index"] for l in open(sys.argv[1]) if json.loads(l)["object"] == sys.argv[2])
print("\n".join(str(e) for e in eps))
PY
)
echo "== prep-mesh ${OBJECT}: episodes ${EPS[*]}"

if [[ -s "inputs/object_meshes/${OBJECT}.glb" ]]; then
  echo "SKIP: inputs/object_meshes/${OBJECT}.glb already installed"
  exit 0
fi

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }
png_count() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
ep_ready() { local tag; tag="$(episode_tag "$1")"; \
  [[ "$(png_count "outputs/object/${tag}/depth")" -eq "$(episode_field "$1" length)" \
  && -s "outputs/object/${tag}/mask_audit.json" ]]; }

# wait for this object's episodes' depth + masks (the episode array runs in parallel)
for try in $(seq 1 36); do
  missing=()
  for ep in "${EPS[@]}"; do ep_ready "${ep}" || missing+=("${ep}"); done
  if [[ "${#missing[@]}" -eq 0 ]]; then break; fi
  echo "waiting for depth+masks of episodes: ${missing[*]} (${try}/36)"; sleep 300
done
for ep in "${EPS[@]}"; do ep_ready "${ep}" || { echo "episodes not ready after 3 h: ${missing[*]}" >&2; exit 4; }; done

t bash pipeline/stages/object_mesh.sh "${OBJECT}"
echo "PASS: prep-mesh ${OBJECT} — inputs/object_meshes/${OBJECT}.glb installed"
