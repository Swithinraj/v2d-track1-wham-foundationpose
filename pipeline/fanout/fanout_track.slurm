#!/bin/bash
#SBATCH --job-name=prep-track
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --array=0-29%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep-track-%A_%a.out
#SBATCH --error=logs/prep-track-%A_%a.err

# Fan-out task, tracking phase: FoundationPose + ESKF/RTS (track_object.sh) and the symmetric
# track (track_symmetric.sh, no-op for non-symmetric objects) for ONE episode. Split from
# fanout_ep.slurm because track_object needs inputs/object_meshes/<object>.glb, which the
# prep-mesh array builds in parallel; this task polls for the mesh (and the episode's sam3d
# outputs) for up to 3 h, then runs. Stages already complete are skipped, so it is resumable.
# Submit: mkdir -p logs && sbatch pipeline/fanout/fanout_track.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"

EP="${SLURM_ARRAY_TASK_ID:?}"
TAG="$(episode_tag "${EP}")"
FRAMES="$(episode_field "${EP}" length)"
OBJECT="$(episode_field "${EP}" object)"
echo "== prep-track ${TAG} (${OBJECT}, ${FRAMES} frames) on $(hostname)"

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }
json_count() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }

for try in $(seq 1 36); do
  ok=1
  [[ -s "inputs/object_meshes/${OBJECT}.glb" ]] || { ok=0; echo "waiting for ${OBJECT}.glb (${try}/36)"; }
  [[ -s "outputs/hands/${TAG}.npz" && -s "outputs/sam3d_body/${TAG}/refit.npz" ]] || { ok=0; echo "waiting for sam3d_body ${TAG} (${try}/36)"; }
  [[ "${ok}" == 1 ]] && break
  sleep 300
done
[[ -s "inputs/object_meshes/${OBJECT}.glb" ]] || { echo "${OBJECT}.glb not installed after 3 h" >&2; exit 4; }
[[ -s "outputs/hands/${TAG}.npz" ]] || { echo "sam3d_body ${TAG} incomplete after 3 h" >&2; exit 4; }

if [[ "$(json_count "outputs/object/${TAG}/foundationpose/smoothed")" -eq "${FRAMES}" ]]; then
  echo "SKIP: track_object ${TAG} complete"
else
  t bash pipeline/stages/track_object.sh "${EP}"
fi

if [[ -n "$(ls "outputs/object/${TAG}/foundationpose/symmetric" 2>/dev/null)" ]]; then
  echo "SKIP: track_symmetric ${TAG} complete"
else
  t bash pipeline/stages/track_symmetric.sh "${EP}"
fi

echo "PASS: prep-track ${TAG} — object track (+ symmetric where applicable) present"
