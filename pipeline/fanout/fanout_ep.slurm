#!/bin/bash
#SBATCH --job-name=prep-ep
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --array=0-29%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep-ep-%A_%a.out
#SBATCH --error=logs/prep-ep-%A_%a.err

# Fan-out task: rebuild every per-episode output for ONE episode (SLURM_ARRAY_TASK_ID):
#   object_depth -> object_masks -> keypoints2d -> person -> sam3d_body -> track_object
#   -> track_symmetric (no-op for non-symmetric objects).
# Each stage is skipped when its completion sentinel already exists (e.g. the smoke test's
# episode group), so the array is resumable: resubmit failed tasks with the same array range.
# Object meshes (object_mesh.sh) are a separate job: pipeline/fanout/fanout_mesh.slurm.
# Submit (repo root on a CLAIX login node): mkdir -p logs && sbatch pipeline/fanout/fanout_ep.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"

EP="${SLURM_ARRAY_TASK_ID:?}"
TAG="$(episode_tag "${EP}")"
FRAMES="$(episode_field "${EP}" length)"
echo "== prep-ep ${TAG} (${FRAMES} frames) on $(hostname)"

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }

png_count() { find "$1" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }

# 1. MoGe-2 depth + intrinsics
if [[ "$(png_count "outputs/object/${TAG}/depth")" -eq "${FRAMES}" \
   && "$(find "outputs/object/${TAG}/intrinsics" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)" -eq "${FRAMES}" ]]; then
  echo "SKIP: depth ${TAG} complete"
else
  t bash pipeline/stages/object_depth.sh "${EP}"
fi

# 2. object masks
if [[ -s "outputs/object/${TAG}/mask_audit.json" ]]; then
  echo "SKIP: object masks ${TAG} complete"
else
  t bash pipeline/stages/object_masks.sh "${EP}"
fi

# 3. MediaPipe keypoints (CPU)
if [[ -s "outputs/mediapipe/${TAG}.npz" ]]; then
  echo "SKIP: keypoints ${TAG} complete"
else
  t bash pipeline/stages/keypoints2d.sh "${EP}"
fi

# 4. actor masks
if [[ "$(png_count "outputs/person/${TAG}/sam2_masks/0")" -eq "${FRAMES}" ]]; then
  echo "SKIP: person masks ${TAG} complete"
else
  t bash pipeline/stages/person.sh "${EP}"
fi

# 5. SAM 3D Body (body mode + mask prompt, the BMQ recipe's human)
if [[ -s "outputs/hands/${TAG}.npz" && -s "outputs/sam3d_body/${TAG}/refit.npz" ]]; then
  echo "SKIP: sam3d_body ${TAG} complete"
else
  export SAM3D_BODY_ARGS="--inference_type body --mask_prompt"
  t bash pipeline/stages/sam3d_body.sh "${EP}"
fi

# 6. FoundationPose track + ESKF/RTS smoothing
if [[ "$(find "outputs/object/${TAG}/foundationpose/smoothed" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)" -eq "${FRAMES}" ]]; then
  echo "SKIP: track_object ${TAG} complete"
else
  t bash pipeline/stages/track_object.sh "${EP}"
fi

# 7. symmetric track (no-op unless the object is in config/symmetric_objects.json)
if [[ -n "$(ls "outputs/object/${TAG}/foundationpose/symmetric" 2>/dev/null)" ]]; then
  echo "SKIP: track_symmetric ${TAG} complete"
else
  t bash pipeline/stages/track_symmetric.sh "${EP}"
fi

echo "PASS: prep-ep ${TAG} — all per-episode outputs present"
