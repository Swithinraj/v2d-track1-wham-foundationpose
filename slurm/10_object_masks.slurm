#!/bin/bash
#SBATCH --job-name=v2d_t1_masks
#SBATCH --partition=gpu
# GroundingDINO's prebuilt MSDeformAttn extension in this image targets the
# A100 architecture available on wr22-wr25. Older cluster GPUs emit
# "no kernel image is available" and silently return no detections.
#SBATCH --nodelist=wr22,wr23,wr24,wr25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/masks_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/masks_%A_%a.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
if [[ -n "${EPISODE:-}" ]]; then
  EPISODE="${EPISODE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EPISODE="$(( ${EPISODE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
else
  EPISODE=""
fi
[[ "${EPISODE}" =~ ^[0-9]+$ ]] && ((EPISODE >= 0 && EPISODE <= 29)) || {
  echo "Set EPISODE to an integer from 0 through 29." >&2; exit 2;
}
PADDED="$(printf '%06d' "${EPISODE}")"
VIDEO="${ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4"
OUT="${ROOT}/outputs/object/episode_${PADDED}"
PROMPTS="${OUT}/prompt"
MASKS="${OUT}/sam2_masks"
mkdir -p "${PROMPTS}" "${MASKS}"

test -s "${VIDEO}"
test -s "${ROOT}/weights/object/grounding_dino/groundingdino_swint_ogc.pth"
test -s "${ROOT}/weights/sam2/sam2.1_hiera_large.pt"

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_grounding_dino.sif" \
  python /track1/scripts/detect_object_prompts.py \
  --episode "${EPISODE}" \
  --video "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4" \
  --model-dir /track1/weights/object/grounding_dino \
  --output-dir "/track1/outputs/object/episode_${PADDED}/prompt" \
  --stride "${DINO_STRIDE:-15}"

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_sam2.sif" \
  python -m v2d.sam2.lib.video_to_masks \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4" \
  --prompts_path "/track1/outputs/object/episode_${PADDED}/prompt/sam2_prompts.json" \
  --masks_dir "/track1/outputs/object/episode_${PADDED}/sam2_masks" \
  --weights_dir /track1/weights/sam2

EXPECTED="$(python3 -c "import json; print(next(x['length'] for x in map(json.loads, open('${ROOT}/data/track_1/meta/episodes.jsonl')) if x['episode_index']==${EPISODE}))")"
python3 "${ROOT}/scripts/audit_object_masks.py" \
  --masks "${MASKS}/1" --frames "${EXPECTED}" --output "${OUT}/mask_audit.json"
echo "PASS: GroundingDINO + SAM2 object masks episode ${PADDED}"
