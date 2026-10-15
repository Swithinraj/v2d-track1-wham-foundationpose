#!/bin/bash
#SBATCH --job-name=v2d_t1_obj_weights
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/object_weights_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/object_weights_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
OBJECT_WEIGHTS="${ROOT}/weights/object"
cd "${ROOT}"
mkdir -p "${OBJECT_WEIGHTS}/moge2" "${OBJECT_WEIGHTS}/foundationpose" \
  "${OBJECT_WEIGHTS}/grounding_dino" "${ROOT}/weights/sam2" \
  "${ROOT}/.cache/huggingface_object" "${ROOT}/.cache/xdg"

for image in v2d_cari4d.sif v2d_sam2.sif v2d_grounding_dino.sif; do
  [[ -s "${ROOT}/containers/${image}" ]] || { echo "Missing ${image}" >&2; exit 2; }
done

apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_object \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_sam2.sif" \
  python -m v2d.sam2.lib.download_weights --output_dir /track1/weights/sam2

apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_object \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  python -m v2d.moge.lib.download_weights --output_dir /track1/weights/object/moge2

apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_object \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  python -m v2d.foundation_pose.lib.download_weights \
  --output_dir /track1/weights/object/foundationpose \
  --backend nvlabs_pytorch

apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_object \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_grounding_dino.sif" \
  python -m v2d.grounding_dino.lib.download_weights \
  --output_dir /track1/weights/object/grounding_dino

test -s "${ROOT}/weights/sam2/sam2.1_hiera_large.pt"
test -s "${OBJECT_WEIGHTS}/moge2/model.pt"
test -s "${OBJECT_WEIGHTS}/grounding_dino/groundingdino_swint_ogc.pth"
test -s "${OBJECT_WEIGHTS}/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth"
test -s "${OBJECT_WEIGHTS}/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth"
echo "PASS: all non-gated object-branch weights downloaded"
