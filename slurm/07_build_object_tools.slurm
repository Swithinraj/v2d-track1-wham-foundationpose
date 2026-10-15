#!/bin/bash
#SBATCH --job-name=v2d_t1_obj_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/object_build_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/object_build_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
apptainer build --fakeroot --force \
  "${ROOT}/containers/v2d_grounding_dino.sif" \
  "${ROOT}/config/grounding_dino.def"
apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  "${ROOT}/containers/v2d_grounding_dino.sif" \
  python -c 'import torch, groundingdino; print(torch.__version__, torch.cuda.get_device_name(0))'
echo "PASS: GroundingDINO object-prompt image built"
