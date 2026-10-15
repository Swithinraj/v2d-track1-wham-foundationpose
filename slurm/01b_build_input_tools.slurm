#!/bin/bash
#SBATCH --job-name=v2d_t1_inputs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/build_inputs_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/build_inputs_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "job=${SLURM_JOB_ID} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

apptainer build --fakeroot --force \
  "${ROOT}/containers/v2d_sam2.sif" \
  "${ROOT}/config/sam2.def"
apptainer build --fakeroot --force \
  "${ROOT}/containers/v2d_sam3d.sif" \
  "${ROOT}/config/sam3d.def"

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  "${ROOT}/containers/v2d_sam2.sif" \
  python -c 'import torch; print("sam2", torch.__version__, torch.cuda.get_device_name(0))'
apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  "${ROOT}/containers/v2d_sam3d.sif" \
  python -c 'import torch; print("sam3d", torch.__version__, torch.cuda.get_device_name(0))'
echo "PASS: SAM2 and SAM3D Apptainer images built"

