#!/bin/bash
#SBATCH --job-name=v2d_t1_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/build_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/build_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "job=${SLURM_JOB_ID} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

apptainer build --fakeroot --force \
  "${ROOT}/containers/v2d_cari4d.sif" \
  "${ROOT}/config/cari4d.def"
apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  "${ROOT}/containers/v2d_cari4d.sif" \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
echo "PASS: CARI4D Apptainer image built"

