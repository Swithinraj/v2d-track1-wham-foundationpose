#!/bin/bash
#SBATCH --job-name=v2d_t1_wham_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_build_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_build_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"

export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "job=${SLURM_JOB_ID} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

apptainer build --fakeroot --ignore-fakeroot-command --force \
  "${ROOT}/containers/wham.sif" \
  "${ROOT}/config/wham.def"

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" --pwd /track1/third_party/WHAM \
  "${ROOT}/containers/wham.sif" \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" --pwd /track1/third_party/WHAM \
  "${ROOT}/containers/wham.sif" \
  python demo.py --help >/dev/null

echo "PASS: WHAM Apptainer image built and demo imports validated"
