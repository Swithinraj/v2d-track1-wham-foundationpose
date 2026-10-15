#!/bin/bash
#SBATCH --job-name=v2d_t1_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/apptainer_probe_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/apptainer_probe_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"
export PATH="${ROOT}/bin:${PATH}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "job=${SLURM_JOB_ID} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "apptainer=$(command -v apptainer)"
apptainer --version

# Test both the HPC-native definition build path and NVIDIA GPU passthrough.
apptainer build --fakeroot --force \
  "${ROOT}/containers/apptainer_probe.sif" \
  "${ROOT}/config/apptainer_probe.def"
apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  "${ROOT}/containers/apptainer_probe.sif" nvidia-smi
echo "PASS: Apptainer definition builds work and can access the allocated GPU"

