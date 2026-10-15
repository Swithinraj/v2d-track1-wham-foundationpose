#!/bin/bash
#SBATCH --job-name=v2d_dpvo_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/dpvo_build_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/dpvo_build_%j.err

set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
IMAGE="${ROOT}/containers/wham.sif"
SOURCE="${ROOT}/third_party/WHAM/third-party/DPVO"
RUNTIME="${ROOT}/.cache/dpvo_extensions"

[[ -s "${IMAGE}" ]] || { echo "Missing ${IMAGE}" >&2; exit 2; }
[[ -s "${SOURCE}/setup.py" ]] || { echo "Missing DPVO source" >&2; exit 3; }
[[ -s "${SOURCE}/thirdparty/eigen-3.4.0/Eigen/Core" ]] || {
  echo "Missing Eigen 3.4.0 headers under ${SOURCE}/thirdparty" >&2
  exit 4
}

mkdir -p "${RUNTIME}/lib" "${RUNTIME}/build" "${RUNTIME}/home"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# The public WHAM image ships DPVO binaries compiled only for sm_86. Build
# workspace-local extensions for this cluster's V100 (sm_70) and A100 (sm_80).
apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" --pwd /track1/third_party/WHAM/third-party/DPVO \
  "${IMAGE}" \
  env HOME=/track1/.cache/dpvo_extensions/home \
      CUDA_HOME=/usr/local/cuda-11.3 \
      MAX_JOBS="${SLURM_CPUS_PER_TASK}" \
      TORCH_CUDA_ARCH_LIST='7.0;8.0' \
  python setup.py build_ext \
    --build-lib /track1/.cache/dpvo_extensions/lib \
    --build-temp /track1/.cache/dpvo_extensions/build \
    --force

for module in cuda_corr cuda_ba lietorch_backends; do
  test -n "$(find "${RUNTIME}/lib" -maxdepth 1 -type f -name "${module}*.so" -print -quit)" || {
    echo "Missing rebuilt ${module} extension" >&2
    exit 5
  }
done

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" "${IMAGE}" \
  env PYTHONPATH=/track1/.cache/dpvo_extensions/lib \
  python -c 'import torch; import cuda_corr, cuda_ba, lietorch_backends; print("PASS: rebuilt DPVO CUDA extensions import")'

echo "PASS: DPVO extensions built for sm_70 and sm_80 under ${RUNTIME}/lib"
