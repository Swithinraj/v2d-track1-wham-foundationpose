#!/bin/bash
# Rebuild WHAM's DPVO CUDA extensions for the local GPUs. The public WHAM image only ships
# sm_86 binaries. Set TORCH_CUDA_ARCH_LIST to your GPUs; the image's CUDA 11.3 supports up to 8.6.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

ARCHS="${TORCH_CUDA_ARCH_LIST:-7.0;8.0;8.6}"
SOURCE=third_party/WHAM/third-party/DPVO
OUT="${TRACK1_ROOT}/.cache/dpvo_extensions"
require_file "${CONTAINERS}/wham.sif" "run pipeline/setup/build_containers.sh wham"
require_file "${TRACK1_ROOT}/${SOURCE}/thirdparty/eigen-3.4.0/Eigen/Core" "run pipeline/setup/fetch_sources.sh"
mkdir -p "${OUT}/lib" "${OUT}/build"

apptainer exec --nv --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" --pwd "/track1/${SOURCE}" \
  "${CONTAINERS}/wham.sif" \
  env HOME=/tmp CUDA_HOME=/usr/local/cuda-11.3 MAX_JOBS="${MAX_JOBS:-$(nproc)}" TORCH_CUDA_ARCH_LIST="${ARCHS}" \
  python setup.py build_ext --build-lib /track1/.cache/dpvo_extensions/lib \
    --build-temp /track1/.cache/dpvo_extensions/build --force

for module in cuda_corr cuda_ba lietorch_backends; do
  [[ -n "$(find "${OUT}/lib" -maxdepth 1 -name "${module}*.so" -print -quit)" ]] || {
    echo "Missing rebuilt ${module} extension" >&2
    exit 5
  }
done
# torch must be imported first so the extensions find its shared libraries.
apptainer exec --nv --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" "${CONTAINERS}/wham.sif" \
  env PYTHONPATH=/track1/.cache/dpvo_extensions/lib \
  python -c 'import torch, cuda_corr, cuda_ba, lietorch_backends'
echo "PASS: DPVO extensions for ${ARCHS} in .cache/dpvo_extensions/lib"
