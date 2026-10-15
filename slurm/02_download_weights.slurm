#!/bin/bash
#SBATCH --job-name=v2d_t1_weights
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/weights_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/weights_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd "${ROOT}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" \
  "${ROOT}/weights/cari4d" "${ROOT}/weights/sam2" "${ROOT}/weights/sam3d"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Export it only when submitting this job." >&2
  exit 2
fi

if [[ ! -f "${ROOT}/containers/v2d_cari4d.sif" ]]; then
  echo "Missing ${ROOT}/containers/v2d_cari4d.sif; complete the build job first." >&2
  exit 3
fi
for image in v2d_sam2.sif v2d_sam3d.sif; do
  if [[ ! -f "${ROOT}/containers/${image}" ]]; then
    echo "Missing ${ROOT}/containers/${image}; complete the input-tools build job first." >&2
    exit 3
  fi
done

export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
apptainer exec --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}/weights/cari4d:/data/weights" \
  "${ROOT}/containers/v2d_cari4d.sif" \
  python -m v2d.cari4d.lib.download_weights --output_dir /data/weights
apptainer exec --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}/weights/sam2:/data/weights" \
  "${ROOT}/containers/v2d_sam2.sif" \
  python -m v2d.sam2.lib.download_weights --output_dir /data/weights
apptainer exec --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}/weights/sam3d:/data/weights" \
  "${ROOT}/containers/v2d_sam3d.sif" \
  python -m v2d.sam3d.lib.download_weights --output_dir /data/weights
echo "PASS: CARI4D, SAM2, and SAM3D weights downloaded"

