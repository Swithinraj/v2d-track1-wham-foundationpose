#!/bin/bash
#SBATCH --job-name=v2d_t1_cari4d_weights
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_weights_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_weights_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
TOKEN_FILE=/work/smoses2s/nvidea_chal/hugging_face_token.txt
IMAGE="${ROOT}/containers/v2d_cari4d.sif"
DEST="${ROOT}/weights/cari4d"

[[ -s "${TOKEN_FILE}" ]] || { echo "Missing Hugging Face token file" >&2; exit 2; }
[[ "$(stat -c '%a' "${TOKEN_FILE}")" == "600" ]] || {
  echo "Token file must have mode 600" >&2; exit 3;
}
[[ -s "${IMAGE}" ]] || { echo "Missing ${IMAGE}" >&2; exit 4; }
mkdir -p "${DEST}"
mkdir -p "${ROOT}/.cache/huggingface_cari4d" "${ROOT}/.cache/xdg"

# Download only the repository the user can access.  This deliberately skips
# the still-gated SAM 3D Body repository in the upstream all-in-one downloader.
export HF_TOKEN
HF_TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_cari4d \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${IMAGE}" \
  python -c "from pathlib import Path; from v2d.cari4d.lib.download_weights import _download_cari4d; _download_cari4d(Path('/track1/weights/cari4d'))"
unset HF_TOKEN APPTAINERENV_HF_TOKEN

test -s "${DEST}/cari4d/2026-08-25-09-35-57/step200000.pth"
test -s "${DEST}/cari4d/2026-08-25-09-35-57/resolved_config.yaml"
test -s "${DEST}/cari4d/2026-08-25-09-35-57/manifest.json"
echo "PASS: accepted nvidia/cari4d_commercial checkpoint downloaded without SAM 3D Body"
