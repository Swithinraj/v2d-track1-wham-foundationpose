#!/bin/bash
#SBATCH --job-name=v2d_t1_wham_assets
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_assets_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_assets_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
IMAGE="${ROOT}/containers/wham.sif"
ASSETS="${ROOT}/weights/wham"
DOWNLOADS="${ASSETS}/.downloads"
MHR_ROOT="${ROOT}/weights/mhr"

cd "${ROOT}"
mkdir -p "${ASSETS}/checkpoints" "${ASSETS}/dataset" "${DOWNLOADS}" "${MHR_ROOT}"
mkdir -p "${ASSETS}/.home/.cache/gdown"

if [[ ! -f "${IMAGE}" ]]; then
  echo "Missing ${IMAGE}; complete 04_build_wham.slurm first." >&2
  exit 2
fi

download_gdrive() {
  local file_id="$1"
  local relative_path="$2"
  local destination="${ASSETS}/${relative_path}"
  if [[ -s "${destination}" ]]; then
    echo "SKIP existing ${relative_path}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  apptainer exec --no-home --cleanenv \
    --bind "${ASSETS}:/assets" "${IMAGE}" \
    env HOME=/assets/.home gdown --no-cookies "https://drive.google.com/uc?id=${file_id}" \
    -O "/assets/${relative_path}"
}

# Public WHAM checkpoints used by configs/yamls/demo.yaml.
download_gdrive 19qkI-a6xuwob9_RFNSPWf1yWErwVVlks checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
download_gdrive 1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y checkpoints/hmr2a.ckpt
download_gdrive 1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT checkpoints/dpvo.pth
download_gdrive 1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo checkpoints/yolov8x.pt
download_gdrive 1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso checkpoints/vitpose-h-multi-coco.pth

# Public auxiliary regressors and template metadata. This archive intentionally does not
# contain the separately licensed SMPL_NEUTRAL.pkl file.
download_gdrive 1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ .downloads/body_models.tar.gz
tar -xzf "${DOWNLOADS}/body_models.tar.gz" -C "${ASSETS}/dataset"

# Public Apache-2.0 MHR release: the Meta topology converter needs the full
# asset set, while the official Track 1 packer also uses mhr_model.pt.
MHR_ZIP="${MHR_ROOT}/assets.zip"
if [[ ! -s "${MHR_ZIP}" ]]; then
  apptainer exec --no-home --cleanenv \
    --bind "${MHR_ROOT}:/mhr" "${IMAGE}" \
    wget -q --show-progress \
    https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip \
    -O /mhr/assets.zip
fi
unzip -o "${MHR_ZIP}" -d "${MHR_ROOT}"
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  ${MHR_ROOT}/assets/mhr_model.pt" | sha256sum -c -

find "${ASSETS}/checkpoints" -maxdepth 1 -type f -size 0 -print -quit | \
  grep -q . && { echo "One or more WHAM checkpoints are empty" >&2; exit 3; }

sha256sum "${ASSETS}"/checkpoints/* "${MHR_ROOT}/assets/mhr_model.pt" \
  > "${ASSETS}/SHA256SUMS"

echo "PASS: public WHAM checkpoints, auxiliary files, and full MHR assets downloaded"
echo "ACTION REQUIRED: place licensed SMPL_NEUTRAL.pkl at:"
echo "  ${ASSETS}/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"
