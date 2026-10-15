#!/bin/bash
#SBATCH --job-name=v2d_t1_trellis_weights
#SBATCH --partition=hpc3,hpc1,hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/trellis_weights_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/trellis_weights_%j.err

# Public, ungated, MIT-licensed TRELLIS image-to-3D weights plus the DINOv2
# conditioning encoder that TRELLIS loads through torch.hub. Both are cached
# inside the workspace so mesh generation jobs need no network access.
set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
TRELLIS_WEIGHTS="${ROOT}/weights/trellis"
TRELLIS_REVISION=25e0d31ffbebe4b5a97464dd851910efc3002d96
IMAGE="${ROOT}/containers/v2d_sam3d.sif"
cd "${ROOT}"
mkdir -p "${TRELLIS_WEIGHTS}/TRELLIS-image-large" "${TRELLIS_WEIGHTS}/torch_home" \
  "${ROOT}/.cache/huggingface_object" "${ROOT}/.cache/xdg"
[[ -s "${IMAGE}" ]] || { echo "Missing ${IMAGE}" >&2; exit 2; }

apptainer exec --no-home --cleanenv \
  --env HF_HOME=/track1/.cache/huggingface_object \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --env HF_HUB_DISABLE_XET=1 \
  --bind "${ROOT}:/track1" "${IMAGE}" \
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/TRELLIS-image-large', revision='${TRELLIS_REVISION}',
                  local_dir='/track1/weights/trellis/TRELLIS-image-large')
"

apptainer exec --no-home --cleanenv \
  --env TORCH_HOME=/track1/weights/trellis/torch_home \
  --env XDG_CACHE_HOME=/track1/.cache/xdg \
  --bind "${ROOT}:/track1" "${IMAGE}" \
  python -c "
import torch
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
print('dinov2_vitl14_reg params', sum(p.numel() for p in m.parameters()))
"

printf '%s\n' "${TRELLIS_REVISION}" > "${TRELLIS_WEIGHTS}/TRELLIS-image-large/REVISION"
for name in ss_dec_conv3d_16l8_fp16 ss_flow_img_dit_L_16l8_fp16 \
    slat_flow_img_dit_L_64l8p2_fp16 slat_dec_mesh_swin8_B_64l8m256c_fp16 \
    slat_dec_gs_swin8_B_64l8gs32_fp16 slat_dec_rf_swin8_B_64l8r16_fp16; do
  test -s "${TRELLIS_WEIGHTS}/TRELLIS-image-large/ckpts/${name}.safetensors"
done
test -s "${TRELLIS_WEIGHTS}/TRELLIS-image-large/pipeline.json"
test -n "$(find "${TRELLIS_WEIGHTS}/torch_home/hub/checkpoints" -name 'dinov2_vitl14_reg4_pretrain.pth' -size +100M -print -quit)"
echo "PASS: TRELLIS-image-large @ ${TRELLIS_REVISION} and DINOv2 ViT-L/14-reg cached"
