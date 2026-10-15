#!/bin/bash
# Download every public, ungated checkpoint the pipeline uses, then derive the SMPL face array.
#   weights/wham     WHAM, HMR2a, DPVO, YOLOv8x, ViTPose-H + WHAM body-model regressors
#   weights/mhr      Meta MHR v1.0.1 assets (Apache-2.0), checksum-verified mhr_model.pt
#   weights/sam2     SAM 2.1 Hiera-L
#   weights/object   MoGe-2, FoundationPose, GroundingDINO
#   weights/trellis  TRELLIS-image-large (MIT) + DINOv2 ViT-L/14-reg torch.hub cache
# SMPL_NEUTRAL.pkl is licensed separately and cannot be downloaded here: register at
# https://smpl.is.tue.mpg.de, then place it at weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

for image in wham.sif v2d_cari4d.sif v2d_sam2.sif v2d_grounding_dino.sif v2d_sam3d.sif; do
  require_file "${CONTAINERS}/${image}" "run pipeline/setup/build_containers.sh"
done
W="${TRACK1_ROOT}/weights"
mkdir -p "${W}/wham/checkpoints" "${W}/wham/dataset" "${W}/wham/.downloads" "${W}/mhr" "${W}/sam2" \
  "${W}/object/moge2" "${W}/object/foundationpose" "${W}/object/grounding_dino" \
  "${W}/trellis/TRELLIS-image-large" "${W}/trellis/torch_home" "${TRACK1_ROOT}/.cache/huggingface_object"

# --- WHAM public checkpoints (Google Drive ids from WHAM's fetch_demo_data.sh) ---
gdrive() {  # gdrive FILE_ID PATH_UNDER_weights/wham
  [[ -s "${W}/wham/$2" ]] && { echo "SKIP existing wham/$2"; return; }
  in_container wham.sif gdown --no-cookies "https://drive.google.com/uc?id=$1" -O "/track1/weights/wham/$2"
}
gdrive 19qkI-a6xuwob9_RFNSPWf1yWErwVVlks checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
gdrive 1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y checkpoints/hmr2a.ckpt
gdrive 1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT checkpoints/dpvo.pth
gdrive 1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo checkpoints/yolov8x.pt
gdrive 1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso checkpoints/vitpose-h-multi-coco.pth
gdrive 1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ .downloads/body_models.tar.gz
tar -xzf "${W}/wham/.downloads/body_models.tar.gz" -C "${W}/wham/dataset"

# --- Meta MHR assets ---
if [[ ! -s "${W}/mhr/assets.zip" ]]; then
  curl -fsSL "https://github.com/facebookresearch/MHR/releases/download/${MHR_ASSETS_RELEASE}/assets.zip" \
    -o "${W}/mhr/assets.zip"
fi
unzip -q -o "${W}/mhr/assets.zip" -d "${W}/mhr"
echo "${MHR_MODEL_SHA256}  ${W}/mhr/assets/mhr_model.pt" | sha256sum -c -

# --- Object branch (the toolkit's own downloaders) ---
HF=(HF_HOME=/track1/.cache/huggingface_object XDG_CACHE_HOME=/tmp/xdg HF_HUB_DISABLE_XET=1)
in_container v2d_sam2.sif "${HF[@]}" python -m v2d.sam2.lib.download_weights --output_dir /track1/weights/sam2
in_container v2d_cari4d.sif "${HF[@]}" python -m v2d.moge.lib.download_weights --output_dir /track1/weights/object/moge2
in_container v2d_cari4d.sif "${HF[@]}" python -m v2d.foundation_pose.lib.download_weights \
  --output_dir /track1/weights/object/foundationpose --backend nvlabs_pytorch
in_container v2d_grounding_dino.sif "${HF[@]}" python -m v2d.grounding_dino.lib.download_weights \
  --output_dir /track1/weights/object/grounding_dino

# --- TRELLIS + DINOv2 (cached so mesh generation runs offline) ---
in_container v2d_sam3d.sif "${HF[@]}" python -c "
from huggingface_hub import snapshot_download
snapshot_download('${TRELLIS_WEIGHTS_REPO}', revision='${TRELLIS_WEIGHTS_REVISION}',
                  local_dir='/track1/weights/trellis/TRELLIS-image-large')
"
printf '%s\n' "${TRELLIS_WEIGHTS_REVISION}" > "${W}/trellis/TRELLIS-image-large/REVISION"
in_container v2d_sam3d.sif TORCH_HOME=/track1/weights/trellis/torch_home python -c "
import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)"

for path in sam2/sam2.1_hiera_large.pt object/moge2/model.pt object/grounding_dino/groundingdino_swint_ogc.pth \
    object/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth \
    object/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth \
    trellis/TRELLIS-image-large/pipeline.json; do
  require_file "${W}/${path}"
done
echo "PASS: public weights downloaded"

# --- Licensed SMPL model: derive the triangle array the MHR fitter needs ---
SMPL="${W}/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"
if [[ -s "${SMPL}" ]]; then
  [[ -s "${W}/wham/dataset/body_models/smpl_faces.npy" ]] || \
    in_container wham.sif python /track1/scripts/prepare_smpl_faces.py \
      --model /track1/weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl \
      --output /track1/weights/wham/dataset/body_models/smpl_faces.npy
  echo "PASS: SMPL faces derived"
else
  echo "ACTION REQUIRED: place the licensed SMPL_NEUTRAL.pkl at ${SMPL#"${TRACK1_ROOT}"/} and re-run this script."
fi
