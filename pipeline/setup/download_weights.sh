#!/bin/bash
# Download every public, ungated checkpoint the pipeline uses.
#   weights/mhr      Meta MHR v1.0.1 assets (Apache-2.0), checksum-verified mhr_model.pt
#   weights/sam2     SAM 2.1 Hiera-L
#   weights/object   MoGe-2, FoundationPose, GroundingDINO
# The gated models have their own scripts: download_sam3d_body.sh (human) and download_sam3d_objects.sh (meshes).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

for image in v2d_cari4d.sif v2d_sam2.sif v2d_grounding_dino.sif v2d_sam3d.sif; do
  require_file "${CONTAINERS}/${image}" "run pipeline/setup/build_containers.sh"
done
W="${TRACK1_ROOT}/weights"
mkdir -p "${W}/mhr" "${W}/sam2" "${W}/object/moge2" "${W}/object/foundationpose" "${W}/object/grounding_dino" \
  "${TRACK1_ROOT}/.cache/huggingface_object"

# --- Meta MHR assets ---
if [[ ! -s "${W}/mhr/assets.zip" ]]; then
  curl -fsSL "https://github.com/facebookresearch/MHR/releases/download/${MHR_ASSETS_RELEASE}/assets.zip" \
    -o "${W}/mhr/assets.zip"
fi
unzip -q -o "${W}/mhr/assets.zip" -d "${W}/mhr"
echo "${MHR_MODEL_SHA256}  ${W}/mhr/assets/mhr_model.pt" | sha256sum -c -

# --- CoTracker3 (roller handle masks; CC BY-NC 4.0) ---
mkdir -p "${W}/cotracker3"
[[ -s "${W}/cotracker3/scaled_offline.pth" ]] || curl -fsSL "${COTRACKER_WEIGHTS_URL}" -o "${W}/cotracker3/scaled_offline.pth"
echo "${COTRACKER_WEIGHTS_SHA256}  ${W}/cotracker3/scaled_offline.pth" | sha256sum -c -

# --- Object branch (the toolkit's own downloaders) ---
HF=(HF_HOME=/track1/.cache/huggingface_object XDG_CACHE_HOME=/tmp/xdg HF_HUB_DISABLE_XET=1)
in_container v2d_sam2.sif "${HF[@]}" python -m v2d.sam2.lib.download_weights --output_dir /track1/weights/sam2
in_container v2d_cari4d.sif "${HF[@]}" python -m v2d.moge.lib.download_weights --output_dir /track1/weights/object/moge2
in_container v2d_cari4d.sif "${HF[@]}" python -m v2d.foundation_pose.lib.download_weights \
  --output_dir /track1/weights/object/foundationpose --backend nvlabs_pytorch
in_container v2d_grounding_dino.sif "${HF[@]}" python -m v2d.grounding_dino.lib.download_weights \
  --output_dir /track1/weights/object/grounding_dino

for path in sam2/sam2.1_hiera_large.pt object/moge2/model.pt object/grounding_dino/groundingdino_swint_ogc.pth \
    object/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth \
    object/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth; do
  require_file "${W}/${path}"
done
echo "PASS: public weights downloaded"
