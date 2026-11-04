#!/bin/bash
# Generated metric mesh for one object -> inputs/object_meshes/<object>.glb (+ .provenance.json).
#   1. scripts/sam3d_object_mesh.py   SAM 3D Objects candidates (each episode's cleanest view x SEEDS), grounded in that
#                                     frame's MoGe-2 depth, metric directly -> outputs/object_meshes/<object>/sam3d/
#   2. scripts/select_object_mesh.py  FoundationPose fit in all three cameras picks the candidate and refines its scale
#                                     within 0.75-1.33x; an already installed mesh is scored as the reference only
# Needs all three episodes' depth and masks, and weights/sam3d_objects (pipeline/setup/download_sam3d_objects.sh).
# Usage: object_mesh.sh OBJECT   SEEDS="1 2"; SKIP_GENERATION=1 only re-runs the selection on existing candidates;
#   SKIP_SELECTION=1 only generates.
# An image built before config/sam3d.def installed libusb (open3d needs it): copy libusb-1.0.so.0 into .cache/lib/
# and set SAM3D_LIB_DIR=/track1/.cache/lib (prepended to the image's library path).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
OBJECT="${1:-}"
all_objects | grep -qx "${OBJECT}" || { echo "Object must be one of: $(all_objects | tr '\n' ' ')" >&2; exit 2; }
if [[ "${SKIP_GENERATION:-0}" != 1 ]]; then
  W=/track1/weights/sam3d_objects
  require_file "${TRACK1_ROOT}/weights/sam3d_objects/hf-download/checkpoints/pipeline.yaml" "run pipeline/setup/download_sam3d_objects.sh"
  # torch.hub writes into TORCH_HOME (trusted_list): a writable cache that links the checkpoints and the pinned code
  T="${TRACK1_ROOT}/.cache/sam3d_torch_home"; mkdir -p "${T}/hub/checkpoints"
  for ckpt in dinov2_vitl14_reg4_pretrain.pth dinov2_vitb14_reg4_pretrain.pth; do
    ln -sfn "${W}/torch_home/hub/checkpoints/${ckpt}" "${T}/hub/checkpoints/${ckpt}"
  done
  rm -rf "${T}/hub/facebookresearch_dinov2_main"
  ln -sfn "${W}/torch_home/hub/facebookresearch_dinov2_main" "${T}/hub/facebookresearch_dinov2_main"
  in_container v2d_sam3d.sif TORCH_HOME=/track1/.cache/sam3d_torch_home HF_HOME="${W}/hf_home" HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH="${SAM3D_LIB_DIR:+${SAM3D_LIB_DIR}:}/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs" \
    python -u /track1/scripts/sam3d_object_mesh.py --object "${OBJECT}" --seeds ${SEEDS:-1 2}
fi
[[ "${SKIP_SELECTION:-0}" == 1 ]] && exit 0
reference=()
[[ -s "${TRACK1_ROOT}/inputs/object_meshes/${OBJECT}.glb" ]] && reference=(--reference "/track1/inputs/object_meshes/${OBJECT}.glb")
in_container v2d_cari4d.sif python -u /track1/scripts/select_object_mesh.py --object "${OBJECT}" ${reference[@]+"${reference[@]}"}
