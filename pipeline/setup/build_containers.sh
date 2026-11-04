#!/bin/bash
# Build the Apptainer images from config/*.def (all of them, or the names given):
#   cari4d          MoGe-2 depth, FoundationPose tracking/smoothing/scale search, CARI4D runtime
#   sam2            SAM 2 video masks
#   grounding_dino  GroundingDINO object and person prompts
#   sam3d           SAM 3D Objects mesh generation (gated weights: download_sam3d_objects.sh)
#   mhr_converter   MHR model: refits, hand points, joints (the kit's MHR wrapper)
#   sam3d_body      SAM 3D Body (optional dedicated image; v2d_cari4d.sif ships the same library)
# Needs `apptainer build --fakeroot` (or root) and network access. The CUDA extensions in the
# cari4d, sam3d and grounding_dino images target compute capability 8.0+ (A100 and newer).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

declare -A IMAGE=(
  [cari4d]=v2d_cari4d.sif [sam2]=v2d_sam2.sif [grounding_dino]=v2d_grounding_dino.sif
  [sam3d]=v2d_sam3d.sif [mhr_converter]=mhr_converter.sif [sam3d_body]=v2d_sam3d_body.sif
)
names=("$@")
((${#names[@]})) || names=(cari4d sam2 grounding_dino sam3d mhr_converter)

cd "${TRACK1_ROOT}"
for name in "${names[@]}"; do
  [[ -n "${IMAGE[${name}]:-}" ]] || { echo "Unknown image ${name}" >&2; exit 2; }
  apptainer build --fakeroot --force "${CONTAINERS}/${IMAGE[${name}]}" "config/${name}.def"
  in_container "${IMAGE[${name}]}" python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
  echo "PASS: built containers/${IMAGE[${name}]}"
done
