#!/bin/bash
# SAM 3D Body human for one episode (GPU) -> outputs/sam3d_body/episode_X/:
#   1. scripts/sam3d_body_video.py     per-frame SAM 3D Body (body + hand decoders) on the selected actor: box from
#                                      the actor's ViTPose keypoints, full-resolution frames, one intrinsics estimate
#                                      per episode; raw outputs kept unchanged (raw_<episode>.npz)
#   2. scripts/sam3d_body_identity.py  one shared identity (medoid scales + shape over reliable frames) and the
#                                      MoGe/SAM depth-scale series
#   3. scripts/sam3d_body_refit.py     every frame's pose refitted with that identity (kit MHR wrapper, lm_pose)
# Needs outputs/vitpose and outputs/alignment (intrinsics) of the episode and weights/sam3d_body
# (pipeline/setup/download_sam3d_body.sh). SAM3D_BODY_IMAGE selects the image with the SAM 3D Body library
# (default v2d_cari4d.sif, which ships it; v2d_sam3d_body.sif from config/sam3d_body.def works as well).
# SAM3D_BODY_TORCH_HOME overrides the (writable) torch hub cache. A complete raw output is reused unless REDO=1.
# Usage: sam3d_body.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/sam3d_body/${TAG}"
W="${TRACK1_ROOT}/weights/sam3d_body"
require_file "${TRACK1_ROOT}/outputs/vitpose/${TAG}.npz" "run pipeline/stages/keypoints2d.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/alignment/${TAG}/alignment.json" "run pipeline/stages/align.sh ${EPISODE}"
require_file "${W}/sam-3d-body-dinov3/model.ckpt" "run pipeline/setup/download_sam3d_body.sh"
require_file "${W}/sam-3d-body-dinov3/assets/mhr_model.pt" "run pipeline/setup/download_sam3d_body.sh"
mkdir -p "${OUT}"
FRAMES="$(episode_field "${EPISODE}" length)"
done_frames() { "${HOST_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('frames', 0))" "${OUT}/summary.json" 2>/dev/null || echo 0; }
if [[ "${REDO:-0}" == 1 || ! -s "${OUT}/raw_${TAG}.npz" || "$(done_frames)" != "${FRAMES}" ]]; then
  in_container "${SAM3D_BODY_IMAGE:-v2d_cari4d.sif}" TORCH_HOME="${SAM3D_BODY_TORCH_HOME:-/track1/weights/sam3d_body/torch_home}" \
    python -u /track1/scripts/sam3d_body_video.py --episodes "${EPISODE}"
fi
in_container wham.sif python /track1/scripts/sam3d_body_identity.py --episodes "${EPISODE}"
in_container mhr_converter.sif python /track1/scripts/sam3d_body_refit.py --episodes "${EPISODE}"
require_file "${OUT}/refit.npz"
echo "PASS: SAM 3D Body ${TAG}"
