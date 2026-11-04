#!/bin/bash
# SAM 3D Body human for one episode (GPU) -> outputs/sam3d_body/episode_X/ (+ camera, hands, alignment_sam, keypoints):
#   0. scripts/camera_intrinsics.py    one static camera: median MoGe-2 intrinsics -> outputs/camera/<ep>/intrinsics.json
#   1. scripts/sam3d_body_video.py     per-frame SAM 3D Body (body + hand decoders) on the actor, prompted with the box of
#                                      its SAM 2 person mask (person.sh); raw outputs kept unchanged (raw_<episode>.npz)
#   2. scripts/sam3d_body_identity.py  one shared identity (medoid scales + shape over reliable frames), the MoGe/SAM
#                                      depth-scale series, SAM's 2D body keypoints (outputs/keypoints2d_sam)
#   3. scripts/sam3d_body_refit.py     every frame's pose refitted with that identity (kit MHR wrapper, lm_pose)
#   4. scripts/sam3d_body_hands.py     hand points (outputs/hands) + camera alignment (outputs/alignment_sam)
# Needs the episode's depth (object_depth.sh) and actor masks (person.sh), and weights/sam3d_body
# (pipeline/setup/download_sam3d_body.sh). SAM3D_BODY_IMAGE selects the image with the SAM 3D Body library
# (default v2d_cari4d.sif, which ships it; v2d_sam3d_body.sif from config/sam3d_body.def works as well).
# SAM3D_BODY_ARGS adds options to step 1, e.g. "--inference_type body --mask_prompt" (pipeline/recipes/bmq_final.sh).
# SAM3D_BODY_TORCH_HOME overrides the (writable) torch hub cache. A complete raw output is reused unless REDO=1.
# Usage: sam3d_body.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/sam3d_body/${TAG}"
W="${TRACK1_ROOT}/weights/sam3d_body"
FRAMES="$(episode_field "${EPISODE}" length)"
require_file "${TRACK1_ROOT}/outputs/person/${TAG}/sam2_masks/0/$(printf '%06d' $((FRAMES - 1))).png" "run pipeline/stages/person.sh ${EPISODE}"
require_file "${W}/sam-3d-body-dinov3/model.ckpt" "run pipeline/setup/download_sam3d_body.sh"
require_file "${W}/sam-3d-body-dinov3/assets/mhr_model.pt" "run pipeline/setup/download_sam3d_body.sh"
mkdir -p "${OUT}"
"${HOST_PYTHON}" "${TRACK1_ROOT}/scripts/camera_intrinsics.py" "${EPISODE}"
done_frames() { "${HOST_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('frames', 0))" "${OUT}/summary.json" 2>/dev/null || echo 0; }
if [[ "${REDO:-0}" == 1 || ! -s "${OUT}/raw_${TAG}.npz" || "$(done_frames)" != "${FRAMES}" ]]; then
  in_container "${SAM3D_BODY_IMAGE:-v2d_cari4d.sif}" TORCH_HOME="${SAM3D_BODY_TORCH_HOME:-/track1/weights/sam3d_body/torch_home}" \
    python -u /track1/scripts/sam3d_body_video.py --episodes "${EPISODE}" ${SAM3D_BODY_ARGS:-}
fi
in_container v2d_cari4d.sif python /track1/scripts/sam3d_body_identity.py --episodes "${EPISODE}"
in_container mhr_converter.sif python /track1/scripts/sam3d_body_refit.py --episodes "${EPISODE}"
in_container mhr_converter.sif python /track1/scripts/sam3d_body_hands.py --episodes "${EPISODE}"
require_file "${OUT}/refit.npz"
require_file "${TRACK1_ROOT}/outputs/hands/${TAG}.npz"
echo "PASS: SAM 3D Body ${TAG}"
