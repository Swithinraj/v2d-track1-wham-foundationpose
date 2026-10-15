#!/bin/bash
# Symmetry-aware track for one episode whose object is listed in config/symmetric_objects.json (the hula
# hoop); a no-op for every other object. Builds the exactly round <object>.symmetric.glb once, then runs
# scripts/track_symmetric_object.py -> outputs/object/episode_X/foundationpose/symmetric/. CPU only.
# Needs the episode's masks, FoundationPose smoothed track, alignment and hand points.
# Usage: track_symmetric.sh EPISODE   (NO_HANDS=1 disables the hand-contact prior)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OBJECT="$(episode_field "${EPISODE}" object)"
if ! "${HOST_PYTHON}" -c "import json, sys; sys.exit(0 if '${OBJECT}' in json.load(open('${TRACK1_ROOT}/config/symmetric_objects.json')) else 1)"; then
  echo "SKIP: ${OBJECT} is not a symmetric object"
  exit 0
fi
require_file "${TRACK1_ROOT}/outputs/object/${TAG}/foundationpose/smoothed/000000.json" "run pipeline/stages/track_object.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/hands/${TAG}.npz" "run pipeline/stages/hands.sh ${EPISODE}"

if [[ ! -s "${TRACK1_ROOT}/inputs/object_meshes/${OBJECT}.symmetric.glb" ]]; then
  in_container v2d_sam3d.sif python /track1/scripts/make_round_mesh.py --object "${OBJECT}"
fi
hands_flag=()
[[ "${NO_HANDS:-0}" == 1 ]] && hands_flag=(--no_hands)
apptainer exec --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" "${CONTAINERS}/wham.sif" \
  env HOME=/tmp python -u /track1/scripts/track_symmetric_object.py --episode "${EPISODE}" "${hands_flag[@]}"
