#!/bin/bash
# WHAM world -> MoGe-2 camera alignment for one episode (static-camera PnP, WHAM drift correction,
# MoGe/SMPL depth scale) -> outputs/alignment/episode_X/{alignment.json,drift.npz}. CPU only.
# Usage: align.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
require_file "${TRACK1_ROOT}/outputs/wham/${TAG}/wham_smpl_world.npz" "run pipeline/stages/wham.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/object/${TAG}/intrinsics/000000.json" "run pipeline/stages/object_depth.sh ${EPISODE}"

apptainer exec --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" "${CONTAINERS}/wham.sif" \
  env HOME=/tmp python /track1/scripts/align_human_camera.py --episode "${EPISODE}"
