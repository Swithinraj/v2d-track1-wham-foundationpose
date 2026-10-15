#!/bin/bash
# MoGe-2 depth-trend series of the human for one episode -> outputs/depth_trend/episode_X.npz (+ .json).
# pack.sh uses it to remove the human's slow depth drift (scripts/depth_trend_correction.py apply).
# Usage: depth_trend.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
require_file "${TRACK1_ROOT}/outputs/wham/${TAG}/wham_smpl_world.npz" "run pipeline/stages/wham.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/alignment/${TAG}/drift.npz" "run pipeline/stages/align.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/object/${TAG}/mask_audit.json" "run pipeline/stages/object_masks.sh ${EPISODE}"

in_container wham.sif python /track1/scripts/depth_trend_correction.py series "${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/depth_trend/${TAG}.npz"
