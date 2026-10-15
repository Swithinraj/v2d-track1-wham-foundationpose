#!/bin/bash
# MHR hand points (2 x 256 per frame, drift-corrected human) for one episode -> outputs/hands/episode_X.npz.
# Used by the symmetric-object tracker's contact prior and the optional penetration step.
# Usage: hands.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
require_file "${TRACK1_ROOT}/outputs/human/${TAG}/mhr_human.npz" "run pipeline/stages/wham_to_mhr.sh ${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/alignment/${TAG}/drift.npz" "run pipeline/stages/align.sh ${EPISODE}"

in_container mhr_converter.sif python /track1/scripts/extract_mhr_hands.py --episodes "${EPISODE}"
require_file "${TRACK1_ROOT}/outputs/hands/${TAG}.npz"
