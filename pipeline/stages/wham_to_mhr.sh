#!/bin/bash
# Fit Track 1 MHR parameters to one episode's WHAM SMPL sequence -> outputs/human/episode_X/mhr_human.npz.
# Uses Meta's SMPL->MHR topology map and the official kit's mesh_to_mhr_params.py. A validated
# output is reused. Usage: wham_to_mhr.sh EPISODE   (MHR_MAP_BATCH_SIZE 64, MHR_MODEL_BATCH_SIZE 256)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
"${HOST_PYTHON}" "${TRACK1_ROOT}/scripts/validate_mhr_setup.py"
require_file "${TRACK1_ROOT}/outputs/wham/${TAG}/wham_smpl_world.npz" "run pipeline/stages/wham.sh ${EPISODE}"

CHECK="
import numpy as np
z = np.load('/track1/outputs/human/${TAG}/mhr_human.npz')
n = np.load('/track1/outputs/wham/${TAG}/wham_smpl_world.npz')['vertices'].shape[0]
assert z['pose'].shape == (n, 136) and z['scales'].shape == (68,) and z['shape'].shape == (45,)
assert all(np.isfinite(z[k]).all() for k in ('pose', 'scales', 'shape'))"
if [[ -s "${TRACK1_ROOT}/outputs/human/${TAG}/mhr_human.npz" ]] && in_container mhr_converter.sif python -c "${CHECK}" 2>/dev/null; then
  echo "PASS: reusing validated WHAM -> MHR output for ${TAG}"
  exit 0
fi
mkdir -p "${TRACK1_ROOT}/outputs/human/${TAG}"

in_container mhr_converter.sif python /track1/scripts/convert_wham_to_mhr.py \
  --input "/track1/outputs/wham/${TAG}/wham_smpl_world.npz" \
  --output "/track1/outputs/human/${TAG}/mhr_human.npz" \
  --smpl-faces /track1/weights/wham/dataset/body_models/smpl_faces.npy \
  --mapping /track1/third_party/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz \
  --mhr-model /track1/weights/mhr/assets/mhr_model.pt \
  --official-converter /track1/official/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py \
  --batch-size "${MHR_MAP_BATCH_SIZE:-64}" \
  --model-batch "${MHR_MODEL_BATCH_SIZE:-256}"
in_container mhr_converter.sif python -c "${CHECK}"
echo "PASS: WHAM -> MHR human parameters ${TAG}"
