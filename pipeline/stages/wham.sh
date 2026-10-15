#!/bin/bash
# WHAM world-frame human motion for one episode -> outputs/wham/episode_X/{wham_output.pkl,wham_smpl_world.npz}.
# A validated dense SMPL export is reused. Usage: wham.sh EPISODE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
FRAMES="$(episode_field "${EPISODE}" length)"
OUT="${TRACK1_ROOT}/outputs/wham/${TAG}"
RUNTIME="${TRACK1_ROOT}/.cache/wham_runtime"
TRACK_OVERRIDES="${TRACK1_ROOT}/config/human_track_overrides.json"
EXPECTED_SUBJECTS="$("${HOST_PYTHON}" -c '
import json, sys
data = json.load(open(sys.argv[1]))
print(json.dumps(data.get("episodes", {}).get(str(int(sys.argv[2])), {}).get("subjects", [])))
' "${TRACK_OVERRIDES}" "${EPISODE}")"
"${HOST_PYTHON}" "${TRACK1_ROOT}/scripts/validate_wham_setup.py"
require_file "${VIDEOS}/${TAG}.mp4"
for module in cuda_corr cuda_ba lietorch_backends; do
  [[ -n "$(find "${TRACK1_ROOT}/.cache/dpvo_extensions/lib" -maxdepth 1 -name "${module}*.so" -print -quit 2>/dev/null)" ]] || {
    echo "Missing DPVO extension ${module}; run pipeline/setup/build_dpvo.sh" >&2
    exit 6
  }
done

wham_exec() {  # WHAM resolves configs/, checkpoints/ and dataset/ relative to its runtime directory
  apptainer exec --nv --no-home --cleanenv --bind "${TRACK1_ROOT}:/track1" --pwd /track1/.cache/wham_runtime \
    "${CONTAINERS}/wham.sif" \
    env HOME=/track1/.cache/wham_runtime/home MPLCONFIGDIR=/track1/.cache/wham_runtime/matplotlib \
        PYTHONPATH=/track1/.cache/dpvo_extensions/lib:/track1/third_party/WHAM "$@"
}

if [[ -s "${OUT}/wham_smpl_world.npz" ]] && wham_exec python -c "
import json, numpy as np
z = np.load('/track1/outputs/wham/${TAG}/wham_smpl_world.npz')
v = z['vertices']; r = json.loads(str(z['report']))
expected = json.loads('${EXPECTED_SUBJECTS}')
selected = [int(x) for x in r.get('subjects', [r['subject']])]
assert v.shape == (${FRAMES}, 6890, 3) and np.isfinite(v).all()
assert not expected or (selected == expected and r.get('fragment_stitching') == 'translation_continuity')" 2>/dev/null; then
  echo "PASS: reusing validated WHAM mesh for ${TAG} (${FRAMES} frames)"
  exit 0
fi

mkdir -p "${TRACK1_ROOT}/outputs/wham" "${RUNTIME}/home" "${RUNTIME}/matplotlib"
# Relative links resolve both on the host and under /track1 inside the container.
ln -sfn ../../third_party/WHAM/configs "${RUNTIME}/configs"
ln -sfn ../../weights/wham/checkpoints "${RUNTIME}/checkpoints"
ln -sfn ../../weights/wham/dataset "${RUNTIME}/dataset"
# DetectionModel derives its ViTPose/YOLO paths from the WHAM source tree itself.
ln -sfn ../../weights/wham/checkpoints "${TRACK1_ROOT}/third_party/WHAM/checkpoints"
ln -sfn ../../weights/wham/dataset "${TRACK1_ROOT}/third_party/WHAM/dataset"

if [[ ! -s "${OUT}/wham_output.pkl" ]]; then
  wham_exec python /track1/third_party/WHAM/demo.py \
    --video "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
    --output_pth /track1/outputs/wham --save_pkl
else
  echo "Reusing WHAM inference output ${OUT}/wham_output.pkl; rebuilding the selected dense export"
fi
require_file "${OUT}/wham_output.pkl" "WHAM did not produce output"
mapfile -t SUBJECTS < <("${HOST_PYTHON}" -c 'import json,sys; [print(x) for x in json.loads(sys.argv[1])]' "${EXPECTED_SUBJECTS}")
SUBJECT_ARGS=()
if (( ${#SUBJECTS[@]} )); then
  SUBJECT_ARGS=(--subjects "${SUBJECTS[@]}")
fi
wham_exec python /track1/scripts/export_wham_world_mesh.py \
  --input "/track1/outputs/wham/${TAG}/wham_output.pkl" \
  --output "/track1/outputs/wham/${TAG}/wham_smpl_world.npz" \
  --frames "${FRAMES}" "${SUBJECT_ARGS[@]}"
require_file "${OUT}/wham_smpl_world.npz"
echo "PASS: WHAM ${TAG}, dense ${FRAMES}-frame SMPL mesh"
