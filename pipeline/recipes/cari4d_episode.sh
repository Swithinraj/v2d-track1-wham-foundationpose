#!/bin/bash
# One episode through NVIDIA's official CARI4D MHR pipeline (see cari4d.md). Reruns reuse finished stages.
# Usage: cari4d_episode.sh EPISODE GPU [EXTRA run_inference ARGS]
#   CARI4D_OUT (default work/cari4d/out), CARI4D_SIZES (default outputs/floor/object_sizes.json), CARI4D_BASE: the scored
#   episode directory whose mesh and scale back up an input mesh of the wrong shape.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
EPISODE="$1"; GPU="$2"; shift 2
TAG="$(episode_tag "${EPISODE}")"; I="work/cari4d/inputs/${TAG}"; OUT="${CARI4D_OUT:-work/cari4d/out}"
SIZES="${CARI4D_SIZES:-outputs/floor/object_sizes.json}"; BASE="${CARI4D_BASE:-submissions/episodes_cand_Mtp_in}"
mkdir -p "${I}" "${OUT}"
# A hard link: the pipeline resolves symlinks and then insists on this file name.
ln -f "data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" "${I}/${TAG}.0.color.mp4"
read -r FRAMES OBJECT SCALE < <("${VENV_PYTHON}" - "${EPISODE}" "${BASE}/${TAG}.npz" 2>/dev/null <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from object_true_size import object_kind
episode = int(sys.argv[1])
meta = {json.loads(line)["episode_index"]: json.loads(line) for line in open("data/track_1/meta/episodes.jsonl")}
print(meta[episode]["length"], object_kind(meta[episode]["tasks"][0]), float(np.load(sys.argv[2])["object_scale"]))
PY
)
[[ -n "${FRAMES:-}" ]] || { echo "FAIL: no metadata for episode ${EPISODE}" >&2; exit 2; }
[[ -f "${I}/object.glb" ]] || "${VENV_PYTHON}" scripts/cari4d_mesh.py --sizes "${SIZES}" --object "${OBJECT}" \
  --scored_mesh "${BASE}/${TAG}_object.glb" --scored_scale "${SCALE}" --out "${I}/object.glb"
[[ -f "${I}/masks_k0.h5" ]] || in_container v2d_cari4d.sif python -m v2d.cari4d.lib.pack_masks \
  --video_path "/track1/${I}/${TAG}.0.color.mp4" --human_masks_path "/track1/outputs/person/${TAG}/sam2_masks/0" \
  --object_masks_path "/track1/outputs/object/${TAG}/sam2_masks/1" --output_path "/track1/${I}/masks_k0.h5"
# The image was built without git-lfs, so these two assets are only pointers inside it. They come from the same upstream
# commit, verified by the pointers' sha256 (pipeline/setup/fetch_cari4d_assets.sh).
export APPTAINER_BIND="${TRACK1_ROOT}/weights/cari4d/lib_mhr_assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets:ro"
if [[ -n "${CARI4D_EXPORTER_PATCH:-}" ]]; then
  require_file "${CARI4D_EXPORTER_PATCH}" "generate with scripts/cari4d_runtime_patch.py from this container's exporter"
  [[ "${CARI4D_EXPORTER_PATCH}" == "${TRACK1_ROOT}/"* ]] || { echo "FAIL: exporter patch must be inside TRACK1_ROOT" >&2; exit 2; }
  export APPTAINER_BIND="${APPTAINER_BIND},${CARI4D_EXPORTER_PATCH}:/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:ro"
fi
SYMMETRIC=""; [[ "${OBJECT}" == hula_hoop ]] && SYMMETRIC="--symmetric_object"
in_container v2d_cari4d.sif CUDA_VISIBLE_DEVICES="${GPU}" python -m v2d.cari4d.lib.run_inference \
  --video_path "/track1/${I}/${TAG}.0.color.mp4" --mask_h5_path "/track1/${I}/masks_k0.h5" \
  --object_mesh_path "/track1/${I}/object.glb" --weights_path /track1/weights/cari4d --output_dir "/track1/${OUT}" \
  --expected_frames "${FRAMES}" --skip_weight_download ${SYMMETRIC} "$@"
echo "PASS: CARI4D ${TAG} -> ${OUT}/${TAG}"
