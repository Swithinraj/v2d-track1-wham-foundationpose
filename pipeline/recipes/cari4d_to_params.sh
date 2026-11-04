#!/bin/bash
# One episode's CARI4D stage to scored MHR parameters: export the stage, smooth on the mesh, run the kit's converter.
# Usage: cari4d_to_params.sh EPISODE STAGE LAM GPU  -> work/cari4d/params/STAGE_lLAM/episode_*.npz
#   CARI4D_OUT (default work/cari4d/out). Reruns skip finished files.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
EPISODE="$1"; STAGE="$2"; LAM="$3"; GPU="$4"; TAG="$(episode_tag "${EPISODE}")"; OUT="${CARI4D_OUT:-work/cari4d/out}"
X="${CARI4D_EXPORT:-work/cari4d/export}/${STAGE}"; M="${CARI4D_EXPORT:-work/cari4d/export}/${STAGE}_l${LAM}"; D="${CARI4D_PARAMS:-work/cari4d/params}/${STAGE}_l${LAM}"
mkdir -p "${M}" "${D}"
L=/workspace/v2d_cari4d/lib/cari4d
if [[ ! -f "${X}/${TAG}.verts.npy" ]]; then
  APPTAINER_BIND="${TRACK1_ROOT}/weights/cari4d/lib_mhr_assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets:ro" \
  in_container v2d_cari4d.sif CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${L}:/workspace/v2d_sam3d_body/lib" MHR_ASSETS_ROOT=/track1/weights/cari4d/sam3d_body \
    python /track1/scripts/cari4d_export.py --root "/track1/${OUT}" --episode "${EPISODE}" --stage "${STAGE}" --out "/track1/${X}" | tail -1
fi
[[ -f "${M}/${TAG}.verts.npy" ]] || "${VENV_PYTHON}" scripts/mesh_smooth.py --verts "${X}/${TAG}.verts.npy" --episode "${EPISODE}" --lam "${LAM}" --out "${M}/${TAG}.verts.npy"
cp -u "${X}/${TAG}.object.npz" "${X}/${TAG}_object.glb" "${M}/"
[[ -f "${D}/${TAG}.npz" ]] || in_container mhr_converter.sif CUDA_VISIBLE_DEVICES="${GPU}" python /track1/official/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py \
  --model /track1/weights/mhr/assets/mhr_model.pt --input "/track1/${M}/${TAG}.verts.npy" --output "/track1/${D}/${TAG}.npz" | tail -1
echo "PASS: ${TAG} ${STAGE} lam ${LAM} -> ${D}"
