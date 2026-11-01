#!/bin/bash
# A candidate from CARI4D's person on the episodes finished so far (see cari4d.md). The rest keep the current best person, so a
# partial run already gives Kaggle's answer, scaled by the share of episodes. Objects are carried to the person.
# Usage (inside one allocation): cari4d_candidate.sh NAME STAGE LAM BETA
#   STAGE init|coconet|refined; LAM mesh smoothing; BETA floor slide strength. Writes submissions/episodes_cand_NAME.
#   FILL (default episodes_person_V3): person for unfinished episodes. OBJECTS (default episodes_baseold_k_OTa), OBJECT_MESHES.
#   EPISODES: override the list of episodes taken from CARI4D. OWN_OBJECTS=1: CARI4D's objects on its episodes.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
NAME="$1"; STAGE="$2"; LAM="$3"; BETA="$4"; S=submissions; A=work/analysis; X="${STAGE}_l${LAM}"; E="work/cari4d/export/${X}"
FILL="${FILL:-episodes_personm_V3}"; OBJECTS="${OBJECTS:-episodes_baseold_k_OTa}"; OBJECT_MESHES="${OBJECT_MESHES:-${A}/meshes/BMQ3KGM_old}"
FILL_MESHES="${FILL_MESHES:-${A}/meshes/BMQ3KGM}"
EPISODES="${EPISODES:-$(ls work/cari4d/queue | sed -n 's/\.done$//p' | sort -n | tr '\n' ' ')}"
for e in ${EPISODES}; do bash pipeline/recipes/cari4d_to_params.sh "${e}" "${STAGE}" "${LAM}" "${CONTACT_GPU:-0}" | tail -1; done
rm -rf "${S:?}/episodes_c4_${NAME}" "${S}/episodes_c4tp_${NAME}" "${S}/episodes_c4p_${NAME}" "work/cari4d/camroot/${NAME}"
SUB="work/cari4d/select/${NAME}"; rm -rf "${SUB}"; mkdir -p "${SUB}/export" "${SUB}/params"
for e in ${EPISODES}; do
  t="$(episode_tag "${e}")"
  cp -l "${E}/${t}.object.npz" "${E}/${t}_object.glb" "${SUB}/export/"
  cp -l "work/cari4d/params/${X}/${t}.npz" "${SUB}/params/"
done
"${VENV_PYTHON}" scripts/cari4d_episodes.py --export "${SUB}/export" --params "${SUB}/params" --out "${S}/episodes_c4_${NAME}"
"${VENV_PYTHON}" scripts/cari4d_camroot.py --export "${SUB}/export" --k_measured outputs/camera_measured/k_new.json --out "work/cari4d/camroot/${NAME}"
in_container mhr_converter.sif python -u /track1/scripts/true_cam_params.py --input "/track1/${S}/episodes_c4_${NAME}" --out "/track1/${S}/episodes_c4tp_${NAME}" \
  --old_root "/track1/work/cari4d/camroot/${NAME}" --k_new "/track1/work/cari4d/camroot/${NAME}/k_new.json" | tail -1
BODIES="${A}/bodies_c4_${NAME}"; rm -rf "${BODIES}"
in_container mhr_converter.sif python -u /track1/scripts/floor_bodies.py --episodes_dir "/track1/${S}/episodes_c4tp_${NAME}" --out "/track1/${BODIES}" | tail -1
"${VENV_PYTHON}" scripts/floor_root.py --input "${S}/episodes_c4tp_${NAME}" --bodies "${BODIES}" --planes outputs/floor/planes_measured.json \
  --out "${S}/episodes_c4p_${NAME}" --lam 1e4 --beta "${BETA}" | tail -1
# The person: CARI4D's where finished, FILL elsewhere; anchor meshes to match.
P="${S}/episodes_person_${NAME}"; M="${A}/meshes/person_${NAME}"; rm -rf "${P}" "${M}"; mkdir -p "${P}" "${M}"
rm -rf "${A}/meshes/c4p_${NAME}"
in_container mhr_converter.sif python -u /track1/scripts/ensemble_meshes.py --episodes_dir "/track1/${S}/episodes_c4p_${NAME}" --out "/track1/${A}/meshes/c4p_${NAME}" | tail -1
for e in $(seq 0 29); do
  t="$(episode_tag "${e}")"
  if [[ -f "${S}/episodes_c4p_${NAME}/${t}.npz" ]]; then
    cp -l "${S}/episodes_c4p_${NAME}/${t}.npz" "${P}/"; cp -l "${A}/meshes/c4p_${NAME}/${t}.npy" "${M}/"
  else
    cp -l "${S}/${FILL}/${t}.npz" "${P}/"; cp -l "${FILL_MESHES}/${t}.npy" "${M}/"
  fi
done
OBJ="${S}/${OBJECTS}"; OBJM="${OBJECT_MESHES}"
if [[ "${OWN_OBJECTS:-0}" == 1 ]]; then
  # CARI4D's objects on its episodes, before the camera change, carried to the person like any others.
  rm -rf "${A}/meshes/c4old_${NAME}" "${S}/episodes_objs_${NAME}" "${A}/meshes/objs_${NAME}"
  in_container mhr_converter.sif python -u /track1/scripts/ensemble_meshes.py --episodes_dir "/track1/${S}/episodes_c4_${NAME}" --out "/track1/${A}/meshes/c4old_${NAME}" | tail -1
  OBJ="${S}/episodes_objs_${NAME}"; OBJM="${A}/meshes/objs_${NAME}"; mkdir -p "${OBJ}" "${OBJM}"
  for e in $(seq 0 29); do
    t="$(episode_tag "${e}")"
    if [[ -f "${S}/episodes_c4_${NAME}/${t}.npz" ]]; then
      cp -l "${S}/episodes_c4_${NAME}/${t}.npz" "${S}/episodes_c4_${NAME}/${t}_object.glb" "${OBJ}/"; cp -l "${A}/meshes/c4old_${NAME}/${t}.npy" "${OBJM}/"
    else
      cp -l "${S}/${OBJECTS}/${t}.npz" "${S}/${OBJECTS}/${t}_object.glb" "${OBJ}/"; cp -l "${OBJECT_MESHES}/${t}.npy" "${OBJM}/"
    fi
  done
fi
rm -rf "${S}/episodes_cand_${NAME}"
"${VENV_PYTHON}" scripts/object_transfer.py --people "${P}" --people_meshes "${M}" --objects "${OBJ}" --objects_meshes "${OBJM}" --out "${S}/episodes_cand_${NAME}"
echo "BUILT ${NAME}: CARI4D ${STAGE} lam ${LAM} beta ${BETA} on episodes: ${EPISODES}"
