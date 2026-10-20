#!/bin/bash
# Held objects' depth from hand contact (CPU + one GPU step), human fixed:
#   scripts/mhr_bodies.py           dense hand vertices of the reference human (GPU)
#   scripts/interaction_states.py   free / contact / grasp per hand and frame (forward-backward over the whole video)
#   scripts/held_object_depth.py    inside grasp intervals, a smooth shift of the object along the camera ray that puts
#                                   the hands' contact patches on its surface (objects' rotation and scale unchanged)
#   scripts/held_object_depth_gate.py  keeps an episode's shift only if the object's silhouette IoU does not drop and
#                                   the added object acceleration stays <= MAX_DACC (default 0.003 cm/frame^2)
# Usage: held_object_depth.sh BASE OUTPUT   (episode directory names under submissions/; BASE holds the reference human
#   and objects, e.g. a contact-solve output). Follow with contact_solve.sh so the hands react to the moved objects.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
BASE="$1"; OUT="$2"; W="outputs/held_object_depth/${BASE}"
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/contact_sdf.py" --episodes "${TRACK1_ROOT}/submissions/${BASE}" --out "${TRACK1_ROOT}/outputs/contact_sdf/${BASE}"
in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir "/track1/submissions/${BASE}" --out "/track1/${W}/bodies"
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/interaction_states.py" "${TRACK1_ROOT}/${W}/states" "${TRACK1_ROOT}/submissions/${BASE}" \
  "${TRACK1_ROOT}/${W}/bodies" "${TRACK1_ROOT}/outputs/contact_sdf/${BASE}"
in_container v2d_cari4d.sif python -u /track1/scripts/held_object_depth.py "/track1/submissions/${BASE}" "/track1/${W}/bodies" \
  "/track1/outputs/contact_sdf/${BASE}" "/track1/${W}/states" "/track1/${W}/shifted"      # needs OpenCV + Pillow + SciPy
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/held_object_depth_gate.py" "${TRACK1_ROOT}/submissions/${BASE}" "${TRACK1_ROOT}/${W}/shifted" \
  "${TRACK1_ROOT}/submissions/${OUT}" "${MAX_DACC:-0.003}"
echo "PASS: held-object depth ${BASE} -> ${OUT}"
