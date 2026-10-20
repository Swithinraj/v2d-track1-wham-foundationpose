#!/bin/bash
# Contact-constrained human solve with the objects held fixed (GPU):
#   scripts/contact_sdf.py            scorer-consistent signed-distance grid of every object (budgeted mesh, winding
#                                     number sign) -> outputs/contact_sdf/<OBJECTS>/
#   scripts/human_contact_solver.py   per-frame arm/hand corrections against non-penetration (every hand vertex),
#                                     gated contact + rigidity in stable grasps, MediaPipe 2D, image preservation of
#                                     reliable landmarks (corrections prefer the camera-ray direction, depth bounded),
#                                     priors and temporal smoothness; objects and identity unchanged
# Usage: contact_solve.sh HUMAN OBJECTS OUTPUT   (episode directory names under submissions/)
#   HUMAN   episodes whose human is solved (identity and pose)
#   OBJECTS episodes whose object poses, scale and meshes are used (copied into OUTPUT unchanged)
#   CONTACT_ARGS overrides the solver settings (default: the settings of candidate IPf30).
#   CONTACT_STATES=1 first computes the offline interaction states (free / contact / grasp per hand and frame,
#   scripts/interaction_states.py on the input human's MHR bodies) and passes them to the solver (--states_dir).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HUMAN="$1"; OBJECTS="$2"; OUT="$3"
ARGS="${CONTACT_ARGS:---w_prior_arm 2000 --w_img 20 --w_cap 1000 --w_con 200000 --w_rig 20000 --nsub 0}"
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/contact_sdf.py" --episodes "${TRACK1_ROOT}/submissions/${OBJECTS}" \
  --out "${TRACK1_ROOT}/outputs/contact_sdf/${OBJECTS}"
if [[ "${CONTACT_STATES:-0}" == 1 ]]; then
  W="outputs/contact_states/${HUMAN}"
  in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir "/track1/submissions/${HUMAN}" --out "/track1/${W}/bodies"
  "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/interaction_states.py" "${TRACK1_ROOT}/${W}/states" "${TRACK1_ROOT}/submissions/${OBJECTS}" \
    "${TRACK1_ROOT}/${W}/bodies" "${TRACK1_ROOT}/outputs/contact_sdf/${OBJECTS}"
  ARGS="${ARGS} --states_dir /track1/${W}/states"
fi
in_container mhr_converter.sif python -u /track1/scripts/human_contact_solver.py --human "/track1/submissions/${HUMAN}" \
  --objects "/track1/submissions/${OBJECTS}" --out "/track1/submissions/${OUT}" --sdf "/track1/outputs/contact_sdf/${OBJECTS}" \
  --episodes $(seq -s ' ' 0 29) ${ARGS}
echo "PASS: contact solve ${HUMAN} + objects of ${OBJECTS} -> ${OUT}"
