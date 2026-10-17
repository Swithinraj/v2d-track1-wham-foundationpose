#!/bin/bash
# Hand-side contact fix of assembled episodes (GPU; called by pack.sh with HAND_FIX=1):
#   scripts/budget_object_meshes.py  object meshes simplified to the scorer's budget + scored frames (host venv)
#   scripts/hand_contact_fix.py      refits only the pose columns that move the hand but none of the 22 scored joints
#                                    (fingers, wrist rotations), pushing hand vertices out of the object mesh
# PEN counts hand points inside the object; most of that depth is in fingers of grasps, which no object translation
# can clear. The 22 scored joints and the objects are unchanged, so ACC-H, ACC-O and CD-O are exactly unchanged and
# CD-H moves only by the hand vertices. Two passes (margins 4 and 6 mm, five rounds each, scored frames) by default.
# Usage: hand_fix.sh INPUT OUTPUT   (episode directory names under submissions/)
#   HAND_FIX_PASSES (default 2), HAND_FIX_EPISODES (default all). To split the work across GPUs, run one job per
#   disjoint episode subset with the same INPUT/OUTPUT (each writes only its episodes); the subsets must cover 0-29.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
IN="$1"; OUT="$2"
PASSES="${HAND_FIX_PASSES:-2}"
EPISODES="${HAND_FIX_EPISODES:-$(seq -s ' ' 0 29)}"
MESHES="outputs/hand_fix/${IN}_meshes"
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/budget_object_meshes.py" --episodes "${TRACK1_ROOT}/submissions/${IN}" \
  --out "${TRACK1_ROOT}/${MESHES}" --only ${EPISODES}
SRC="${IN}"
for pass in $(seq 1 "${PASSES}"); do
  margin=$([[ "${pass}" == 1 ]] && echo 0.004 || echo 0.006)
  DST="${OUT}_pass${pass}"
  [[ "${pass}" == "${PASSES}" ]] && DST="${OUT}"
  in_container mhr_converter.sif python -u /track1/scripts/hand_contact_fix.py --input "/track1/submissions/${SRC}" \
    --out "/track1/submissions/${DST}" --meshes "/track1/${MESHES}" --margin "${margin}" --rounds 5 --scored_only \
    --episodes ${EPISODES}
  SRC="${DST}"
done
echo "PASS: hand contact fix ${IN} -> ${OUT} (${PASSES} passes)"
