#!/bin/bash
# Human refinement of assembled episodes (GPU; called by pack.sh with HUMAN_REFINE=1):
#   1. contact hands (MHR) and grasp windows / object-anchored hand shapes   grasp_windows.py, hand_anchors.py
#   2. lambda 10 mesh-smoothing refit as the process model, its joints/hands human_refit.py, mhr_bodies.py
#   3. per-hand wrist fusion RTS (MHR + MediaPipe 2D + object anchors)       wrist_fusion.py
#   4. refit with lambda REFINE_LAM, fused wrists and anchored hands, then a second smoothing pass
#   5. lateral root drift RTS from MediaPipe, gated by the actor's SAM 2 mask   person_mask_sdf.py, root_lateral.py
# Usage: human_refine.sh INPUT CONTACT PRE OUTPUT   (episode directory names under submissions/)
#   HUMAN_REFINE_WORK: work directory under outputs/ (default human_refine); REFINE_LAM (default 100).
#   INPUT   assembled episodes to refine (after depth trend and hoop smoothing)
#   CONTACT episodes whose object track defines grasp contact (after the depth trend, before smoothing)
#   PRE     episodes before the depth trend (the contact hands are computed from them and moved by it)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
IN="$1"; CONTACT="$2"; PRE="$3"; OUT="$4"
LAM="${REFINE_LAM:-100}"
WORK="${HUMAN_REFINE_WORK:-human_refine}"
S="${TRACK1_ROOT}/submissions"; W="${TRACK1_ROOT}/outputs/${WORK}"; C="/track1/outputs/${WORK}"; mkdir -p "${W}"
missing_sdf=()
for e in $(seq 0 29); do
  tag="$(episode_tag "${e}")"
  require_file "${TRACK1_ROOT}/outputs/mediapipe/${tag}.npz" "run pipeline/stages/keypoints2d.sh ${e}"
  require_file "${TRACK1_ROOT}/outputs/person/${tag}/sam2_masks/0/000000.png" "run pipeline/stages/person.sh ${e}"
  [[ -s "${TRACK1_ROOT}/outputs/person/${tag}/mask_sdf.npz" ]] || missing_sdf+=("${e}")
done
mhr() { in_container mhr_converter.sif python "/track1/scripts/$1" "${@:2}"; }
py() { "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/$1" "${@:2}"; }

mhr mhr_bodies.py --episodes_dir "/track1/submissions/${PRE}" --out ${C}/hands_contact --hands-only
py grasp_windows.py --contact "${S}/${CONTACT}" --pre "${S}/${PRE}" --hands "${W}/hands_contact" --out "${W}/grasp_windows.json"
py hand_anchors.py --windows "${W}/grasp_windows.json" --contact "${S}/${CONTACT}" --pre "${S}/${PRE}" \
  --hands "${W}/hands_contact" --out "${W}/anchors"
mhr human_refit.py --input "/track1/submissions/${IN}" --out "/track1/submissions/${IN}_s10" --lam 10
mhr mhr_bodies.py --episodes_dir "/track1/submissions/${IN}_s10" --out ${C}/bodies_s10
py wrist_fusion.py --bodies "${W}/bodies_s10" --episodes "${S}/${IN}" --anchors "${W}/anchors" \
  --mediapipe "${TRACK1_ROOT}/outputs/mediapipe" --out "${W}/fusion"
mhr human_refit.py --input "/track1/submissions/${IN}" --out "/track1/submissions/${IN}_fit" --lam "${LAM}" \
  --fusion ${C}/fusion --anchors ${C}/anchors
mhr human_refit.py --input "/track1/submissions/${IN}_fit" --out "/track1/submissions/${IN}_fit2" --lam "${LAM}"
mhr mhr_bodies.py --episodes_dir "/track1/submissions/${IN}" --out ${C}/bodies_base
if ((${#missing_sdf[@]})); then
  in_container v2d_cari4d.sif python /track1/scripts/person_mask_sdf.py --episodes "${missing_sdf[@]}"
fi
py root_lateral.py --bodies "${W}/bodies_base" --mediapipe "${TRACK1_ROOT}/outputs/mediapipe" \
  --masks "${TRACK1_ROOT}/outputs/person" --base "${S}/${IN}" --refined "${S}/${IN}_fit2" --out "${S}/${OUT}"
require_file "${S}/${OUT}/root_lateral_report.json"
echo "PASS: human refinement ${IN} -> ${OUT}"
