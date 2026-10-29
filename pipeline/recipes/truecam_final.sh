#!/bin/bash
# Candidate "<base> on the measured rig camera" (see truecam.md): every stage uses MoGe-2's camera, whose principal point
# is the image centre while the rig's real one lies ~54 / 38 px away (4.3 deg). The motion head's pre-contact output is
# re-expressed in the measured camera (same image evidence, new 3D interpretation: one linear map per episode, the human
# moved and turned with its viewing ray, the objects through their FoundationPose camera), then the same contact stage
# (w_pen 4e6), whose image term projects with the measured camera.
#   BMQ3KGM on the measured camera:  truecam_final.sh episodes_cand_BMQ3KGM_in episodes_cand_BMQ3KGMtp
# Usage: pipeline/recipes/truecam_final.sh IN OUTPUT
#   IN: the motion head's pre-contact output (gru_depth_final.sh writes it as <its OUTPUT>_in or _gd_in).
#   TRUECAM_MODE: tcp (default) = principal point only, MoGe's focal kept; tc = focal too (scripts/measured_camera.py).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; OUT="$2"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; I="${OUT}_in"
require_file "${S}/${IN}/episode_000029.npz" "build the base candidate first (gru_depth_final.sh)"
require_file "${TRACK1_ROOT}/config/camera/episode_cameras.json" "scripts/camera_identify.py"
"${HOST_PYTHON}" scripts/measured_camera.py --mode "${TRUECAM_MODE:-tcp}"
rm -rf "${S:?}/${I}" "${S}/${I}_afw" "${S}/${I}_sd4" "${S}/${OUT}"
in_container mhr_converter.sif python -u /track1/scripts/true_cam_params.py --input "/track1/submissions/${IN}" \
  --out "/track1/submissions/${I}" --old_root /track1 --k_new /track1/outputs/camera_measured/k_new.json
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 4e6"
export CONTACT_ALIGNMENT_DIR=outputs/alignment_measured
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${I}" "${I}" "${I}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${I}_afw" \
  --base "/track1/submissions/${I}" --out "/track1/submissions/${I}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${I}_sd4" "${I}_sd4" "${OUT}"
echo "PASS: ${IN} on the measured camera + contact stage -> submissions/${OUT}"
