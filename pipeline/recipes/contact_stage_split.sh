#!/bin/bash
# The production two-pass contact stage (the one in truecam_final.sh) split across the GPUs of ONE allocation.
# Each GPU solves its own episodes into its own directory; the results are merged afterwards, so parallel solvers never
# touch the same file. Same settings and same scripts as the single-GPU recipe.
# Usage (inside one Slurm allocation): contact_stage_split.sh IN OUTPUT
#   IN: pre-contact episodes under submissions/. Writes IN_afw, IN_sd4 and OUTPUT.
#   CONTACT_GPUS (default 4), CONTACT_ALIGNMENT_DIR (default outputs/alignment_measured).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; OUT="$2"; S=submissions; GPUS="${CONTACT_GPUS:-4}"; KP=outputs/kp2d_vpx
export CONTACT_ALIGNMENT_DIR="${CONTACT_ALIGNMENT_DIR:-outputs/alignment_measured}"
require_file "${S}/${IN}/episode_000029.npz" "build the pre-contact episodes first"
ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 4e6"

# Episodes per GPU, balanced by scored frames (largest first, onto the least loaded GPU).
mapfile -t CHUNKS < <("${VENV_PYTHON}" - "${GPUS}" <<'PY'
import json, sys
scored = json.load(open("outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json"))
gpus = int(sys.argv[1]); load = [0] * gpus; chunks = [[] for _ in range(gpus)]
for episode, frames in sorted(scored.items(), key=lambda item: -len(item[1])):
    k = load.index(min(load)); load[k] += len(frames); chunks[k].append(int(episode))
for chunk in chunks:
    print(" ".join(map(str, sorted(chunk))))
PY
)

solve_pass() {   # HUMAN_AND_OBJECTS OUTPUT EXTRA_ARGS
  local base="$1" out="$2" extra="$3" pids=() failed=0
  "${VENV_PYTHON}" scripts/contact_sdf.py --episodes "${TRACK1_ROOT}/${S}/${base}" --out "${TRACK1_ROOT}/outputs/contact_sdf/${base}"
  rm -rf "${S:?}/${out}"
  for k in $(seq 0 $((GPUS - 1))); do
    in_container mhr_converter.sif CUDA_VISIBLE_DEVICES="${k}" CONTACT_ALIGNMENT_DIR="${CONTACT_ALIGNMENT_DIR}" \
      python -u /track1/scripts/human_contact_solver.py --human "/track1/${S}/${base}" --objects "/track1/${S}/${base}" \
      --out "/track1/${S}/${out}.part${k}" --sdf "/track1/outputs/contact_sdf/${base}" --episodes ${CHUNKS[$k]} ${ARGS} ${extra} &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == 0 ]] || { echo "FAIL: contact pass ${out}" >&2; exit 1; }
  mkdir -p "${S}/${out}"
  for k in $(seq 0 $((GPUS - 1))); do
    cp -a "${S}/${out}.part${k}/." "${S}/${out}/"
    rm -rf "${S:?}/${out}.part${k}"
  done
  # Frames and meshes come from the base; every episode must have been solved.
  [[ "$(ls "${S}/${out}"/episode_??????.npz | wc -l)" == 30 ]] || { echo "FAIL: ${out} is not complete" >&2; exit 1; }
}

solve_pass "${IN}" "${IN}_afw" ""
rm -rf "${S:?}/${IN}_sd4"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/${S}/${IN}_afw" \
  --base "/track1/${S}/${IN}" --out "/track1/${S}/${IN}_sd4" --sigma 4
solve_pass "${IN}_sd4" "${OUT}" "--no_arm"
echo "PASS: contact stage ${IN} -> ${S}/${OUT}"
