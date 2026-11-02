#!/bin/bash
#SBATCH --job-name=gru-tc
#SBATCH --account=rwth2243
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gru-tc-%j.out
#SBATCH --error=logs/gru-tc-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com
#
# GRU motion-head retrain in the measured rig camera (v2 recipe: 6000 steps, all groups).
#
# Stage 0 (CPU, minutes): re-express the training caches into the measured camera
#                         (pipeline/gru_retrain/prepare_caches.sh -> outputs/gru_cache_tc).
# Stage 1 (GPU, ~1 h):    scripts/gru_train.py inside mhr_converter.sif, 1 GPU
#                         (the script is single-device; multi-GPU buys nothing).
# Stage 2:                evaluate the final checkpoint on the validation group.
#
# The depth fine-tune (gru_depth_train.py) is deliberately NOT part of this job: the deployed
# chain runs EPS_SCALE=0 (depth head off) and the cue caches would need their own camera
# treatment — open item, see /data/formhoi/RUNBOOK.md §6.
#
# SUBMIT ONLY WITH THE OWNER'S APPROVAL (from the repo root on a CLAIX login node):
#   ssh claix
#   cd <repo> && mkdir -p logs
#   CACHE_DIR=<dir with <group>_episode_XXXXXX.npz> GROUPS=pipeline/gru_retrain/groups.tsv \
#     TRAIN_GROUPS=<14 training group names> VAL_GROUPS=<validation group name> \
#     sbatch pipeline/gru_retrain/gru_retrain.slurm
#
# Required env:
#   CACHE_DIR    dir with the exported caches <group>_episode_XXXXXX.npz
#   GROUPS       TSV manifest <group_name><TAB><group_root> (root = processed checkout with
#                outputs/camera/episode_*/intrinsics.json and episodes_metadata.jsonl)
#   TRAIN_GROUPS comma-separated group names trained on (v2: 14 recording-disjoint groups)
#   VAL_GROUPS   the validation group (v2's fifth group, never trained on)
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh

CACHE_DIR="${CACHE_DIR:?set CACHE_DIR (dir with <group>_episode_XXXXXX.npz)}"
GROUPS="${GROUPS:?set GROUPS to the groups.tsv manifest}"
TRAIN_GROUPS="${TRAIN_GROUPS:?set TRAIN_GROUPS (comma-separated group names)}"
VAL_GROUPS="${VAL_GROUPS:?set VAL_GROUPS (validation group name)}"
OUT="config/gru_motion_v2_tc"
OUT_CACHE="${TRACK1_ROOT}/outputs/gru_cache_tc"
STEPS=6000
mkdir -p logs "${OUT}"

echo "== [0/3] re-expressing caches into the measured camera (CPU, minutes)"
bash pipeline/gru_retrain/prepare_caches.sh "${GROUPS}" 2>&1 | tee "logs/gru-tc-prep-${SLURM_JOB_ID}.log"
FIRST="${TRAIN_GROUPS%%,*}"
ls "${OUT_CACHE}/${FIRST}_episode_"*.npz >/dev/null 2>&1 || { echo "prep produced no ${FIRST} caches" >&2; exit 3; }
N_TC=$(ls "${OUT_CACHE}"/*_episode_??????.npz | wc -l)
N_ORIG=$(ls "${CACHE_DIR}"/*_episode_??????.npz | wc -l)
[[ "${N_TC}" -eq "${N_ORIG}" ]] || { echo "cache count changed: ${N_ORIG} -> ${N_TC}" >&2; exit 3; }
grep -h "PASS\|anchor residual" "logs/gru-tc-prep-${SLURM_JOB_ID}.log" | tail -20 || true

echo "== [1/3] motion head, ${STEPS} steps, v2 recipe (GPU, ~1 h)"
in_container mhr_converter.sif python -u /track1/scripts/gru_train.py "/track1/outputs/gru_cache_tc" \
  "/track1/${OUT}" --train "${TRAIN_GROUPS}" --val "${VAL_GROUPS}" --steps "${STEPS}" \
  2>&1 | tee "logs/gru-tc-motion-${SLURM_JOB_ID}.log"
require_file "${OUT}/refiner_motion_best.pt" \
  "validation never beat E2 — inspect logs/gru-tc-motion-${SLURM_JOB_ID}.log"

echo "== [2/3] evaluate the final checkpoint on the validation group"
in_container mhr_converter.sif python -u /track1/scripts/gru_train.py "/track1/outputs/gru_cache_tc" \
  "/track1/${OUT}" --train "${TRAIN_GROUPS}" --val "${VAL_GROUPS}" \
  --eval_only "/track1/${OUT}/refiner_motion_best.pt" 2>&1 | tee "logs/gru-tc-eval-${SLURM_JOB_ID}.log"

echo "== [3/3] done"
echo "model: ${OUT}/refiner_motion_best.pt (+ refiner_motion.pt final, refiner_motion_norm.pt statistics)"
echo "deployment (separate, also approval-gated): apply the measured-camera map BEFORE the motion"
echo "head instead of after it, then contact; see RUNBOOK §6 and truecam_final.sh for the map."
echo "PASS"
