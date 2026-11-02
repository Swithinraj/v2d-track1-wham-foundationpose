#!/bin/bash
#SBATCH --job-name=asm-smoke-01
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/asm-smoke-%j.out
#SBATCH --error=logs/asm-smoke-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com

# ASSEMBLY smoke test: exercise the chain stages BELOW the per-episode outputs (which the
# prep-ep / prep-track / prep-mesh arrays already rebuilt and verified) on TWO episodes (0, 1):
#   build_track1_episodes_sam (human+object assembly, depth-scale mapping)
#   -> sam3d_body_depth_trend (root follows MoGe depth trend, alpha 0.75)
#   -> human_refine.sh (lambda 300, object-anchored hands, MediaPipe RTS wrist fusion, root drift)
#   -> contact_solve.sh (contact-constrained human solve, objects fixed)
#   -> hand_fix.sh (fingers/wrist rotations)
#   -> eval_reconstruction.py -> local validation parquet (2 episodes, NOT for upload)
# The CANDIDATE chain (bmq/bmqk/bmq3k + ViTPose landmarks) is a follow-up: it needs
# weights/vitpose (downloading) + kp2d_vpx + the objects decision; this smoke first proves the
# machinery runs on the cluster.
# Submit: mkdir -p logs && sbatch pipeline/fanout/asm_smoke.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"
EPS="0 1"
RUN="smoke"
S="${TRACK1_ROOT}/submissions"

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }

echo "== [1/5] assemble episodes ${EPS}"
rm -rf "${S:?}/episodes_${RUN}" "${S:?}/episodes_${RUN}_depth"
t in_container v2d_cari4d.sif python /track1/scripts/build_track1_episodes_sam.py \
  --out "/track1/submissions/episodes_${RUN}" --episodes ${EPS}
t in_container v2d_cari4d.sif python /track1/scripts/sam3d_body_depth_trend.py \
  --input "/track1/submissions/episodes_${RUN}" --out "/track1/submissions/episodes_${RUN}_depth" \
  --episodes ${EPS} --alpha 0.75

echo "== [2/5] human refine (lambda 300, MediaPipe landmarks)"
rm -rf "${S:?}/episodes_${RUN}_refine" "outputs/human_refine_${RUN}"
t env REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK="human_refine_${RUN}" \
  bash pipeline/stages/human_refine.sh "episodes_${RUN}_depth" "episodes_${RUN}_depth" "episodes_${RUN}" "episodes_${RUN}_refine"

echo "== [3/5] contact solve (default IPf30 settings)"
rm -rf "${S:?}/episodes_${RUN}_afw"
t bash pipeline/stages/contact_solve.sh "episodes_${RUN}_refine" "episodes_${RUN}_refine" "episodes_${RUN}_afw"

echo "== [4/5] hand fix"
rm -rf "${S:?}/episodes_${RUN}_handfix"
t env HAND_FIX_EPISODES="${EPS}" bash pipeline/stages/hand_fix.sh "episodes_${RUN}_afw" "episodes_${RUN}_handfix"

echo "== [5/5] pack (local validation parquet, 2 episodes)"
t env PACK_EPISODES="episodes_${RUN}_handfix" RUN_NAME="${RUN}" bash pipeline/stages/pack.sh

ls -la "${S}"/*.parquet | tail -3
echo "PASS: assembly smoke complete — episodes ${EPS} through the full assembly chain"
