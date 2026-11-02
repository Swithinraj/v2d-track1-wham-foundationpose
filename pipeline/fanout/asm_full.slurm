#!/bin/bash
#SBATCH --job-name=asm-chain-a
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/asm-chain-%j.out
#SBATCH --error=logs/asm-chain-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com

# IMPORTANT (cluster): the .venvs' python needs `module load Python/3.12.3` even for pip
# (EasyBuild-built interpreter links module libs; without the module pip/python segfault).
# numba (needed by scripts/contact_sdf.py via the kit's penetration kernels) is NOT in the
# kit's requirements.txt; installed manually into .venvs/submission on CLAIX 2026-11-02.

# Full assembly chain, RUN_NAME=sam (the pack.sh default recipe): build_track1_episodes_sam
# (all 30) -> depth trend -> human_refine (lambda 300, MediaPipe) -> contact_solve (IPf30
# settings) -> hand_fix -> pack (LOCAL validation parquet, no COMMIT_URL -> not uploadable).
# Stage-gated: each stage is skipped when its output sentinel exists, so the job is resumable
# at stage granularity (the refine/solve/fix scripts themselves always process full dirs —
# 33 scripts hardcode range(30) — which is fine because all 30 episodes' inputs are rebuilt).
# The candidate BMQ chain (ViTPose landmarks, bm work-root) is a follow-up; this is the
# pipeline's own default chain and the fastest path to a complete local parquet.
# Submit: mkdir -p logs && sbatch pipeline/fanout/asm_full.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"
S="${TRACK1_ROOT}/submissions"
RUN="sam"

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }
have() { [[ -s "${S}/$1/episode_000029.npz" ]]; }

echo "== [1/5] assemble + depth trend (all 30)"
if have "episodes_${RUN}_depth"; then
  echo "SKIP: episodes_${RUN}_depth present"
else
  rm -rf "${S:?}/episodes_${RUN}" "${S:?}/episodes_${RUN}_depth" "${S:?}/episodes_${RUN}_refine"
  t in_container v2d_cari4d.sif python /track1/scripts/build_track1_episodes_sam.py \
    --out "/track1/submissions/episodes_${RUN}"
  t in_container v2d_cari4d.sif python /track1/scripts/sam3d_body_depth_trend.py \
    --input "/track1/submissions/episodes_${RUN}" --out "/track1/submissions/episodes_${RUN}_depth" --alpha 0.75
fi

echo "== [2/5] human refine (lambda 300, MediaPipe landmarks)"
if have "episodes_${RUN}_refine"; then
  echo "SKIP: episodes_${RUN}_refine present"
else
  rm -rf "${S:?}/episodes_${RUN}_refine" "${TRACK1_ROOT}/outputs/human_refine_${RUN}"
  t env REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK="human_refine_${RUN}" \
    bash pipeline/stages/human_refine.sh "episodes_${RUN}_depth" "episodes_${RUN}_depth" "episodes_${RUN}" "episodes_${RUN}_refine"
fi

echo "== [3/5] contact solve (default IPf30 settings)"
if have "episodes_${RUN}_afw"; then
  echo "SKIP: episodes_${RUN}_afw present"
else
  rm -rf "${S:?}/episodes_${RUN}_afw"
  t bash pipeline/stages/contact_solve.sh "episodes_${RUN}_refine" "episodes_${RUN}_refine" "episodes_${RUN}_afw"
fi

echo "== [4/5] hand fix"
if have "episodes_${RUN}_handfix"; then
  echo "SKIP: episodes_${RUN}_handfix present"
else
  rm -rf "${S:?}/episodes_${RUN}_handfix"
  t bash pipeline/stages/hand_fix.sh "episodes_${RUN}_afw" "episodes_${RUN}_handfix"
fi

echo "== [5/5] pack (local validation parquet)"
PARQUET=$(ls "${S}"/track1_LOCAL_CHECK*.parquet 2>/dev/null | head -1 || true)
if [[ -n "${PARQUET}" && -s "${PARQUET}" && "${PARQUET}" == *track1_LOCAL_CHECK* ]]; then
  echo "SKIP: local parquet already present: ${PARQUET}"
else
  t env PACK_EPISODES="episodes_${RUN}_handfix" RUN_NAME="${RUN}" bash pipeline/stages/pack.sh
fi

ls -la "${S}"/*.parquet | tail -3
echo "PASS: assembly chain complete — submissions/episodes_${RUN}_handfix packed"
