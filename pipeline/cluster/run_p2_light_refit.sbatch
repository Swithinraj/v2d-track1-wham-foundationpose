#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=v2d-p1-p2-light
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p2_light_refit_%j.log
# Light mesh smoothing of P2 (the ACC-H best), then the usual per-frame MHR refit. Earlier refits used lam 100 (about
# 1.5 Hz), which removes real motion; lam 1 is about 5 Hz and lam 0.3 about 7 Hz. Objects and identity stay byte-identical.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
source pipeline/env.sh
trap 'echo "P2LIGHT_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "P2LIGHT_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} commit=$(git -C /work/mnidgu2s/project1/vpq-vitpose rev-parse HEAD)"
for pair in 1:P2L1 0.3:P2L03; do
  lam=${pair%%:*}; out=submissions/episodes_cand_${pair##*:}
  [[ -e $out ]] && { echo "$out exists; refusing to overwrite"; exit 2; }
  in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
    python -u /track1/scripts/human_refit.py --input /track1/submissions/episodes_cand_P2 --out "/track1/$out" --lam "$lam" | tail -1
done
echo P2LIGHT_BUILT
