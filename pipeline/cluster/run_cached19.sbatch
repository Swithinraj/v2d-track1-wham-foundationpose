#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-cached19
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/cached19_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export CONTACT_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export EPISODES="3 9 10 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 29"
trap 'echo "CACHED19_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "CACHED19_START node=$(hostname) gpu=${CONTACT_GPU} commit=$(git rev-parse HEAD)"
# These builders are unchanged from published commit 5e0bc96. Reuse its runtime
# directory and the validated CoCoNet caches; development lives in vpq-vitpose.
bash pipeline/recipes/cari4d_candidate.sh N19H coconet 300 0.55
OWN_OBJECTS=1 bash pipeline/recipes/cari4d_candidate.sh N19HO coconet 300 0.55
echo CACHED19_BUILT
