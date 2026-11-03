#!/bin/bash
#SBATCH --job-name=ens-full
#SBATCH --partition=gpu
#SBATCH --nodelist=wr14
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/ens-full-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/ens-full-%j.err
set -euo pipefail
P1=/work/mnidgu2s/project1
SIF=$P1/repo/containers/mhr_converter.sif
pids=()
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for i in 0 1 2 3; do
  apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=$i --bind $P1:/p1 $SIF \
    python /p1/track1_objbias/ens/ensemble_fit.py \
    --ref /p1/track1_objbias/outputs/pseudo_gt --base /p1/track1_objbias/calib_npz/P2 \
    --out /p1/track1_objbias/ens/full --model /p1/repo/weights/mhr/assets/mhr_model.pt \
    --kit /p1/repo/official/v2d_submission_kit --shard $i --nshards 4 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
n=$(ls $P1/track1_objbias/ens/full/episode_*.npz 2>/dev/null | wc -l)
echo "episodes written: $n/30"
[[ $fail == 0 && $n == 30 ]] && echo "PASS: ens full" || { echo "FAIL: ens full"; exit 1; }
