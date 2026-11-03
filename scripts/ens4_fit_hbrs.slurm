#!/bin/bash
#SBATCH --job-name=model-fit
#SBATCH --partition=gpu
#SBATCH --nodelist=wr24
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/model-fit-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/model-fit-%j.err
set -euo pipefail
P1=/work/mnidgu2s/project1
SIF=$P1/repo/containers/mhr_converter.sif
C=/p1/track1_objbias/calib_npz
E=/p1/track1_objbias/ens
COMMON="--members P4=$C/P4:1 P2=$C/P2:1 C1=$C/C1:1 --align temporal --base $C/P4tab102 --model /p1/repo/weights/mhr/assets/mhr_model.pt --kit /p1/repo/official/v2d_submission_kit"
run() { apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=$1 --bind $P1:/p1 $SIF python "${@:2}"; }
for variant in free pin; do
  extra=""; [[ $variant == pin ]] && extra="--pin-first"
  pids=()
  for i in 0 1 2 3; do run $i $E/ensemble_fit.py $COMMON $extra --out $E/ens4_$variant --shard $i --nshards 4 & pids+=($!); done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  n=$(ls $P1/track1_objbias/ens/ens4_$variant/episode_*.npz | wc -l); echo "ens4_$variant: $n/30 (fail=$fail)"
  [[ $fail == 0 && $n == 30 ]] || { echo "FAIL fit $variant"; exit 1; }
done
run 0 $E/ens_reframe.py --old $C/P4tab102 --new $E/ens4_free --proxies MF=$C/MF P1=$C/P1 OTa=$C/OTa P2=$C/P2 C1=$C/C1 \
  --out $E/ens4_reframe.json --model /p1/repo/weights/mhr/assets/mhr_model.pt --kit /p1/repo/official/v2d_submission_kit
echo "PASS: model fit"
