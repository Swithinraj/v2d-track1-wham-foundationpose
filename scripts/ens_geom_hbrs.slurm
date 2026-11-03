#!/bin/bash
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=wr24
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/model-eval-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/model-eval-%j.err
set -euo pipefail
P1=/work/mnidgu2s/project1
SIF=$P1/repo/containers/mhr_converter.sif
C=/p1/track1_objbias/calib_npz
CH=""
for n in P4 P3 P2 P1 MF C1 C2 V5 OTa; do CH="$CH --chain $n=$C/$n"; done
CH="$CH --chain ENS3=/p1/track1_objbias/ens/full"
pids=()
for i in 0 1 2 3; do
  apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=$i,TRACK1_ROOT=/p1/repo,LOCALSCORE_DIR=/p1/track1_objbias/ens \
    --bind $P1:/p1 $SIF python /p1/track1_objbias/ens/ens_geometry.py $CH --step 6 \
    --out /p1/track1_objbias/ens/geom --shard $i --nshards 4 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] && echo "PASS: ens geom" || { echo "FAIL: ens geom"; exit 1; }
