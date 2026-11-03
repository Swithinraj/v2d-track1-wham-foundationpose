#!/bin/bash
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=wr24
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/model-eval-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/model-eval-%j.err
set -euo pipefail
P1=/work/mnidgu2s/project1
SIF=$P1/repo/containers/mhr_converter.sif
C=/p1/track1_objbias/calib_npz; E=/p1/track1_objbias/ens; S=/p1/repo/submissions
run() { apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=0 --bind $P1:/p1 $SIF python "$@"; }
for n in P4Cs23f08 E3W5s; do run $E/kit2calib.py $S/episodes_cand_$n $C/P4 $E/z_$n; done
run $E/ens_reframe.py --old $E/z_P4Cs23f08 --new $E/z_E3W5s --proxies MF=$C/MF P1=$C/P1 OTa=$C/OTa P2=$C/P2 C1=$C/C1 \
  --out $E/e3w5_reframe.json --model /p1/repo/weights/mhr/assets/mhr_model.pt --kit /p1/repo/official/v2d_submission_kit
echo "PASS: model eval"
