#!/bin/bash
#SBATCH --job-name=locscore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/locscore-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/locscore-%j.err

# Pseudo-GT reference build + full candidate sweep, 4-way data parallel over GPUs.
set -euo pipefail
R=/work/mnidgu2s/project1/repo
W=/work/mnidgu2s/project1/track1_objbias
cd "$R"
SIF=containers/mhr_converter.sif
RUN="apptainer exec --cleanenv --nv --bind /work/mnidgu2s/project1:/p1 $SIF"

# 1) pseudo-GT reference: vertex-average of 3 independent human systems + best objects
$RUN python /p1/repo/scripts/localscore.py build-ref \
  --humans /p1/track1_objbias/calib_npz/P2 \
           /p1/track1_objbias/calib_npz/MF \
           /p1/track1_objbias/calib_npz/C1 \
  --objects /p1/track1_objbias/calib_npz/P4tab102 \
  --out /p1/track1_objbias/outputs/pseudo_gt --device cuda

# 2) task list: every converted candidate
$RUN python - <<'EOF'
import json, glob, os
tasks = [{"name": os.path.basename(p), "path": p}
         for p in sorted(glob.glob("/p1/track1_objbias/calib_npz/*")) if os.path.isdir(p)]
json.dump(tasks, open("/p1/track1_objbias/tasks.json", "w"))
print(len(tasks), "tasks")
EOF

# 3) sweep, one shard per GPU (--cleanenv strips env vars; --env survives)
for i in 0 1 2 3; do
  apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=$i \
    --bind /work/mnidgu2s/project1:/p1 $SIF python /p1/repo/scripts/score_sweep.py \
    --ref /p1/track1_objbias/outputs/pseudo_gt \
    --tasks /p1/track1_objbias/tasks.json \
    --out /p1/track1_objbias/locscores --shard $i --nshards 4 &
done
wait
echo "PASS: sweep complete"
ls /work/mnidgu2s/project1/track1_objbias/locscores/*.json | wc -l
