#!/bin/bash
#SBATCH --job-name=locscore2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mnidgu2s/project1/track1_objbias/logs/locscore2-%j.out
#SBATCH --error=/work/mnidgu2s/project1/track1_objbias/logs/locscore2-%j.err

# Variant sweep against the existing pseudo-GT reference (job 2: smoothing / frame-0).
set -euo pipefail
R=/work/mnidgu2s/project1/repo
W=/work/mnidgu2s/project1/track1_objbias
cd "$R"
SIF=containers/mhr_converter.sif

apptainer exec --cleanenv --nv --bind /work/mnidgu2s/project1:/p1 $SIF python - <<'EOF'
import json, glob, os
tasks = [{"name": os.path.basename(p), "path": p}
         for p in sorted(glob.glob("/p1/track1_objbias/variants/*")) if os.path.isdir(p)]
json.dump(tasks, open("/p1/track1_objbias/tasks_variants.json", "w"))
print(len(tasks), "variant tasks")
EOF

for i in 0 1 2 3; do
  apptainer exec --cleanenv --nv --env CUDA_VISIBLE_DEVICES=$i \
    --bind /work/mnidgu2s/project1:/p1 $SIF python /p1/repo/scripts/score_sweep.py \
    --ref /p1/track1_objbias/outputs/pseudo_gt \
    --tasks /p1/track1_objbias/tasks_variants.json \
    --out /p1/track1_objbias/locscores --shard $i --nshards 4 &
done
wait
echo "PASS: variant sweep complete"
ls /work/mnidgu2s/project1/track1_objbias/locscores/*.json | wc -l
