#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-gates
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/gates_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
LIST=()
for pair in "$@"; do
  name=${pair%%=*}; dir=${pair#*=}
  [[ -f work/analysis/full_$name/episode_000029.npz ]] || in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/$dir --out /track1/work/analysis/full_$name | tail -1
  LIST+=("$name=$dir:work/analysis/full_$name")
done
.venvs/submission/bin/python scripts/local_gates.py --scored outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json "${LIST[@]}"
