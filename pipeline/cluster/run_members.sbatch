#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-members
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/members_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
one() {   # GPU NAME
  local gpu=$1 name=$2 src=submissions/episodes_tp_$2
  if [[ "$name" == ktcp || "$name" == abl ]]; then src=submissions/episodes_base_$name    # already on the measured camera
  elif [[ "$name" == BMQ3KGM ]]; then src=submissions/episodes_cand_Mtp_in
  else
    [[ -f "$src/episode_000029.npz" ]] || in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=$gpu python -u /track1/scripts/true_cam_params.py \
      --input /track1/submissions/episodes_base_$name --out /track1/$src --old_root /track1 --k_new /track1/outputs/camera_measured/k_new.json | tail -1
  fi
  in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=$gpu python -u /track1/scripts/ensemble_meshes.py --episodes_dir /track1/$src --out /track1/work/analysis/meshes/$name
}
NAMES=(BMQ3KGM ktcp abl a0 VPQ3 nowham BMQO D R2 Afr rdL100 lam300 h2t BMQ k)
for i in "${!NAMES[@]}"; do
  one $((i % 4)) "${NAMES[$i]}" &
  if (( (i + 1) % 4 == 0 )); then wait; fi
done
wait
echo ALLDONE
