#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-pva-all
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pva_all_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export TRACK1_EXTRA_BINDS="/work/mnidgu2s/project1/vpq-vitpose /work/mnidgu2s/project1/research/HTD-Refine /work/mnidgu2s/project1/research/pva_site"
source pipeline/env.sh
CODE=/work/mnidgu2s/project1/vpq-vitpose
trap 'echo "PVA_ALL_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "PVA_ALL_START node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES}"
mkdir -p work/pva/inputs_all
ready=()
for episode in {0..29}; do
  if "$VENV_PYTHON" "$CODE/scripts/pva_inputs.py" --root "$TRACK1_ROOT" --episodes "$episode" --out work/pva/inputs_all; then
    ready+=("$episode")
  else
    echo "PVA_PREP_FAILED episode=$episode; baseline fallback retained"
  fi
done
(( ${#ready[@]} > 0 ))
printf '%s\n' "${ready[@]}" > work/pva/ready_all.txt
in_container v2d_cari4d.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  PYTHONPATH=/work/mnidgu2s/project1/research/pva_site \
  python -u "$CODE/scripts/run_pva_track1.py" \
  --upstream /work/mnidgu2s/project1/research/HTD-Refine \
  --model /work/mnidgu2s/project1/research/HTD-Refine/inputs/checkpoints/pvanet.pt \
  --inputs /track1/work/pva/inputs_all --out /track1/work/pva/predictions_all --episodes "${ready[@]}"
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u "$CODE/scripts/pva_mhr_joints.py" --root /track1 \
  --episodes-dir /track1/submissions/episodes_cand_P2 --out /track1/work/pva/joints_P2_all --episodes "${ready[@]}"
"$VENV_PYTHON" "$CODE/scripts/pva_root_dynamics.py" \
  --base submissions/episodes_cand_P2 --joints work/pva/joints_P2_all --predictions work/pva/predictions_all \
  --scored outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json \
  --out submissions/episodes_cand_PVA_ACC
echo PVA_ALL_BUILT
