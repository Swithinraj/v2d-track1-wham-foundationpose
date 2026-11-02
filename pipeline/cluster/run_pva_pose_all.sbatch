#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=v2d-p1-pva-pose-all
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pva_pose_all_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source pipeline/env.sh
CODE=/work/mnidgu2s/project1/vpq-vitpose
OUT=submissions/episodes_cand_PVA_POSE
trap 'echo "PVA_POSE_ALL_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "PVA_POSE_ALL_START node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES}"
# Refuse accidental overwrite. Each episode emits its own acceptance/provenance report.
mkdir "$OUT"
for episode in {0..29}; do
  in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
    python -u "$CODE/scripts/pva_pose_refine.py" \
    --root /track1 --base /track1/submissions/episodes_cand_P2 \
    --predictions /track1/work/pva/predictions_all --out "/track1/$OUT" --episode "$episode" --steps 100
done
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import shutil
base=Path('submissions/episodes_cand_P2');out=Path('submissions/episodes_cand_PVA_POSE')
assert len(list(out.glob('episode_??????.npz')))==30
for source in base.iterdir():
    target=out/source.name
    if target.exists(): continue
    if source.is_dir(): shutil.copytree(source,target)
    else: shutil.copy2(source,target)
PY
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u "$CODE/scripts/pva_mhr_joints.py" --root /track1 \
  --episodes-dir "/track1/$OUT" --out /track1/work/pva/joints_PVA_POSE --episodes {0..29}
echo PVA_POSE_ALL_BUILT
