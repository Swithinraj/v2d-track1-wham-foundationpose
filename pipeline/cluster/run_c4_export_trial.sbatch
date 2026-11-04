#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --job-name=v2d-p1-c4x
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4x_%j.log
source pipeline/env.sh
export APPTAINER_BIND="$PWD/weights/cari4d/lib_mhr_assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets:ro"
L=/workspace/v2d_cari4d/lib/cari4d
for stage in init coconet refined; do
  in_container v2d_cari4d.sif PYTHONPATH=$L:/workspace/v2d_sam3d_body/lib MHR_ASSETS_ROOT=/track1/weights/cari4d/sam3d_body \
    python /track1/scripts/cari4d_export.py --root /track1/work/cari4d/out --episode 28 --stage $stage --out /track1/work/cari4d/export/$stage 2>&1 | tail -1
done
