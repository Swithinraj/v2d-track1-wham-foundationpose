#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-c4-one
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/c4_coconet_one_%j.log
# CARI4D through its CoCoNet stage on the 11 episodes the cancelled batch never reached. Adding CoCoNet's person to the mesh
# ensemble on 19 episodes cut CD-H by 0.22 (P4C), so the rest should gain too. The upstream runner has no stop-after
# flag; each run is stopped once 06_coconet has completed (refinement and render are not used; reruns reuse finished
# stages). One GPU and ONE episode per job (EPISODES=N): stopping the container also ends its job, so a job holds one episode. Codex's repaired FoundationPose exporter (undefined diagnostics as NaN) is bound for every run.
set -uo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export CARI4D_EXPORTER_PATCH=/work/mnidgu2s/project1/repo/work/cari4d/overrides/run_foundationpose_mhr_export.py
source pipeline/env.sh
trap 'echo "C4_11_EXIT time=$(date -u +%FT%TZ)"' EXIT
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
echo "C4_11_START node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
mkdir -p work/cari4d/coconet_done
worker() { # GPU EPISODES...
  local gpu="$1"; shift
  for e in "$@"; do
    [[ -f work/cari4d/coconet_done/$e ]] && continue
    local log=logs/c4_cc_$e.log
    local tag; tag=$(episode_tag "$e")
    bash pipeline/recipes/cari4d_episode.sh "$e" "$gpu" >> "$log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      # Stop only this episode's inference (its own video path), never a process group shared with other workers.
      if grep -q 'CARI4D_STAGE_COMPLETED 06_coconet' "$log"; then sleep 5; pkill -TERM -f "inputs/${tag}/${tag}.0.color.mp4"; break; fi
      sleep 10
    done
    wait "$pid" 2>/dev/null
    if grep -q 'CARI4D_STAGE_COMPLETED 06_coconet' "$log"; then touch work/cari4d/coconet_done/$e; echo "COCONET_DONE $e"
    else echo "COCONET_FAILED $e; see $log"; fi
  done
}
worker "${GPUS[0]}" ${EPISODES}
echo C4_11_FINISHED
