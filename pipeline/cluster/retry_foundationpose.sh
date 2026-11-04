#!/bin/bash
# Run inside the existing wr20 allocation. Reuse upstream caches; mark success only
# after the complete episode exits successfully. Failed attempts remain explicit.
set -euo pipefail
cd /work/mnidgu2s/project1/repo
export TMPDIR=/work/mnidgu2s/project1/.tmp
export CARI4D_EXPORTER_PATCH=/work/mnidgu2s/project1/repo/work/cari4d/overrides/run_foundationpose_mhr_export.py
GPU="${1:-1}"
for episode in 12 18 20; do
  if [[ -f "work/cari4d/queue/${episode}.done" ]]; then continue; fi
  if bash pipeline/recipes/cari4d_episode.sh "${episode}" "${GPU}" > "logs/c4_retry_${episode}.log" 2>&1; then
    touch "work/cari4d/queue/${episode}.done"
    rm -f "work/cari4d/queue/${episode}.failed"
    echo "RETRY_DONE ${episode}"
  else
    echo "RETRY_FAILED ${episode}; see logs/c4_retry_${episode}.log"
  fi
done
