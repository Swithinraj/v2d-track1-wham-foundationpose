#!/bin/bash
# Matched ablation of V5: change only root-mix smoothing to use actual video time.
set -euo pipefail
cd /work/mnidgu2s/project1/repo
export TMPDIR=/work/mnidgu2s/project1/.tmp OPENBLAS_NUM_THREADS=4
P=.venvs/submission/bin/python
$P scripts/human_root_mix.py --base submissions/episodes_person_V5 \
  --base_meshes work/analysis/meshes/BMQ3KGM --other_meshes work/analysis/meshes/a0 \
  --out submissions/episodes_person_TIMEAXIS --weight 0.3
$P scripts/combine_human_objects.py submissions/episodes_person_TIMEAXIS \
  submissions/episodes_cand_V5 submissions/episodes_cand_TIMEAXIS
echo TIMEAXISBUILT
