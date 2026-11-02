#!/bin/bash
# Official weight download inside the CARI4D container. The token stays in the environment, never on a command line.
cd /work/mnidgu2s/project1/repo
source pipeline/env.sh
export APPTAINERENV_HF_TOKEN="$(cat ../v2d_track1_handoff_2026-10-31/tokens/huggingface.txt)"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p weights/cari4d
in_container v2d_cari4d.sif python -m v2d.cari4d.lib.download_weights --output_dir /track1/weights/cari4d
echo "WEIGHTS_EXIT=$?"
