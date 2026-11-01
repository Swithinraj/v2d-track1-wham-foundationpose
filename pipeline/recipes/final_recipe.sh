#!/bin/bash
# From the released videos to the final episode files of the file selected on Kaggle. The submission container's
# `run.sh reproduce` runs this, and docker/collect_container.py refuses to build while the marker line below remains.
# Usage: final_recipe.sh OUT_DIR [EPISODE...]    (no episodes: all 30)
#   Writes OUT_DIR/episode_XXXXXX.npz and OUT_DIR/episode_XXXXXX_object.glb.
# It must be the chain that built the selected candidate, starting from the videos. Candidates built from a scored
# Parquet (floor.md) reach this script by rebuilding that Parquet's chain first; the file itself is not an input here.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
OUT="$1"; shift
RUN_EPISODES="$*" bash "${TRACK1_ROOT}/pipeline/run_all.sh"
# FINAL_RECIPE_NOT_WRITTEN: replace these two lines with the selected candidate's steps, then copy its episodes to ${OUT}.
echo "final_recipe.sh: the selected candidate's steps are not written yet" >&2; exit 3
