#!/bin/bash
# Penetration to zero (called by pack.sh with PEN_ZERO=1, after the hand fix): dense hand vertices of the episodes
# (mhr_bodies.py --hands-only, float64, GPU) and scripts/pen_zero.py, which pushes the object out of every hand
# vertex on every scored frame with smoothed offsets (the human is untouched).
# Usage: pen_zero.sh INPUT OUTPUT   (episode directory names under submissions/; PEN_ZERO_WORKERS, default 8)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
IN="$1"; OUT="$2"
H="outputs/pen_zero/${IN}_hands"
in_container mhr_converter.sif python /track1/scripts/mhr_bodies.py --episodes_dir "/track1/submissions/${IN}" \
  --out "/track1/${H}" --hands-only --precision float64
"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/pen_zero.py" --input "${TRACK1_ROOT}/submissions/${IN}" \
  --hands "${TRACK1_ROOT}/${H}" --out "${TRACK1_ROOT}/submissions/${OUT}" --workers "${PEN_ZERO_WORKERS:-8}"
require_file "${TRACK1_ROOT}/submissions/${OUT}/pen_zero_report.json"
echo "PASS: penetration cleared ${IN} -> ${OUT}"
