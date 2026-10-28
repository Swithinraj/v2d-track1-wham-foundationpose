#!/bin/bash
# Candidates BMQ3KGMk / BMQ3KGMkR: BMQ3KGM's human unchanged (depth trend alpha 0.75) with the objects re-expressed by the
# gravity kappa of the alpha-0 human (trend.md, "Objects"). Kaggle 10-28: the depth trend off (BMQ3KGMa0) improved CD-O
# but hurt CD-H, so the object part is applied alone here: kappa is computed on the full-mode human without the trend
# (episodes_repro) and with it (episodes_repro_depth), and BMQ3KGM's motion-head input episodes_bmq3k_in has its objects
# re-scaled about the camera by kappa(alpha 0) / kappa(alpha 0.75) per episode (scripts/kappa_rescale_objects.py). The
# later object steps are multiplicative or linear, so this equals building with the re-scaled objects. Then either the
# motion head + contact stage (BMQ3KGMk) or the depth-reliability module (rel_final.sh, BMQ3KGMkR).
# Usage: pipeline/recipes/kcal_final.sh OUTPUT [rel]   (after the BMQ3KGM recipes; "rel" = with the module)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:?output name}"; MODE="${2:-}"; S="${TRACK1_ROOT}/submissions"; BM="${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}"
for d in "${S}/episodes_repro" "${S}/episodes_repro_depth" "${S}/episodes_bmq3k_in"; do require_file "${d}/episode_000029.npz" "build BMQ3KGM first"; done
grep -q '"alpha": 0.75' "${S}/episodes_repro_depth/sam_depth_trend_report.json" || { echo "episodes_repro_depth is not the alpha-0.75 trend" >&2; exit 3; }
rm -rf "${S:?}/episodes_kg_a0" "${S:?}/episodes_kg_a075"
for a in a0:episodes_repro a075:episodes_repro_depth; do
  in_container mhr_converter.sif python -u /track1/scripts/kappa_gravity.py --input "/track1/submissions/${a#*:}" \
    --out "/track1/submissions/episodes_kg_${a%%:*}" --floor /track1/outputs/physics/kappa_floor.json
done
"${VENV_PYTHON}" scripts/kappa_rescale_objects.py --objects "${S}/episodes_bmq3k_in" --kappa_new "${S}/episodes_kg_a0" \
  --kappa_old "${S}/episodes_kg_a075" --out "${S}/episodes_bmq3k_in_k"
if [[ "${MODE}" == rel ]]; then
  bash pipeline/recipes/rel_final.sh episodes_bmq3k_in_k "${OUT}"
else
  EPS_SCALE=0 bash pipeline/recipes/gru_depth_final.sh episodes_bmq3k_in_k episodes_bm_depth mask_visibility_bm "${BM}" "${OUT}"
fi
echo "PASS: BMQ3KGM human + alpha-0 object calibration${MODE:+ + reliability module} -> submissions/${OUT}"
