#!/bin/bash
#SBATCH --job-name=feet-refine
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/feet-refine-%j.out
#SBATCH --error=logs/feet-refine-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com

# Floor-consistent human placement (improvement #2): re-run the human refinement on the SAME
# inputs as the baseline chain (episodes_sam_depth -> refine), with HUMAN_FEET planting the
# feet on the MoGe-depth-RANSAC floor planes (scripts/floor_planes_soles.py output). The foot
# gates are widened (foot-near 0.06 m, foot-shift 0.12 m) because the fresh chain's humans
# float 3-14 cm above the detected floor in ~28/30 episodes (sole-gap report); the stillness
# gate (5 mm/frame, >=5 frames) still protects mid-step feet. Output is a SEPARATE refine dir
# for offline comparison; nothing is packed or uploaded from this job.
# Submit: sbatch pipeline/fanout/feet_refine.slurm
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh
module load Python/3.12.3 || echo "(module load failed)"
FLOORS="outputs/physics/floor_planes_soles.json"
[[ -s "${FLOORS}" ]] || { echo "missing ${FLOORS} — run floor_planes_soles.py first"; exit 1; }
REFINE_IN="episodes_sam_depth"
[[ -s "submissions/${REFINE_IN}/episode_000029.npz" ]] || { echo "missing ${REFINE_IN} — run the assembly chain first"; exit 1; }

rm -rf "submissions/episodes_sam_feet_refine" "outputs/human_refine_sam_feet"
env REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam \
  HUMAN_REFINE_WORK="human_refine_sam_feet" \
  HUMAN_FEET="${FLOORS}" HUMAN_FOOT_NEAR=0.06 HUMAN_FOOT_MAX_SHIFT=0.12 \
  bash pipeline/stages/human_refine.sh "${REFINE_IN}" "${REFINE_IN}" "episodes_sam" "episodes_sam_feet_refine"

echo "== feet-refine done; planted counts (human_refit_report of the feet-applied pass):"
.venvs/submission/bin/python - <<'EOF'
import json
r = json.load(open("submissions/episodes_sam_depth_fit2/human_refit_report.json"))["episodes"]
tot = 0
for tag, v in sorted(r.items()):
    pl = v.get("planted_frames") or []
    print(f"{tag}: planted {pl} | vertex_move {v.get('vertex_move_mm'):.1f} mm")
    tot += sum(pl)
print("total planted frames (both feet):", tot)
EOF
echo "PASS: feet refine complete — submissions/episodes_sam_feet_refine (offline compare next)"
