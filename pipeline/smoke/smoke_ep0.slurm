#!/bin/bash
#SBATCH --job-name=prep-smoke-01
#SBATCH --account=rwth2243
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep-smoke-%j.out
#SBATCH --error=logs/prep-smoke-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com

# Environment smoke test: run every PIPELINE STAGE TYPE once, on hula_hoop (episodes 0-2),
# and measure per-stage wall time. Purpose: (a) prove the CLAIX environment (containers,
# weights, venvs, apptainer+GPU) executes the real stages, (b) measure per-episode runtimes
# to extrapolate the 30-episode rebuild. NOT a quality test: outputs are checked for
# completeness by the stage scripts themselves.
#
# Stage coverage:
#   1  object_depth.sh (MoGe-2, GPU)          episodes 0-2   (object_mesh needs all three)
#   1  object_masks.sh (GroundingDINO+SAM2)   episodes 0-2
#   1  keypoints2d.sh (MediaPipe, CPU)        episodes 0-2
#   2  person.sh (actor masks, GPU)           episode 0
#   2  object_mesh.sh hula_hoop (SAM3D-Objects + FP fit)  (seeds 1 2)
#   3  sam3d_body.sh (body mode + mask prompt, GPU)       episode 0
#   3  track_object.sh (FoundationPose, GPU)              episode 0
#   4  track_symmetric.sh (hula_hoop is a torus, CPU)     episode 0
# Skipped (need more of the chain; come with the fan-out): object_depth/masks for the other 27
# episodes, track_symmetric for other symmetric objects, human_refine/hand_fix/pack.
#
# Run (repo root on a CLAIX login node):  mkdir -p logs && sbatch pipeline/smoke/smoke_ep0.slurm
# Mutually exclusive clone: the same script is queued on c25g AND c23g under the same job name.
# Whichever starts first claims a lock (atomic mkdir on the shared $HPCWORK) and cancels the
# pending sibling; the sibling that starts second finds the claim held and cancels itself
# (a FAILED/TIMEOUT/PRAMPTED owner releases the claim so the other partition can take over).
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(pwd)}"
export TRACK1_ROOT
source pipeline/env.sh

JOBNAME=$(squeue -j "${SLURM_JOB_ID}" -h -o %j 2>/dev/null || echo "smoke")
LOCK="${TRACK1_ROOT}/logs/claim_$(echo "${JOBNAME}" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p logs
claimed=0
for try in 1 2 3; do
  if mkdir "$LOCK" 2>/dev/null; then echo "${SLURM_JOB_ID}" > "$LOCK/jobid"; claimed=1; break; fi
  owner=$(cat "$LOCK/jobid" 2>/dev/null || true)
  st=$(sacct -j "${owner:-0}" -n -o State 2>/dev/null | head -1 | tr -d ' ' || true)
  if [[ "$st" =~ ^(FAILED|CANCELLED|TIMEOUT|PREEMPTED|OUT_OF) ]]; then
    echo "claim owner $owner ended with $st — taking over the claim" >&2
    rm -rf "$LOCK"; continue
  fi
  echo "sibling smoke job (${owner:-?}, state ${st:-queued}) started first — cancelling this job ${SLURM_JOB_ID}"
  scancel "${SLURM_JOB_ID}" || true
  sleep 60
  exit 0
done
[[ "$claimed" == 1 ]] || { echo "could not claim the lock" >&2; exit 3; }
echo "smoke claim owned by job ${SLURM_JOB_ID} on $(squeue -j "${SLURM_JOB_ID}" -h -o %R 2>/dev/null)"
# cancel any PENDING sibling with the same job name (it would only duplicate this run)
squeue -n "${JOBNAME}" -h -o "%i %T" 2>/dev/null | while read -r sib st; do
  [[ "${sib}" == "${SLURM_JOB_ID}" ]] && continue
  if [[ "${st}" == PENDING* ]]; then scancel "${sib}" 2>/dev/null || true; echo "cancelled pending sibling ${sib}"; fi
done

module load Python/3.12.3 || echo "(module load failed — assuming python3 >= 3.10)"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

t() { local s=$(date +%s); "$@"; local rc=$?; [[ $rc -eq 0 ]] || exit $rc; \
  echo "TIMING stage=$(( $(date +%s) - s ))s :: $*" >&2; }

echo "== [1/4] one-time host setup (venvs; ~5 min)"
[[ -x .venvs/submission/bin/python ]] || t bash pipeline/setup/setup_venv.sh
[[ -x .venvs/mediapipe/bin/python ]] || t bash pipeline/setup/setup_mediapipe.sh

echo "== [2/4] stage-1 tasks, episodes 0-2 (depth, masks, keypoints)"
for ep in 0 1 2; do
  t bash pipeline/stages/object_depth.sh "$ep"
  t bash pipeline/stages/object_masks.sh "$ep"
  t bash pipeline/stages/keypoints2d.sh "$ep"
done

echo "== [3/4] stage-2 tasks (actor masks ep0; hula_hoop mesh)"
t bash pipeline/stages/person.sh 0
t bash pipeline/stages/object_mesh.sh hula_hoop

echo "== [4/4] stage-3/4 tasks for episode 0 (SAM 3D Body, FoundationPose, symmetric)"
t bash pipeline/stages/sam3d_body.sh 0
t bash pipeline/stages/track_object.sh 0
t bash pipeline/stages/track_symmetric.sh 0

echo "== summary"
grep -o "TIMING stage=[0-9]*s :: .*" "${SLURM_JOB_ID:+/dev/stdout}" 2>/dev/null || true
echo "PASS: smoke test complete — all stage types ran on episode group hula_hoop"
