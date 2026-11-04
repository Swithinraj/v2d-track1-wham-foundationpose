#!/bin/bash
# Pack a candidate with the official kit, citing the commit that holds its code, and refuse a commit that cannot be
# inspected: organizers read code_commit_url, so it must exist on the remote and contain the scripts that built the file.
# Usage: pack_candidate.sh NAME  -> submissions/track1_cand_NAME.parquet (packs submissions/episodes_cand_NAME)
#   COMMIT: full sha, default HEAD. CANDIDATE_SCRIPTS: scripts the commit must contain.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
NAME="$1"; COMMIT="${COMMIT:-$(git rev-parse HEAD)}"; KIT=official/v2d_submission_kit
OUT="submissions/track1_cand_${NAME}.parquet"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "FAIL: COMMIT must be a full sha, got ${COMMIT}" >&2; exit 2; }
git fetch --quiet origin
git branch -r --contains "${COMMIT}" | grep -q . || { echo "FAIL: ${COMMIT} is not on the remote; push it first" >&2; exit 2; }
for script in ${CANDIDATE_SCRIPTS:-scripts/parquet_episodes.py scripts/true_cam_params.py pipeline/recipes/contact_stage_split.sh}; do
  git cat-file -e "${COMMIT}:${script}" || { echo "FAIL: ${COMMIT} does not contain ${script}" >&2; exit 2; }
done
REMOTE="$(git remote get-url origin)"; REMOTE="${REMOTE%.git}"
(cd "${KIT}" && "${VENV_PYTHON}" eval_reconstruction.py --episodes "${TRACK1_ROOT}/submissions/episodes_cand_${NAME}" \
  --sample data/track_1_sample_submission.parquet --commit "${REMOTE}/commit/${COMMIT}" --out "${TRACK1_ROOT}/${OUT}") | tail -1
echo "PASS: ${OUT} cites ${COMMIT}"
