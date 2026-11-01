#!/usr/bin/env bash
# Entry point of the Track 1 submission container, from the kit's examples/container/run.sh with its settings block and
# reconstruct_episodes() filled in. See CONTAINER_SUBMISSION.md in the kit.
#
# Usage:
#   run.sh verify [verify_container.py options]
#       check every hash and the commit in /submission/submission.json against the container
#   run.sh reproduce --label LABEL --output DIR [--dataset-root DIR] [--episodes LIST] [--parallel N]
#       Tracks 2 and 3: roll out each episode's policy with the evaluator and pack the submission
#       Track 1: run the reconstruction pipeline on the episodes and pack the submission
#       With --episodes the run covers only those episodes and does not pack.
#   run.sh pack --label LABEL --output DIR
#       Track 1: pack the stored final reconstructions into the submission Parquet
#
# The challenge dataset is mounted read-only at /data and results are written to /output:
#   docker run --rm --gpus all --network none -v DATASET:/data:ro -v OUT:/output IMAGE \
#       /submission/run.sh reproduce --label tier1 --dataset-root /data --output /output
set -euo pipefail

# ---- settings: adapt to your image -------------------------------------------------------------
SUBMISSION="${SUBMISSION:-/submission}"
MANIFEST="${SUBMISSION}/submission.json"
# Python with the submission kit's requirements (pandas, pyarrow, ...).
KIT_PYTHON="${KIT_PYTHON:-/opt/venvs/submission/bin/python}"
# The kit that holds tools/verify_container.py. It can be newer than the kit in submission.json,
# which is the copy that packed and uploaded your submissions.
VERIFIER_KIT="${VERIFIER_KIT:-${SUBMISSION}/verifier_kit}"
# Python with your FlashCHORD runtime (Tracks 2 and 3).
EVAL_PYTHON="${EVAL_PYTHON:-python3}"
# The FlashCHORD checkout the evaluator runs from (robot assets are resolved relative to it).
RUNTIME_ROOT="${RUNTIME_ROOT:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export HOME="${HOME:-/tmp}"; [[ -w "${HOME}" ]] || export HOME=/tmp
# -------------------------------------------------------------------------------------------------

# Track 1: reconstruct the listed episodes from the videos under $DATASET_ROOT into $1 as
# episode_XXXXXX.npz plus meshes, in the layout tools/pack_reconstruction.py reads.
# The pipeline writes under its root, so it runs in a work directory next to $1 that links everything read-only from
# /submission. Stages call Apptainer, which is the proot shim here (docker/apptainer_shim.sh).
reconstruct_episodes() {
    local out="$1"; shift
    local work name
    work="$(dirname "${out}")/work"
    mkdir -p "${work}/official" "${work}/.venvs"
    for name in config docs pipeline scripts; do ln -sfn "${SUBMISSION}/code/${name}" "${work}/${name}"; done
    for name in containers weights third_party inputs; do ln -sfn "${SUBMISSION}/${name}" "${work}/${name}"; done
    ln -sfn "${SUBMISSION}/kit" "${work}/official/v2d_submission_kit"
    ln -sfn "${DATASET_ROOT}" "${work}/data"
    for name in submission mediapipe; do ln -sfn "/opt/venvs/${name}" "${work}/.venvs/${name}"; done
    TRACK1_ROOT="${work}" TRACK1_EXTRA_BINDS="${SUBMISSION} ${DATASET_ROOT}" \
        APPTAINER_CACHEDIR="${work}/.apptainer" APPTAINER_TMPDIR="${work}/.apptainer" \
        bash "${SUBMISSION}/code/pipeline/recipes/final_recipe.sh" "${out}" "$@"
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

# Read one field of a submission entry (or of the manifest with LABEL empty), resolved to a path
# when it names a file.
field() {
    "${KIT_PYTHON}" - "${MANIFEST}" "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
manifest, label, key = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = json.loads(manifest.read_text())
node = data if not label else next((s for s in data["submissions"] if s.get("label") == label), None)
if node is None:
    sys.exit(f"no submission labelled {label!r} in {manifest}")
for part in key.split("."):
    node = node.get(part) if isinstance(node, dict) else None
if node is None:
    print("")
elif isinstance(node, (dict, list)):
    print(json.dumps(node))
elif key in ("track", "code.commit_url"):
    print(node)
else:
    path = Path(str(node))
    print(path if path.is_absolute() else manifest.parent / path)
EOF
}

COMMAND="${1:-}"; shift || true
LABEL=""; OUTPUT=""; DATASET_ROOT=""; EPISODES=""; PARALLEL=1
if [[ "${COMMAND}" == "verify" ]]; then
    exec "${KIT_PYTHON}" "${VERIFIER_KIT}/tools/verify_container.py" --manifest "${MANIFEST}" "$@"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "${COMMAND}" == "reproduce" || "${COMMAND}" == "pack" ]] || { usage >&2; exit 2; }
[[ -n "${LABEL}" && -n "${OUTPUT}" ]] || { echo "--label and --output are required" >&2; exit 2; }

TRACK="$(field "${LABEL}" track)"
KIT="$(field "${LABEL}" kit)"; KIT="${KIT:-$(field "" kit)}"; KIT="${KIT:-${SUBMISSION}/kit}"
COMMIT_URL="$(field "" code.commit_url)"
SUBMITTED="$(field "${LABEL}" file)"
OUT="${OUTPUT}/${LABEL}"
mkdir -p "${OUT}"
PACKED="${OUT}/$(basename "${SUBMITTED}")"

if [[ "${TRACK}" == "track_1" ]]; then
    RECON="$(field "${LABEL}" outputs)"
    if [[ "${COMMAND}" == "reproduce" ]]; then
        [[ -n "${DATASET_ROOT}" ]] || { echo "--dataset-root is required to reconstruct" >&2; exit 2; }
        RECON="${OUT}/reconstructions"
        mkdir -p "${RECON}"
        DATASET_ROOT="${DATASET_ROOT}" reconstruct_episodes "${RECON}" ${EPISODES//,/ }
        [[ -z "${EPISODES}" ]] || { echo "reconstructed episodes ${EPISODES} into ${RECON}"; exit 0; }
    fi
    cd "${KIT}"
    "${KIT_PYTHON}" eval_reconstruction.py --episodes "${RECON}" \
        --sample data/track_1_sample_submission.parquet --commit "${COMMIT_URL}" --out "${PACKED}"
    echo "packed ${PACKED}"
    exit 0
fi

[[ "${COMMAND}" == "reproduce" ]] || { echo "pack applies to Track 1; use reproduce" >&2; exit 2; }
EVAL_SCRIPT="$(field "${LABEL}" evaluator.script)"
ROLLOUTS="${OUT}/rollouts"; LOGS="${OUT}/logs"
mkdir -p "${ROLLOUTS}" "${LOGS}"

# One checkpoint map per shard, from the manifest's per-episode checkpoints.
"${KIT_PYTHON}" - "${MANIFEST}" "${LABEL}" "${EPISODES}" "${PARALLEL}" "${LOGS}" <<'EOF'
import json, sys
from pathlib import Path
manifest, label, episodes, parallel, logs = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
entry = next(s for s in json.loads(manifest.read_text())["submissions"] if s["label"] == label)
paths = {int(k): (Path(v) if Path(v).is_absolute() else manifest.parent / v) for k, v in entry["checkpoints"].items()}
wanted = sorted(paths) if not episodes else [int(e) for e in episodes.split(",")]
missing = [e for e in wanted if e not in paths]
if missing:
    sys.exit(f"no checkpoint for episode(s) {missing}")
for index in range(min(parallel, len(wanted))):
    shard = {str(e): str(paths[e]) for e in wanted[index::parallel]}
    (logs / f"checkpoints_shard{index}.json").write_text(json.dumps(shard, indent=1))
EOF

RUNNER=("${EVAL_PYTHON}" "${KIT}/examples/run_policy_evaluation.py" --eval-script "${EVAL_SCRIPT}" --world-count 1)
[[ -z "${RUNTIME_ROOT}" ]] || RUNNER+=(--runtime-root "${RUNTIME_ROOT}")
PIDS=()
for SHARD in "${LOGS}"/checkpoints_shard*.json; do
    NAME="$(basename "${SHARD}" .json)"
    "${RUNNER[@]}" --checkpoints "${SHARD}" --results "${ROLLOUTS}" > "${LOGS}/${NAME#checkpoints_}.log" 2>&1 &
    PIDS+=($!)
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "${PID}" || STATUS=1; done
[[ "${STATUS}" == 0 ]] || { echo "evaluation failed; see ${LOGS}/shard*.log" >&2; exit 1; }
echo "rollouts in ${ROLLOUTS}"
[[ -z "${EPISODES}" ]] || exit 0

PACK=("${KIT_PYTHON}" tools/pack_submission.py --track "${TRACK}" --results "${ROLLOUTS}"
      --code-commit "${COMMIT_URL}" --eval-script "${EVAL_SCRIPT}" -o "${PACKED}")
if [[ "${TRACK}" == track_2_* ]]; then
    [[ -n "${DATASET_ROOT}" ]] || { echo "--dataset-root is required to pack Track 2" >&2; exit 2; }
    PACK+=(--dataset-root "${DATASET_ROOT}")
fi
FRAME_REFERENCES="$(field "${LABEL}" frame_references)"
[[ -z "${FRAME_REFERENCES}" ]] || PACK+=(--frame-references "${FRAME_REFERENCES}")
MESHES="$(field "${LABEL}" meshes)"
[[ -z "${MESHES}" ]] || PACK+=(--meshes "${MESHES}")
cd "${KIT}"
"${PACK[@]}" 2>&1 | tee "${LOGS}/pack.log"
echo "packed ${PACKED}"
