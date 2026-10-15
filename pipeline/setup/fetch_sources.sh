#!/bin/bash
# Clone every upstream source at the commit pinned in config/upstream.env and unpack the
# official submission kit. Idempotent: existing checkouts are verified, not re-cloned.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

clone_pinned() {  # clone_pinned URL COMMIT DIR
  local url="$1" commit="$2" dir="${TRACK1_ROOT}/third_party/$3"
  if [[ ! -d "${dir}/.git" ]]; then
    git clone --quiet "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --quiet origin "${commit}" 2>/dev/null || true
  git -C "${dir}" checkout --quiet "${commit}"
  git -C "${dir}" submodule update --init --recursive --quiet
  [[ "$(git -C "${dir}" rev-parse HEAD)" == "${commit}" ]] || { echo "$3 is not at ${commit}" >&2; exit 4; }
  echo "PASS: third_party/$3 @ ${commit:0:12}"
}

mkdir -p "${TRACK1_ROOT}/third_party" "${TRACK1_ROOT}/official"
clone_pinned "${V2D_GIT_URL}" "${V2D_GIT_COMMIT}" video_to_data
clone_pinned "${WHAM_GIT_URL}" "${WHAM_GIT_COMMIT}" WHAM
clone_pinned "${MHR_GIT_URL}" "${MHR_GIT_COMMIT}" MHR
clone_pinned "${TRELLIS_GIT_URL}" "${TRELLIS_GIT_COMMIT}" TRELLIS

# DPVO's setup.py includes Eigen from thirdparty/ (WHAM's README installs it the same way).
DPVO="${TRACK1_ROOT}/third_party/WHAM/third-party/DPVO"
if [[ ! -s "${DPVO}/thirdparty/eigen-3.4.0/Eigen/Core" ]]; then
  mkdir -p "${DPVO}/thirdparty"
  curl -fsSL "${EIGEN_URL}" -o "${DPVO}/thirdparty/eigen-3.4.0.zip"
  unzip -q -o "${DPVO}/thirdparty/eigen-3.4.0.zip" -d "${DPVO}/thirdparty"
  rm "${DPVO}/thirdparty/eigen-3.4.0.zip"
fi
echo "PASS: Eigen 3.4.0 headers for DPVO"

KIT_ZIP="${TRACK1_ROOT}/third_party/video_to_data/docs/v2d_challenge/assets/v2d_submission_kit.zip"
if [[ ! -f "${TRACK1_ROOT}/official/v2d_submission_kit/KIT_MANIFEST.json" ]]; then
  unzip -q "${KIT_ZIP}" -d "${TRACK1_ROOT}/official"
fi
echo "PASS: official submission kit in official/v2d_submission_kit"
