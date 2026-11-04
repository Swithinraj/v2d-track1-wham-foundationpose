#!/bin/bash
# Runs on the login node independently of any assistant session. It uploads a FIXED list of files, then exits.
# To stop it: pkill -f autopilot[.]sh
cd /work/mnidgu2s/project1/repo
export TMPDIR=/work/mnidgu2s/project1/.tmp
L=../logs/autopilot.log
say() { echo "[$(date -u +%m-%d\ %H:%M)] $*" >> $L; }
FREEZE=$(date -u -d '2026-11-04 21:50' +%s)     # stop waiting just before the freeze: November 4, 22:00 UTC
waitfor() { while ! eval "$1"; do [ "$(date -u +%s)" -ge "$FREEZE" ] && { say "gave up waiting: $1"; exit 0; }; sleep 60; done; }
pack_upload() { # NAME MESSAGE
  CANDIDATE_SCRIPTS="scripts/cari4d_export.py scripts/cari4d_episodes.py scripts/cari4d_camroot.py scripts/mesh_smooth.py scripts/object_transfer.py pipeline/recipes/cari4d_candidate.sh" bash pipeline/recipes/pack_candidate.sh "$1" >> $L 2>&1 && printf '%s|%s\n' "$1" "$2" | ./work/submit_list.sh "$1 uploaded" >> $L 2>&1
}
say started
waitfor "grep -q ROUND7DONE ../logs/kaggle_round7.log 2>/dev/null"
say "rounds 6 and 7 finished"
waitfor "grep -q CANDIDATESBUILT logs/c4cand1.log 2>/dev/null"
pack_upload C1 "C1: CARI4D CoCoNet person, mesh smoothing lam 300, measured camera, floor slide 0.55 on the first finished episodes, V3 person elsewhere; OTa objects; no contact stage"
pack_upload C2 "C2: CARI4D refined person, mesh smoothing lam 30, measured camera, floor slide 0.55 on the first finished episodes, V3 person elsewhere; OTa objects; no contact stage"
waitfor "grep -q FINALBUILT logs/c4final.log 2>/dev/null"
pack_upload C3 "C3: CARI4D CoCoNet person on all finished episodes, mesh smoothing lam 300, measured camera, floor slide 0.55; OTa objects; no contact stage"
pack_upload C4 "C4: CARI4D CoCoNet person and its own objects on all finished episodes, mesh smoothing lam 300, measured camera, floor slide 0.55; no contact stage"
say finished
