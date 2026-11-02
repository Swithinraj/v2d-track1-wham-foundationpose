#!/bin/bash
# Waits for the unlimited window, then uploads the first round through the guarded uploader.
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python
while [ "$(date -u +%s)" -lt "$(date -u -d '2026-11-01 22:00:30' +%s)" ]; do sleep 20; done
submit() { echo "=== $1 $(date -u +%H:%M)"; $P scripts/submit_candidate.py "submissions/track1_cand_$1.parquet" -m "$2" 2>&1 | grep -v 'cannot find name' | tail -9 | cut -c1-170; }
submit MF "MF 13ff13a: BMQ3KGM file + measured camera + person floor-contact slide (lam 1e4, anchor fixed) + per-object floor scale about the camera (17 objects) + contact stage"
submit G97 "G97 13ff13a: BMQ3KGM file + measured camera + every object scaled 0.97 about the camera + contact stage"
submit WF "WF 13ff13a: BMQ3KGM file + measured camera + root mix 0.25 toward WHAM h2t (anchor fixed) + per-object floor scale, others 0.974 + contact stage"
echo ROUND1DONE
