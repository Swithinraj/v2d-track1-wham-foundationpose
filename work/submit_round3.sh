#!/bin/bash
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python
submit() { echo "=== $1 $(date -u +%H:%M)"; $P scripts/submit_candidate.py "submissions/track1_cand_$1.parquet" -m "$2" 2>&1 | grep -v 'cannot find name' | grep -E 'COMPLETE|ERROR|rror|refus|limit' | cut -c1-120; }
submit OTa "OTa af466e2: measured-camera person with floor slide strength 0.5; objects keep their BMQ3KGM placement relative to the person, floor scale with 2 cm offset; no contact stage (test of CD-H and CD-O)"
submit OTb "OTb af466e2: measured-camera person with floor slide strength 0.7; objects keep their BMQ3KGM placement relative to the person, floor scale with 4 cm offset; no contact stage (test of CD-H and CD-O)"
echo ROUND3DONE
