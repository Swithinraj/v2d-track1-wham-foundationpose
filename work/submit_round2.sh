#!/bin/bash
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python
submit() { echo "=== $1 $(date -u +%H:%M)"; $P scripts/submit_candidate.py "submissions/track1_cand_$1.parquet" -m "$2" 2>&1 | grep -v 'cannot find name' | grep -E 'COMPLETE|ERROR|rror|refus|limit' | cut -c1-120; }
submit ML3 "ML3 1280cc0: BMQ3KGM file + measured camera + person floor slide lam 3000, objects unchanged + contact stage"
submit FB15 "FB15 1280cc0: BMQ3KGM file + measured camera + person floor slide strength 1.5 + object floor scale with 2 cm floor offset + contact stage"
submit E2S "E2S 1280cc0: mesh ensemble 0.5 BMQ3KGM + 0.5 Oct 31 full-mode human, measured camera + object translation smoothing lam 3000 + contact stage"
submit R100S "R100S 1280cc0: BMQ3KGM file + measured camera + mesh smoothing lam 100 refit + object translation smoothing lam 10000 + contact stage"
echo ROUND2DONE
