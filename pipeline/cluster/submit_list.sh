#!/bin/bash
# Usage: submit_list.sh LOG_DONE_WORD  < lines "NAME|message"
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python
while IFS='|' read -r name message; do
  [[ -n "$name" ]] || continue
  echo "=== $name $(date -u +%H:%M)"
  $P scripts/submit_candidate.py "submissions/track1_cand_$name.parquet" -m "$message" 2>&1 | grep -v 'cannot find name' | grep -E 'COMPLETE|ERROR|rror|refus|limit' | cut -c1-120
done
echo "$1"
