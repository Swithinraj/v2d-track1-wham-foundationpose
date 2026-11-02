#!/bin/bash
cd /work/mnidgu2s/project1/repo
while ! grep -q ROUND6DONE ../logs/kaggle_round6.log 2>/dev/null; do sleep 20; done
./work/submit_list.sh ROUND7DONE <<L
V9|V9: floor slide 0.55 lam 10000, reference height half trusted at the anchor, root mix 0.3 toward a0; OTa objects; no contact stage
V10|V10: floor slide 0.55 lam 10000, reference height from the median stance only, root mix 0.3 toward a0; OTa objects; no contact stage
V7|V7: measured-camera person, floor slide 0.55 lam 30000, root mix 0.3 toward a0 and 0.3 toward D; OTa objects; no contact stage
V8|V8: measured-camera person, floor slide 0.55 lam 30000, root mix 0.3 toward a0 and 0.3 toward VPQ3; OTa objects; no contact stage
L
