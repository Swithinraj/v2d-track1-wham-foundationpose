#!/bin/bash
cd /work/mnidgu2s/project1/repo
while [ "$(date -u +%s)" -lt "$(date -u -d '2026-11-02 00:00:40' +%s)" ]; do sleep 20; done
C=$(git rev-parse --short=7 HEAD)
./work/submit_list.sh ROUND6DONE <<L
P1|P1: mesh smoothing lam 100, floor slide 0.55; a0's objects with floor scale 2 cm, carried to the person; no contact stage
P2|P2: mesh ensemble with the Oct 31 human, floor slide 0.55; objects floor scale 2 cm plus relative size among table episodes, carried; no contact stage
P3|P3: mesh ensemble, then mesh smoothing lam 100, floor slide 0.55; OTa objects carried; no contact stage
V5|V5: measured-camera person, floor slide 0.55 lam 30000, root mix 0.3 toward a0; OTa objects; no contact stage
V6|V6: measured-camera person, floor slide 0.55 lam 30000, root mix 0.5 toward a0; OTa objects; no contact stage
S30|S30: person and object motion sharpened 0.3 at lam 100, floor slide 0.55; objects floor scale 2 cm, carried; no contact stage (ACC test)
L
