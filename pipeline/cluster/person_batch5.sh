#!/bin/bash
cd /work/mnidgu2s/project1/repo
source <(sed -n '/^P=/p;/^variant()/,/^}/p' work/person_variants.sh | sed '1s#^#cd /work/mnidgu2s/project1/repo;#')
variant V5 0.55 3e4 a0 0.3
variant V6 0.55 3e4 a0 0.5
variant V7 0.55 3e4 a0 0.3 D 0.3
variant V8 0.55 3e4 a0 0.3 VPQ3 0.3
echo BATCH5BUILT
