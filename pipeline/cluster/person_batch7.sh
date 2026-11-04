#!/bin/bash
cd /work/mnidgu2s/project1/repo
source <(sed -n '/^P=/p;/^variant()/,/^}/p' work/person_variants.sh | sed '1s#^#cd /work/mnidgu2s/project1/repo;#')
TRUST=0.5 variant V9 0.55 1e4 a0 0.3
TRUST=0 variant V10 0.55 1e4 a0 0.3
echo BATCH7BUILT
