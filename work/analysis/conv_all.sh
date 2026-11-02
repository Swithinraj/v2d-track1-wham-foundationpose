#!/bin/bash
# Episode directories for each downloaded submission.
P=.venvs/submission/bin/python
while read name src; do
  d=submissions/episodes_base_$name
  [[ -d $d ]] || [[ ! -f $src ]] || $P scripts/parquet_episodes.py "$src" "$d" 2>&1 | grep -v 'cannot find'
done <<LIST
a0 submissions/kaggle_archive/57739575/57739575.raw
VPQ3 submissions/kaggle_archive/57524497/57524497.raw
nowham submissions/kaggle_archive/57321552/57321552.raw
BMQO submissions/kaggle_archive/57590842/57590842.raw
D submissions/kaggle_archive/57625539/57625539.raw
R2 submissions/kaggle_archive/57751281/57751281.raw
Afr submissions/kaggle_archive/57379489/57379489.raw
rdL100 submissions/kaggle_archive/57391725/57391725.raw
lam300 submissions/kaggle_archive/57272500/57272500.raw
h2t submissions/kaggle_archive/57238829/57238829.raw
BMQ submissions/kaggle_archive/57531195/57531195.raw
k submissions/kaggle_archive/57745353/57745353.raw
ktcp submissions/track1_sam3d_kappa_tcp_real.parquet
abl submissions/track1_sam3d_ablation_no_objfix.parquet
LIST
