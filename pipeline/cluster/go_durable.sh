#!/bin/bash
# Restarts the running uploads and builds so they no longer depend on the assistant session.
cd /work/mnidgu2s/project1/repo
export TMPDIR=/work/mnidgu2s/project1/.tmp
for p in $(pgrep -u "$(id -un)" -x srun); do
  c=$(tr '\0' ' ' </proc/$p/cmdline)
  case "$c" in *cari4d_candidate*) kill "$p" 2>/dev/null; echo "stopped session-bound build step $p";; esac
done
sleep 3
EP="$(cat work/cari4d/first_episodes.txt)"
cat > work/autopilot.sh <<E
#!/bin/bash
# Runs on the login node independently of any assistant session. It uploads a FIXED list of files, then exits.
# To stop it: pkill -f autopilot[.]sh
cd /work/mnidgu2s/project1/repo
export TMPDIR=/work/mnidgu2s/project1/.tmp
L=../logs/autopilot.log
say() { echo "[\$(date -u +%m-%d\ %H:%M)] \$*" >> \$L; }
FREEZE=\$(date -u -d '2026-11-04 21:50' +%s)     # stop waiting just before the freeze: November 4, 22:00 UTC
waitfor() { while ! eval "\$1"; do [ "\$(date -u +%s)" -ge "\$FREEZE" ] && { say "gave up waiting: \$1"; exit 0; }; sleep 60; done; }
pack_upload() { # NAME MESSAGE
  CANDIDATE_SCRIPTS="scripts/cari4d_export.py scripts/cari4d_episodes.py scripts/cari4d_camroot.py scripts/mesh_smooth.py scripts/object_transfer.py pipeline/recipes/cari4d_candidate.sh" bash pipeline/recipes/pack_candidate.sh "\$1" >> \$L 2>&1 && printf '%s|%s\n' "\$1" "\$2" | ./work/submit_list.sh "\$1 uploaded" >> \$L 2>&1
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
E
chmod +x work/autopilot.sh
cat > work/run_c4_final.sbatch <<'E'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=v2d-p1-c4f
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4final.log
export TMPDIR=/work/mnidgu2s/project1/.tmp CONTACT_GPU=0
bash pipeline/recipes/cari4d_candidate.sh C3 coconet 300 0.55 && OWN_OBJECTS=1 bash pipeline/recipes/cari4d_candidate.sh C4 coconet 300 0.55 && echo FINALBUILT
E
cp work/autopilot.sh work/run_c4_final.sbatch work/go_durable.sh pipeline/cluster/
sbatch --dependency=afterany:434328 work/run_c4_final.sbatch | tail -1
setsid nohup bash work/submit_round6.sh > ../logs/kaggle_round6.log 2>&1 < /dev/null &
setsid nohup bash work/submit_round7.sh > ../logs/kaggle_round7.log 2>&1 < /dev/null &
setsid nohup srun --jobid=434328 --overlap --ntasks=1 --cpus-per-task=10 --mem=80G bash -c "export TMPDIR=$TMPDIR; export EPISODES='$EP' CONTACT_GPU=1; bash pipeline/recipes/cari4d_candidate.sh C1 coconet 300 0.55 && bash pipeline/recipes/cari4d_candidate.sh C2 refined 30 0.55 && echo CANDIDATESBUILT" > logs/c4cand1.log 2>&1 < /dev/null &
setsid nohup bash work/autopilot.sh > /dev/null 2>&1 < /dev/null &
sleep 3
echo "detached processes started"
