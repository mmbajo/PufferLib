#!/bin/bash
# Does a training world change as it ages? One lean lockstep world; the env's scripted policy plays
# the field task for ROUNDS rounds of 6,000 world steps (10 game minutes) each, every round on fresh
# accounts, and each round's perf and monsters in sight are printed. server/npc/rl_census.txt writes
# the map's monster count to the map log every 10 game minutes. Submit with
# slurm/submit.sh world-age.sbatch [NBOTS] [ROUNDS] [TASK].
#SBATCH --job-name=ro-world-age
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/home/user02198/ragnarok/train/logs/world-age-%j.out
set -uo pipefail
umask 077
NBOTS=${1:-64}
ROUNDS=${2:-50}
TASK=${3:-0}
node=$(hostname -s)
job=$SLURM_JOB_ID
export RATHENA=${RATHENA:-/home/user02198/ragnarok/rathena-2025}  # client 20250402, as the play server; rathena-re is 20221005, rathena-train pre-renewal
export WORLD_LEAN=1
export WORLD_ACCOUNT_BLOCKS=$((ROUNDS + 2))
export WORLD_MAP_CONF=$'lockstep: yes\nautosave_time: 3600'
RG=${RG:-/home/user02198/ragnarok/PufferLib/ocean/ragnarok}
source $RG/slurm/world.sh
export WORLD_MAPS=${WORLD_MAPS:-$RG_TASK_MAPS}
world_check_client || exit 6
wlog "node=$node job=$job bots=$NBOTS rounds=$ROUNDS task=$TASK"
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job" | awk -v j=$job '$1 + 0 < j + 0')
if [ -n "$others" ]; then
    wlog "an older job of this account holds $node: $others; leaving"
    exit 3
fi
trap 'world_stop_all; exit 0' TERM INT
world_db_start || { world_stop_all; exit 5; }
world_create 0 $NBOTS || { world_stop_all; exit 5; }
world_start 0 || { world_stop_all; exit 5; }
export RO_WORLD_ROOT=$WORLD_ROOT RO_DB_DIR=$RATHENA/db
status=0
for r in $(seq 1 $ROUNDS); do
    out=$($RG/ragnarok_check $NBOTS 6000 scripted $TASK 2>&1) || status=1
    ep=$(echo "$out" | grep '^episodes' | grep -o 'perf [0-9.]*\|kills [0-9.]*\|monsters in sight [0-9.]*\|players in sight [0-9.]*' | tr '\n' ' ')
    err=$(echo "$out" | grep -o 'bots with errors: [0-9]*')
    wlog "round $r (game minutes $(( (r - 1) * 10 ))-$(( r * 10 ))): $ep; $err"
done
wlog "census (map log):"
grep -ah rl_census $WORLD_ROOT/w0/log/map.out 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -60
world_alive || wlog "a server process died during the run"
world_stop_all
exit $status
