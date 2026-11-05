#!/bin/bash
# M4 check: NWORLDS lean lockstep worlds side by side, each with NBOTS scripted bots that reset
# their episode with @rlreset every RESET game seconds, for MINUTES game minutes.
# Each world is one map server plus one bot process, so give the job 2 CPUs per world and a few
# more, e.g. slurm/submit.sh -c 20 bench-worlds.sbatch 8 64 10 600.
#SBATCH --job-name=ro-bench
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/home/user02198/ragnarok/train/logs/bench-%j.out
set -uo pipefail
umask 077
NWORLDS=${1:-8}
NBOTS=${2:-64}
MINUTES=${3:-10}
RESET=${4:-600}
node=$(hostname -s)
job=$SLURM_JOB_ID
export RATHENA=${RATHENA:-/home/user02198/ragnarok/rathena-2025}  # client 20250402, as the play server; rathena-re is 20221005, rathena-train pre-renewal
export WORLD_LEAN=1
export WORLD_MAP_CONF=$'lockstep: yes\nautosave_time: 3600'
RG=${RG:-/home/user02198/ragnarok/PufferLib/ocean/ragnarok}
source $RG/slurm/world.sh
world_check_client || exit 6
wlog "node=$node job=$job cpus=$SLURM_CPUS_PER_TASK worlds=$NWORLDS bots=$NBOTS minutes=$MINUTES reset=$RESET $(basename $RATHENA) $(git -C $RATHENA log --oneline -1 | cut -c1-50)"

others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job" | awk -v j=$job '$1 + 0 < j + 0')
if [ -n "$others" ]; then
    wlog "an older job of this account holds $node: $others; leaving"
    exit 3
fi
if ss -ltn | grep -qE ":($WORLD_DB_PORT|70[0-9][0-9]|7[1-6][0-9][0-9])\b"; then
    wlog "world ports are in use on $node; leaving"
    exit 4
fi
trap 'world_stop_all; exit 0' TERM INT

world_db_start || { world_stop_all; exit 5; }
t0=$SECONDS
for w in $(seq 0 $((NWORLDS - 1))); do
    world_create $w $NBOTS || { world_stop_all; exit 5; }
    world_start $w || { world_stop_all; exit 5; }
done
wlog "$NWORLDS worlds up in $((SECONDS - t0))s; map server memory $(ps -o rss= -C map-server | awk '{s += $1} END {printf "%.0f MB", s / 1024}')"

pids=()
for w in $(seq 0 $((NWORLDS - 1))); do
    $RG/client/scripted_bot --host 127.0.0.1 --port $((7000 + 10 * w)) --accounts $WORLD_ROOT/w$w/accounts \
        --db $RATHENA/db --bots $NBOTS --minutes $MINUTES --log-every 120 --reset-every $RESET --lockstep \
        > $WORLD_LOGS/bench-$job-w$w.log 2>&1 &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
    wait $pid || failed=$((failed + 1))
done
wlog "bot processes done, $failed failed"
for w in $(seq 0 $((NWORLDS - 1))); do
    echo "w$w: $(grep -E '^clock' $WORLD_LOGS/bench-$job-w$w.log) | $(grep -E '^summary' $WORLD_LOGS/bench-$job-w$w.log | cut -d: -f2)"
done
grep -h '^clock' $WORLD_LOGS/bench-$job-w*.log | awk '{for (i = 1; i <= NF; i++) if ($i == "bot") s += $(i - 1)} END {printf "total: %.0f bot steps/s over %d worlds\n", s, NR}'
grep -h "^reset" $WORLD_LOGS/bench-$job-w0.log | head -3
world_alive || wlog "a server process died during the run"
world_stop_all
exit $failed
