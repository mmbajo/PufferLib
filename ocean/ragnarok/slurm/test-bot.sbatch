#!/bin/bash
# M2 check: one throwaway world, NBOTS scripted bots for MINUTES minutes. The first bot is a GM
# and types @die after DIE_AFTER seconds (default 120; -1 for never), to test death and respawn.
# Submit with slurm/submit.sh test-bot.sbatch [NBOTS] [MINUTES] [DIE_AFTER].
#SBATCH --job-name=ro-test-bot
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/home/user02198/ragnarok/train/logs/test-bot-%j.out
set -uo pipefail
umask 077
NBOTS=${1:-4}
MINUTES=${2:-10}
DIE_AFTER=${3:-120}
export WORLD_GM_BOT=1
# MODE=lockstep: the training build (~/ragnarok/rathena-train) with the clock stepped by the bots.
if [ "${MODE:-realtime}" = lockstep ]; then
    export RATHENA=${RATHENA:-/home/user02198/ragnarok/rathena-train}
    export WORLD_MAP_CONF=$'lockstep: yes\nlockstep_debug: yes\nautosave_time: 3600'
    BOT_FLAGS="${BOT_FLAGS:-} --lockstep"
fi
node=$(hostname -s)
job=$SLURM_JOB_ID
RG=${RG:-/home/user02198/ragnarok/PufferLib/ocean/ragnarok}
source $RG/slurm/world.sh
world_check_client || exit 6
wlog "node=$node job=$job mode=${MODE:-realtime} bots=$NBOTS minutes=$MINUTES rathena=$RATHENA $(git -C $RATHENA log --oneline -1 | cut -c1-50)"

# Leave if an older job of this account holds the node (another program's job, or a test that
# landed here first; two tests would also clash on ports). The oldest one stays.
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job" | awk -F_ -v j=$job '$1 + 0 < j + 0')
if [ -n "$others" ]; then
    wlog "an older job of this account holds $node: $others; leaving"
    exit 3
fi
for port in $WORLD_DB_PORT 7000 7001 7002; do
    if ss -ltn | grep -qE ":$port\b"; then
        wlog "port $port is in use on $node; leaving"
        exit 4
    fi
done
trap 'world_stop_all; exit 0' TERM INT

world_db_start || { world_stop_all; exit 5; }
world_create 0 $NBOTS || { world_stop_all; exit 5; }
world_start 0 || { world_stop_all; exit 5; }

$RG/client/scripted_bot --host 127.0.0.1 --port 7000 --accounts $WORLD_ROOT/w0/accounts \
    --db $RATHENA/db --bots $NBOTS --minutes $MINUTES --log-every ${LOG_EVERY:-60} --die-after $DIE_AFTER --verbose ${BOT_FLAGS:-} &
bot=$!
wait $bot
status=$?
wlog "scripted_bot exited with $status"
world_alive || wlog "a server process died during the run"
grep -cE "\[(Error|Warning)\]" $WORLD_ROOT/w0/log/map.out | xargs -I{} echo "map server errors and warnings: {}"
world_stop_all
exit $status
