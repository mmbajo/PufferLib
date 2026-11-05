#!/bin/bash
# M5 check without the trainer: one lean lockstep world and NBOTS bots driven through the PufferLib
# env (ocean/ragnarok/ragnarok.h, via ragnarok_check), first with random actions for STEPS steps,
# then the scripted policy on each task (TASKS, default all) for 10x as many steps (each run on its own
# accounts). Submit with slurm/submit.sh env-check.sbatch [NBOTS] [STEPS] [ACTION_REPEAT].
#SBATCH --job-name=ro-env-check
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/home/user02198/ragnarok/train/logs/env-check-%j.out
set -uo pipefail
umask 077
NBOTS=${1:-64}
STEPS=${2:-600}
REPEAT=${3:-1}
node=$(hostname -s)
job=$SLURM_JOB_ID
export RATHENA=${RATHENA:-/home/user02198/ragnarok/rathena-train}  # or ~/ragnarok/rathena-re (renewal)
export WORLD_LEAN=1
export WORLD_ACCOUNT_BLOCKS=5  # each env start claims its own block of accounts
export WORLD_MAP_CONF=$'lockstep: yes\nlockstep_debug: yes\nautosave_time: 3600'
RG=${RG:-/home/user02198/ragnarok/PufferLib/ocean/ragnarok}
source $RG/slurm/world.sh
export WORLD_MAPS=${WORLD_MAPS:-$RG_TASK_MAPS}
world_check_client || exit 6
wlog "node=$node job=$job bots=$NBOTS steps=$STEPS action_repeat=$REPEAT"
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
CHECK=$RG/ragnarok_check
export RG_DEBUG=1
status=0
$CHECK $NBOTS $STEPS random mix $REPEAT || status=1
for task in ${TASKS:-0 1 2 3 4 5 6 7 8}; do
    $CHECK $NBOTS $((STEPS * 10)) scripted $task $REPEAT || status=1
done
wlog "env checks exited with $status"
world_alive || wlog "a server process died during the run"
world_stop_all
exit $status
