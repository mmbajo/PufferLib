#!/bin/bash
# M5 check without the trainer: one lean lockstep world and NBOTS bots driven through the PufferLib
# env (ocean/ragnarok/ragnarok.h, via ragnarok_check), first with random actions for STEPS steps,
# then with the scripted policy for 10x as many (on the next block of accounts). Submit with slurm/submit.sh env-check.sbatch [NBOTS] [STEPS].
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
node=$(hostname -s)
job=$SLURM_JOB_ID
export RATHENA=/home/user02198/ragnarok/rathena-train
export WORLD_LEAN=1
export WORLD_ACCOUNT_BLOCKS=2  # each env start claims its own block of accounts
export WORLD_MAP_CONF=$'lockstep: yes\nlockstep_debug: yes\nautosave_time: 3600'
RG=${RG:-/home/user02198/ragnarok/PufferLib/ocean/ragnarok}
source $RG/slurm/world.sh
wlog "node=$node job=$job bots=$NBOTS steps=$STEPS"
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
$CHECK $NBOTS $STEPS random
s1=$?
$CHECK $NBOTS $((STEPS * 10)) scripted
s2=$?
wlog "random exited $s1, scripted exited $s2"
world_alive || wlog "a server process died during the run"
world_stop_all
exit $((s1 + s2))
