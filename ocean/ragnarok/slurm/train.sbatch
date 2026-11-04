#!/bin/bash
# Train the ragnarok env on one GPU. NWORLDS lean lockstep worlds with NBOTS bots each run on this
# node; the PufferLib trainer runs in the toolchain container (one env per world, so total_agents =
# NWORLDS x NBOTS). Checkpoints and trainer logs go to /lustre/users/user02198/ragnarok/runs/<job>.
#
# Submit with slurm/submit.sh [-t TIME] train.sbatch [NWORLDS] [NBOTS] [TIMESTEPS] [RUNS_FILE].
# RUNS_FILE (optional) lists one run per line as extra ./puffer train arguments; the runs go one
# after another, each into runs/<job>/<n>. Training always runs on 002-partition-default with a
# limit over an hour (your rule, 2026-11-05).
#SBATCH --job-name=ro-train
#SBATCH --partition=002-partition-default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/user02198/ragnarok/train/logs/train-%j.out
set -uo pipefail
umask 077
NWORLDS=${1:-16}
NBOTS=${2:-64}
TIMESTEPS=${3:-400000000}
RUNS_FILE=${4:-}
node=$(hostname -s)
job=$SLURM_JOB_ID
export RATHENA=/home/user02198/ragnarok/rathena-train
export WORLD_LEAN=1
export WORLD_MAPS=${WORLD_MAPS:-"prt_fild08 pay_fild01 pay_fild02"}  # every task's map
export WORLD_MAP_CONF=$'lockstep: yes\nautosave_time: 3600'
# Ports and the database port per job, so two training jobs can share a node.
export WORLD_PORT_BASE=$((10000 + (job % 150) * 200))
export WORLD_DB_PORT=$((40000 + job % 10000))
nruns=1
[ -n "$RUNS_FILE" ] && nruns=$(grep -cvE '^\s*(#|$)' "$RUNS_FILE")
# Every env start claims a fresh block of accounts per world (see ocean/ragnarok/ragnarok.h), and a
# run starts its envs twice (training, then the final evaluation).
export WORLD_ACCOUNT_BLOCKS=$((3 * nruns + 2))
PUF=${PUF:-/home/user02198/ragnarok/PufferLib}  # a frozen worktree can be passed in
export RG=$PUF/ocean/ragnarok
source $RG/slurm/world.sh
IMG=/lustre/users/user02198/ragnarok/images/puffer-cuda12.2-ubuntu22.04.sqsh
RUN=/lustre/users/user02198/ragnarok/runs/$job
wlog "node=$node job=$job worlds=$NWORLDS bots=$NBOTS timesteps=$TIMESTEPS runs=$nruns ports=$WORLD_PORT_BASE db=$WORLD_DB_PORT gpu=${CUDA_VISIBLE_DEVICES:-?}"
wlog "rathena-train $(git -C $RATHENA log --oneline -1 | cut -c1-60); PufferLib $(git -C $PUF log --oneline -1 | cut -c1-60)"

# Leave if an older job of another program holds the node (our own ro-* jobs use other ports).
others=$(squeue -u "$USER" -h -w "$node" -o '%i %j' | awk -v j=$job '$1 + 0 < j + 0 && $2 !~ /^ro-/ {print $1}')
if [ -n "$others" ]; then
    wlog "an older job of this account holds $node: $others; leaving"
    exit 3
fi
if ss -ltn | grep -qE ":($WORLD_DB_PORT|$WORLD_PORT_BASE)\b"; then
    wlog "ports $WORLD_PORT_BASE or $WORLD_DB_PORT are in use on $node; leaving"
    exit 4
fi
if ldd $RATHENA/map-server | grep -q "not found" && LD_LIBRARY_PATH=$RO/lib ldd $RATHENA/map-server | grep "not found"; then
    wlog "map-server is missing libraries on $node"
    exit 5
fi
trap 'world_stop_all; exit 0' TERM INT

world_db_start || { world_stop_all; exit 5; }
t0=$SECONDS
for w in $(seq 0 $((NWORLDS - 1))); do
    world_create $w $NBOTS > /dev/null || { world_stop_all; exit 5; }
    world_start $w > /dev/null || { wlog "world $w did not start"; world_stop_all; exit 5; }
done
wlog "$NWORLDS worlds up in $((SECONDS - t0))s"

mkdir -p $RUN
if [ -n "$RUNS_FILE" ]; then
    grep -vE '^\s*(#|$)' "$RUNS_FILE" > $RUN/runs.txt
else
    echo "" > $RUN/runs.txt
fi
export RO_WORLD_ROOT=$WORLD_ROOT RO_DB_DIR=$RATHENA/db RO_WORLD_PORT_BASE=$WORLD_PORT_BASE
srun --container-image=$IMG \
    --container-mounts=/home/user02198/ragnarok:/home/user02198/ragnarok,/lustre/users/user02198/ragnarok:/lustre/users/user02198/ragnarok,$WORLD_ROOT:$WORLD_ROOT \
    --container-workdir=$PUF bash -s <<RUN
export CCACHE_DIR=/tmp/ccache-$job
./build.sh ragnarok > $RUN/build.log 2>&1 || { tail -20 $RUN/build.log; exit 1; }
n=0
while IFS= read -r args; do
    n=\$((n + 1))
    mkdir -p $RUN/\$n
    echo "\$(date '+%F %T') run \$n: \$args" | tee $RUN/\$n/args.txt
    ./puffer train --vec.total_agents=$((NWORLDS * NBOTS)) --vec.num_threads=$NWORLDS \
        --env.num_agents=$NBOTS --train.total_timesteps=$TIMESTEPS \
        --base.checkpoint_dir=$RUN/\$n/checkpoints --base.log_dir=$RUN/\$n/logs \$args \
        > $RUN/\$n/train.log 2>&1
    echo "\$(date '+%F %T') run \$n exited with \$?"
done < $RUN/runs.txt
RUN
status=$?
wlog "trainer runs done, status $status"
world_alive || wlog "a server process died during the runs"
world_stop_all
exit $status
