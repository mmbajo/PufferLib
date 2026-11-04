#!/bin/bash
#SBATCH --job-name=ro-puffer-smoke
#SBATCH --partition=002-partition-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user02198/ragnarok/toolchain/logs/smoke-%j.out
# Build PufferLib 5.0 (breakout) in the toolchain image and train it briefly.
set -uo pipefail
IMG=/lustre/users/user02198/ragnarok/images/puffer-cuda12.2-ubuntu22.04.sqsh
PUF=/home/user02198/ragnarok/PufferLib
node=$(hostname -s)
job=$SLURM_JOB_ID
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job")
if [ -n "$others" ]; then
    echo "another job of this account holds $node: $others; leaving"
    exit 3
fi
echo "$(date '+%F %T') node=$node job=$job puffer=$(git -C $PUF log --oneline -1)"
srun --container-image=$IMG --container-mounts=$PUF:$PUF --container-workdir=$PUF bash -s <<'RUN'
set -uo pipefail
export CCACHE_DIR=/tmp/ccache
nvidia-smi -L
echo "== native build"
start=$(date +%s)
./build.sh breakout 2>&1 | tail -15
echo "native build exit ${PIPESTATUS[0]} in $(( $(date +%s) - start )) s"
echo "== cpu build"
./build.sh breakout --cpu 2>&1 | tail -5
echo "cpu build exit ${PIPESTATUS[0]}"
echo "== train"
start=$(date +%s)
timeout 300 ./puffer train --base.checkpoint_dir=/tmp/puffer-ckpt --base.log_dir=/tmp/puffer-logs 2>&1 | tail -80
echo "train exit ${PIPESTATUS[0]} in $(( $(date +%s) - start )) s"
RUN
echo "$(date '+%F %T') done"
