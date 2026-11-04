#!/bin/bash
# Build the training rAthena in ~/ragnarok/rathena-train (branch local/lockstep): the same client
# version as the play server, plus the lockstep patch and room for 8192 connections per server.
# Submit with train/submit.sh build-train.sbatch.
#SBATCH --job-name=ro-build-train
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user02198/ragnarok/train/logs/build-%j.out
set -euo pipefail

node=$(hostname -s)
echo "node=$node job=$SLURM_JOB_ID start=$(date '+%F %T')"
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$SLURM_JOB_ID" || true)
if [ -n "$others" ]; then
    echo "another job of this account holds $node: $others; leaving"
    exit 3
fi

export TMPDIR=/tmp/ro-build-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
LOGS=/home/user02198/ragnarok/train/logs

cd /home/user02198/ragnarok/rathena-train
echo "rathena-train $(git log --oneline -1) + $(git status --short | wc -l) changed files"
./configure --enable-epoll --enable-packetver=20130618 --enable-prere=yes --with-maxconn=8192 \
    > $LOGS/configure-$SLURM_JOB_ID.log 2>&1
grep -E 'PACKETVER|prere|epoll|maxconn|MAXCONN' $LOGS/configure-$SLURM_JOB_ID.log | tail -6
make clean > /dev/null 2>&1
mkdir -p 3rdparty/rapidyaml/obj/src/c4/yml 3rdparty/rapidyaml/obj/ext/c4core/src/c4
if ! make server -j"$SLURM_CPUS_PER_TASK" > $LOGS/make-$SLURM_JOB_ID.log 2>&1; then
    echo "make failed:"
    grep -E "error|Error" $LOGS/make-$SLURM_JOB_ID.log | head -20
    exit 1
fi
grep -ciE '\bwarning\b' $LOGS/make-$SLURM_JOB_ID.log || true
ls -l login-server char-server map-server
echo "end=$(date '+%F %T')"
