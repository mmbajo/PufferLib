#!/bin/bash
# Submit one of these job scripts, keeping off the nodes this account already uses.
# Usage: slurm/submit.sh [sbatch options] SCRIPT [ARGS...]. Job logs go to ~/ragnarok/train/logs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /home/user02198/ragnarok/train/logs
excl=$(squeue -u "$USER" -h -o %N | paste -sd, -)
sbatch ${excl:+--exclude="$excl"} "$@"
