#!/bin/bash
# Compile check for the ragnarok env inside the toolchain container on a CPU node: the CUDA
# trainer for H100 (sm_90; -arch=native needs a GPU) and the --cpu play/eval binary.
#SBATCH --job-name=ro-build-env
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user02198/ragnarok/train/logs/build-env-%j.out
set -uo pipefail
node=$(hostname -s)
job=$SLURM_JOB_ID
echo "node=$node job=$job start=$(date '+%F %T')"
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job" | awk -v j=$job '$1 + 0 < j + 0')
[ -n "$others" ] && { echo "an older job of this account holds $node: $others; leaving"; exit 3; }
PUF=${PUF:-/home/user02198/ragnarok/PufferLib}
IMG=/lustre/users/user02198/ragnarok/images/puffer-cuda12.2-ubuntu22.04.sqsh
srun --container-image=$IMG --container-mounts=/home/user02198/ragnarok:/home/user02198/ragnarok \
    --container-workdir=$PUF bash -s <<RUN
export CCACHE_DIR=/tmp/ccache-$job
NVCC_ARCH=sm_90 ./build.sh ragnarok 2>&1 | tail -15 && ls -l puffer
./build.sh ragnarok --cpu 2>&1 | tail -15 && ls -l ragnarok
RUN
echo "end=$(date '+%F %T')"
