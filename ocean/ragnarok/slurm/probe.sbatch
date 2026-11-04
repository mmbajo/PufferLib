#!/bin/bash
#SBATCH --job-name=ro-gpu-probe
#SBATCH --partition=002-partition-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user02198/ragnarok/toolchain/logs/probe-%j.out
# What a GPU node offers for building and running PufferLib 5.0.
set -uo pipefail
node=$(hostname -s)
job=$SLURM_JOB_ID
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job")
if [ -n "$others" ]; then
    echo "another job of this account holds $node: $others; leaving"
    exit 3
fi
section() { echo; echo "== $*"; }
section "node $node job $job"
cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_ID)='
uname -r
section "allocation"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "nproc=$(nproc) of $(grep -c ^processor /proc/cpuinfo)"
free -g | head -2
section "gpu"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvidia-smi | sed -n 1,12p
section "compilers"
for t in nvcc clang clang-18 gcc g++ ccache cmake python3 curl git unzip; do printf '%-9s ' $t; command -v $t || echo "(none)"; done
gcc --version | head -1
ls -d /usr/local/cuda* /opt/nvidia* 2>/dev/null || echo "no /usr/local/cuda*"
ls /usr/include/nccl.h 2>/dev/null || echo "no /usr/include/nccl.h"
section "libraries"
ldconfig -p | grep -E 'libnccl|libnvidia-ml|libomp|libGL\.so|libcuda\.so' | head -20
section "containers"
command -v enroot && enroot version
srun --help 2>&1 | grep -E 'container-image' | head -2 || echo "no pyxis flags in srun"
section "network (HTTP status, 10 s timeout)"
for url in https://github.com https://objects.githubusercontent.com https://nvcr.io/v2/ https://registry-1.docker.io/v2/ https://archive.ubuntu.com/ubuntu/ ; do
    printf '%-40s ' $url
    curl -sI --max-time 10 -o /dev/null -w '%{http_code}\n' $url || echo "failed (exit $?)"
done
section "disk"
df -h /tmp | tail -1
