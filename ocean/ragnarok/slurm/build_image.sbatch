#!/bin/bash
#SBATCH --job-name=ro-puffer-image
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user02198/ragnarok/toolchain/logs/image-%j.out
# Build the PufferLib 5.0 build image: CUDA 12.2 devel (matches the GPU nodes' driver 535,
# which supports CUDA 12.2 at most) on Ubuntu 22.04, plus clang 15, LLVM's libomp, ccache and libGL.
set -euo pipefail
BASE='docker://nvidia/cuda:12.2.2-devel-ubuntu22.04'
OUT=/lustre/users/user02198/ragnarok/images/puffer-cuda12.2-ubuntu22.04.sqsh
node=$(hostname -s)
job=$SLURM_JOB_ID
others=$(squeue -u "$USER" -h -w "$node" -o %i | grep -vx "$job" || true)
if [ -n "$others" ]; then
    echo "another job of this account holds $node: $others; leaving"
    exit 3
fi
export TMPDIR=/tmp/ro-img-$job
export ENROOT_CACHE_PATH=$TMPDIR/cache ENROOT_DATA_PATH=$TMPDIR/data ENROOT_TEMP_PATH=$TMPDIR/tmp
mkdir -p $ENROOT_CACHE_PATH $ENROOT_DATA_PATH $ENROOT_TEMP_PATH
name=ro-puffer-build-$job
cleanup() { enroot remove -f $name > /dev/null 2>&1 || true; rm -rf "$TMPDIR"; }
trap cleanup EXIT
echo "$(date '+%F %T') node=$node job=$job"
df -h /tmp | tail -1
enroot import -o $TMPDIR/base.sqsh "$BASE"
enroot create -n $name $TMPDIR/base.sqsh
enroot start --root --rw $name bash -s <<'INSTALL'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt() { apt-get -o APT::Sandbox::User=root "$@"; }
apt update -qq
apt install -y -qq --no-install-recommends clang-15 libomp-15-dev ccache libgl-dev git curl \
    ca-certificates make build-essential unzip python3 zlib1g-dev libmysqlclient21 libpcre3 > /dev/null
ln -sf /usr/bin/clang-15 /usr/local/bin/clang
# build.sh links -lomp5 on Linux; libomp-15-dev ships libomp5.so, add it only if missing.
ls -l /usr/lib/x86_64-linux-gnu/libomp*.so* 2>/dev/null || true
[ -e /usr/lib/x86_64-linux-gnu/libomp5.so ] || ln -s /usr/lib/llvm-15/lib/libomp.so.5 /usr/lib/x86_64-linux-gnu/libomp5.so
echo /usr/lib/llvm-15/lib > /etc/ld.so.conf.d/llvm-15.conf
ldconfig
echo "libomp5.so soname: $(objdump -p /usr/lib/x86_64-linux-gnu/libomp5.so | awk '/SONAME/{print $2}')"
ldconfig -p | grep -E 'libomp\.so\.5|libnccl\.so|libGL\.so ' || true
ls /usr/include/nccl.h /usr/local/cuda/include/nvml.h
apt-get clean
rm -rf /var/lib/apt/lists/*
clang --version | head -1
/usr/local/cuda/bin/nvcc --version | tail -2
ccache --version | head -1
INSTALL
enroot export -o $TMPDIR/out.sqsh $name
mkdir -p "$(dirname $OUT)"
cp $TMPDIR/out.sqsh $OUT.tmp && mv $OUT.tmp $OUT
ls -l $OUT
sha256sum $OUT
echo "$(date '+%F %T') done"
