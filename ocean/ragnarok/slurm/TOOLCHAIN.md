# PufferLib 5.0 toolchain on sr1

Set up 2026-11-05 for M1 of `../TRAINING_PLAN.md`. PufferLib is the upstream 5.0 clone at
`~/ragnarok/PufferLib`, on branch `local/ragnarok` from `ea7f9893`.

## Image

`/lustre/users/user02198/ragnarok/images/puffer-cuda12.2-ubuntu22.04.sqsh`, 7.4 GB.
SHA-256 `33823b874538b482a5a2c22dea1eaa95565583cf346e474cb9c38200d7679b78`.

- **Base:** `nvidia/cuda:12.2.2-devel-ubuntu22.04`, which brings nvcc 12.2, NCCL and NVML headers.
- **Added:** clang 15.0.7 (as `clang`), libomp 15, ccache 4.5.1, libGL, git, curl, make, unzip and
  python3.
- **libomp:** `libomp-15-dev` already ships `libomp5.so` (soname `libomp.so.5`), which is what
  `build.sh` links with `-lomp5`.
- **Why CUDA 12.2 on Ubuntu 22.04:** the GPU nodes' driver (535.161.08) supports CUDA 12.2 at
  most, and NVIDIA has no 12.2 image for Ubuntu 24.04.
- **Rebuild:** `sbatch --exclude=<this account's nodes> build_image.sbatch` builds it again in about
  3 minutes. It runs on a CPU node, on node-local disk, and writes the image to the path above.

## Build and train

`smoke.sbatch` is the template. Its core is:

```
srun --container-image=$IMG --container-mounts=$PUF:$PUF --container-workdir=$PUF bash -s <<'RUN'
export CCACHE_DIR=/tmp/ccache
./build.sh breakout          # trainer -> ./puffer (nvcc -arch=native: build on the GPU node)
./build.sh breakout --cpu    # play/eval binary -> ./breakout
./puffer train --base.checkpoint_dir=/tmp/puffer-ckpt --base.log_dir=/tmp/puffer-logs
RUN
```

- `build.sh` calls `ccache nvcc` unconditionally, so ccache has to be in the image.
- It downloads raylib 5.5 into the clone on its first run (gitignored). The nodes reach GitHub, so
  that works inside the job.

Result, job 5004267 on srdgx00255:
- the trainer builds in 7 s, and the `--cpu` build works;
- breakout's default 55M steps trained in about 4 s at 14.2M steps/s, reaching perf 0.97 (score
  about 838);
- the whole job took 29 s.

## GPU nodes

- **Hardware:** srdgx nodes have 8 × H100 80GB HBM3, 224 CPUs and about 2 TB of memory. They run
  Ubuntu 22.04 with driver 535.161.08 (CUDA 12.2).
- **Tools:** `/usr/local/cuda-12.2` exists, but nvcc isn't on PATH. There's no clang, no ccache and
  no `nccl.h`. gcc is 11.4. enroot is 3.4.1, and pyxis (`srun --container-image`) works.
- **Job shape:** `--gres=gpu:1 -c 64 --mem=128G` gets exactly 64 CPUs, 128 GB and one GPU.
- **Network:** compute nodes reach github.com, Docker Hub, nvcr.io and the Ubuntu mirrors.
- **Partitions**, from `sbatch --test-only` for 1 GPU and 64 CPUs on 2026-11-05 at 00:22:

| Partition | Limit | A 1-GPU job would start |
|---|---|---|
| `002-partition-ext` | none | Now (srdgx00191–00200, 3 idle) |
| `002-partition-debug` | 3 h | Within about 2 h (srdgx00251–00255; the smoke test started at once) |
| `002-partition-default` | none | Estimated 2057, so effectively full |
| `002-partition-RAD` | none | Estimated 2028 |
| `002-partition-all`, `002-partition-all2` | | Not allowed for this account |

## What went wrong

The first image build (job 5004260) stopped on my own `libomp5.so` symlink step, because the
package already provides that file. I fixed the step in `build_image.sbatch` and job 5004261
built the image. Nothing else failed.
