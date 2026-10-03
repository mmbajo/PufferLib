# Native distributed Transformer training

The native trainer supports **single-machine data parallelism**: each GPU holds
one complete decision model, collects its own environments, and contributes
PPO gradients to an NCCL average. `train.distributed_optimizer=1` additionally
distributes whole-parameter Muon updates and momentum storage across these
replicas. Parameters, gradients and forward/backward computation remain
replicated. Tensor, context and pipeline parallelism are not implemented.

The imported decision policies use the same collector, PPO objective and Muon
optimizer as ordinary Puffer training. They currently run synchronously in FP32
with CUDA graphs disabled. See the [benchmark guide](benchmark_native.md) for
steady-state throughput measurements that exclude startup, evaluation and
checkpoint writes.

## Running on multiple GPUs

Import a supported BERT, ModernBERT or full Laya bundle as described in the
[decision model guide](../ocean/decision_laya/README.md), then build the desired
environment. These examples use the full question/options decision interface:

```sh
./build.sh decision_cartpole build/puffer_decision_cartpole

# Two data-parallel replicas with replicated Muon state.
./build/puffer_decision_cartpole train --policy.bundle=bundles/laya \
    --train.gpus=2 --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=4 \
    --train.horizon=4 --train.minibatch_size=16 \
    --train.total_timesteps=1024 --base.eval_episodes=0

# Same rollout/minibatch configuration; shard whole-tensor momentum and updates.
./build/puffer_decision_cartpole train --policy.bundle=bundles/laya \
    --train.gpus=2 --train.distributed_optimizer=1 \
    --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=4 \
    --train.horizon=4 --train.minibatch_size=16 \
    --train.total_timesteps=1024 --base.eval_episodes=0
```

Relevant options:

| Option | Meaning |
| --- | --- |
| `train.gpus` | Number of GPUs/ranks on this machine; default 1. |
| `base.gpu_offset` | First device in a contiguous block of visible CUDA devices; default 0. Rank 0 uses the last device in the block. |
| `train.distributed_optimizer` | Default 0. Set 1 to distribute whole-tensor Muon ownership, including momentum and optimizer work. |
| `vec.total_agents`, `vec.num_buffers`, `vec.num_threads` | Collection capacity and CPU worker settings **per rank**. Reserve sufficient host CPUs for all ranks. |
| `train.minibatch_size` | Samples **per rank per optimizer update**. All ranks use the same value. |
| `train.horizon` | Steps per local agent collected per rollout. |
| `train.total_timesteps` | Requested **global** environment steps across all ranks. |
| `base.seed` | Shared model initialization seed; also keys distinct distributed environment/action sampling streams. |
| `base.async=0`, `base.cudagraphs=-1` | Required by the imported decision policies and the distributed optimizer. |

The distributed optimizer currently requires FP32. Decision policy builds
select FP32 automatically; ordinary environment builds require `--float` and
explicit synchronous/no-graph settings to use this optimizer mode. It can also
run with one GPU, in which case every parameter belongs to that rank.

The launcher forks all workers before any parent CUDA initialization and passes
the NCCL identifier through a pipe. The parent supervises rank failures,
terminates blocked peers when a rank fails, and waits for all results before
releasing the worker processes. This also keeps completed ranks alive while
rank 0 writes artifacts or performs optional final evaluation. No multi-node
launcher, rendezvous protocol or elastic recovery is provided.

## Batches, seeds and reproducibility

For `W` GPUs, `A` agents per rank, horizon `H` and local minibatch `M`:

- Global collection batch per rollout is `W × A × H` environment steps.
- Global minibatch per optimizer update is `W × M` samples.
- Optimizer updates per rollout follow the existing replay-ratio calculation,
  based on local `A × H` and `M`; increasing GPU count alone does not add local
  optimizer updates.
- Training performs complete rollouts: actual global steps are
  `floor(total_timesteps / (W × A × H)) × (W × A × H)`.
  Choose a budget that includes at least one full rollout.

With horizon 4 and replay ratio 1, for example:

| GPUs | Agents/rank | Minibatch/rank | Global rollout batch | Global minibatch |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 32 | 32 | 32 |
| 2 | 4 | 16 | 32 | 32 |
| 2 | 8 | 32 | 64 | 64 |

The first two rows hold the global batch fixed. The last row holds per-rank work
fixed relative to the first and doubles the global batch. These are different
scaling experiments, and may have different learning behavior.

Model initialization uses the same base seed on every rank. Standard CPU
environment initialization in distributed runs mixes that seed with a distinct
global environment index, avoiding duplicated initial rank-local seeds. This
does not change custom `MY_VEC_INIT` or GPU environment initialization, nor does
it guarantee disjoint environment RNG streams throughout every episode: each
environment controls how its RNG state advances. The single-GPU environment
seeding convention is preserved. Action sampling uses
distinct `(rank, collector buffer)` seed assignments and separate per-agent
subsequences; this also prevents seed collisions between different ranks'
collector buffers.

Changing GPU count, environment layout or batch size can change trajectories
and learning results even with the same base seed. PPO advantage normalization
and other local minibatch calculations are not replaced with global-batch
operations. Data-parallel gradient averaging is therefore not a promise of
bitwise equality with a one-GPU PPO run over concatenated samples. Likewise,
a faster run or a finite loss is not evidence of improved policy quality.

## What the optimizer distributes

Muon orthogonalizes complete parameter matrices. The implementation assigns
each registered tensor to one owner rank, using deterministic largest-first
placement to balance parameter counts. An owner stores that tensor's FP32
momentum and computes its entire update; vectors and padding entries are
assigned as complete registered tensors too.

Each optimizer step:

1. All ranks compute local gradients and average the complete gradient buffer
   through NCCL.
2. Each rank computes the norm of the same averaged gradient buffer, preserving
   the existing global gradient-clipping rule.
3. Each owner applies Nesterov momentum and the existing whole-matrix
   Newton–Schulz calculation to its tensors.
4. NCCL broadcasts each completed update from its owner. Every rank then
   applies the same update to its full parameter replica.

This preserves whole-matrix Muon semantics. Independently orthogonalizing
pieces of a tensor-parallel matrix would be a different optimizer. Ownership
balances element counts, not measured computation time; differently shaped
matrices can require different work. Update broadcasts introduce additional
communication after the existing gradient all-reduce. Speedup depends on how
much optimizer work and memory are saved relative to that communication.

## Logs and checkpoints

Rank 0 determines logging cadence so all ranks enter logging collectives in the
same order. Training environment metrics sum episode accumulators across ranks
before dividing by the total completed-episode count. Loss accumulators are
combined and normalized by their combined minibatch count. Global agent steps
and throughput account for every rank; profile component times use the maximum
for each component across ranks and are not additive percentages.

Logs include `distributed/world_size`, `distributed/global_minibatch_size` and
`distributed/optimizer_sharded`. The ordinary utilization fields describe rank
0's device; use the benchmark's `ranks` array to inspect every device and the
maximum end-of-trial memory.

For ordinary one-policy training, rank 0 writes the log and checkpoint. The
checkpoint filename uses **global** environment steps, matching the logged
step count. For example, a two-GPU run with 512 steps on each rank now writes
`0000000000001024.bin`; the earlier naming used the rank-local count
`0000000000000512.bin`. Single-GPU names are unchanged.

The checkpoint remains a flat FP32 weight file and can be evaluated on one GPU
with the same imported bundle and environment architecture. It does not contain
Muon momentum, RNG streams, environment state or a training cursor; it is not an
exact distributed-training resume artifact. The optimizer ownership layout is
not encoded in the weight file.

Large models should usually disable automatic final evaluation with
`--base.eval_episodes=0` and evaluate the saved checkpoint in a fresh process:

```sh
./build/puffer_decision_cartpole eval path/to/0000000000001024.bin \
    --headless --policy.bundle=bundles/laya --base.eval_episodes=32
```

The current trainer retains its CUDA allocations until process exit. Automatic
final evaluation creates a new evaluator in the rank-0 process, so it can need
substantially more memory than training alone. A separate evaluation process
releases training memory first.

## Larger-model limits and next steps

For an FP32 model with `P` allocated parameter elements, including padding,
replicated weights, gradients and Muon momentum require approximately `12 × P`
bytes per GPU before activations, optimizer scratch and runtime allocations.
With balanced momentum ownership across `W` ranks, this becomes approximately
`8 × P + 4 × P / W` bytes per GPU. Whole tensors are indivisible, so actual
ownership can differ from this ideal. The largest tensor remains a lower bound
on one rank's momentum allocation. Optimizer scratch is sized to each owner's
largest required workspaces and can also be uneven.

As an arithmetic illustration, a 7-billion-element FP32 model needs about
56 GB (decimal) per GPU for just the still-replicated weights and gradients.
At eight ranks, ideally balanced momentum adds another 3.5 GB per GPU. These
figures exclude all activations and scratch and do **not** establish that such
a model is supported or fits on an 80 GB GPU.

Flat optimizer and master-weight conversion counts use 64-bit indexing, and
optimizer planning checks matrix dimensions before cuBLAS calls. This removes
specific 32-bit parameter-count narrowing paths; it is not validation of every
backend operation at multi-billion-parameter sizes. cuBLAS matrix dimensions,
batch dimensions and other kernel/workspace limits still apply. The importer
also accepts only the implemented model families/configurations, rather than
arbitrary Transformer architectures.

The current decision backend uses eager FP32 attention, quadratic attention
workspace and fixed padding to the bundle context budget. The packed adapter
supports at most 2,048 tokens. There is no activation recomputation, fused
attention or parameter/gradient sharding. These limits can dominate well before
optimizer momentum does.

The next implementation priorities are:

1. **Improve each GPU's execution:** BF16 Tensor Core computation with suitable
   FP32 state, fused attention, shorter actual-length execution or bucketing,
   and activation recomputation. Gradient accumulation can increase effective
   training batch without allocating all microbatch activations together.
2. **Extend state sharding:** partition gradients and then parameters with
   explicit gather/release scheduling and resumable distributed checkpoints.
   Parameter gathering must work for both rollout inference and learning.
3. **Add tensor parallelism, with sequence parallelism where useful:** partition
   layer matrices and activation storage. Preserve full-matrix Muon updates
   through a deliberate distributed algorithm or an explicitly different
   optimizer; do not apply Muon independently to matrix shards by accident.
4. **Add context parallelism for long histories:** partition sequence work and
   communicate attention state while preserving bidirectional/local masks.
   This is more relevant once observations or histories outgrow current short
   contexts. Pipeline parallelism becomes useful for sufficiently deep models
   and enough microbatches to occupy the stages.

These are future capabilities, not flags implemented by this change. Use the
[benchmark's replica checks](benchmark_native.md) and the distributed integration
tests to validate a configuration, and run separate seeded evaluations before
making claims about policy learning or comparisons across GPU counts.

## Validation commands

The launcher suite runs on the CPU. The optimizer suite requires one or two
CUDA devices as selected below; the policy integration suite requires two.
Use a small imported BERT fixture for the policy suite, which allocates an
additional model to check gradients against a concatenated-batch reference.

```sh
tests/build_native_distributed_test.sh
./build/test_native_distributed

tests/build_distributed_muon_test.sh
./build/test_distributed_muon 1
./build/test_distributed_muon 2

tests/build_distributed_tests.sh
./build/test_decision_distributed path/to/small-bert-bundle build/distributed-validation
```

The GPU integration checks gradient averaging, optimizer parity, exact replica
agreement, finite PPO updates, weighted global metrics, distinct initial seeds,
checkpoint restoration, and failed-rank cleanup. The optimizer suite additionally
covers rectangular matrices, vectors, empty owners and host-only layout/count
arithmetic beyond `INT_MAX`, without allocating a multi-billion-element model.
