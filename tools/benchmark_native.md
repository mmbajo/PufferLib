# Native training benchmark

See [native distributed training](distributed_training.md) for optimizer
ownership, training flags, checkpoint semantics and larger-model limits.

Build an environment with `--benchmark` to time Puffer's real synchronous
collector and PPO/Muon updates on one machine. The benchmark warms up first,
then runs a fixed number of complete rollout/update iterations or for at least
the requested duration. It emits one aggregate `BENCHMARK {...}` JSON line and
does not write checkpoints or run evaluation.

```sh
./build.sh cartpole build/bench_cartpole --benchmark --float
./build/bench_cartpole --seconds=30 --warmup=3

./build.sh decision_cartpole build/bench_decision_cartpole --benchmark
./build/bench_decision_cartpole --policy.bundle=bundles/laya \
    --seconds=30 --warmup=3

# Two replicas, four environments on each GPU: global rollout/minibatch = 32.
./build/bench_decision_cartpole --policy.bundle=bundles/laya \
    --train.gpus=2 --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=4 \
    --train.horizon=4 --train.minibatch_size=16 --updates=50 --warmup=3 \
    --check-replicas=1

# Same workload, distributing whole-matrix optimizer updates and momentum.
./build/bench_decision_cartpole --policy.bundle=bundles/laya \
    --train.gpus=2 --train.distributed_optimizer=1 \
    --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=4 \
    --train.horizon=4 --train.minibatch_size=16 --updates=50 --warmup=3 \
    --check-replicas=1
```

Ordinary `--section.key=value` overrides are supported. The benchmark forces
`base.async=0` and disables learning-rate/entropy schedules. It requires one
policy and no self-play. `train.gpus` selects the replica count; `base.gpu_offset`
selects the beginning of a contiguous block of visible CUDA devices. As in the
training launcher, rank 0 uses the last device in the block and rank 1 starts
at the first. Devices must be on the same machine. The parent forks every rank
before initializing CUDA; it supervises failures and terminates peers if any
rank fails, including when another rank is blocked in NCCL.

`vec.total_agents`, `vec.num_buffers`, `vec.num_threads` and
`train.minibatch_size` are **per rank**. For example, two GPUs with four agents,
a four-step horizon and minibatch 16 collect 32 global steps per iteration and
use a global minibatch of 32. Each rank holds a full model and gradient replica;
`train.distributed_optimizer=1` distributes whole-parameter momentum and Muon
updates. It is not tensor, context or pipeline parallelism.

Use `--updates=N` for exactly N timed rollout/update iterations on every rank.
One such iteration may contain several optimizer steps, depending on minibatch
size and replay ratio. Alternatively, `--seconds=N` finishes the current update
when rank 0 reaches the duration, broadcasting its stop decision to every rank.
The coordination cost is included in elapsed time and separately reported.
Independent duration loops would risk mismatched NCCL calls and are not used.
`--updates` and `--seconds` are mutually exclusive; the default is ten seconds.
`--warmup` counts complete iterations and must be at least one.

To evaluate scaling, hold model, precision, environment, sequence budget,
horizon and replay ratio constant, and measure both:

- Fixed **per-rank batch**: retain agents and minibatch per rank when adding GPUs.
  Global batch and collected steps per iteration increase with GPU count.
- Fixed **global batch**: divide agents and minibatch per rank by GPU count,
  preserving valid horizon/minibatch divisibility. Global work per iteration
  stays constant. Use the same `--updates` for these trials.

Use a separate invocation for each trial. A fixed global batch does not promise
bitwise-equivalent learning trajectories: environment seeds and sampling are
rank specific, and PPO computes some statistics locally. Throughput scaling is
not evidence that a larger global batch improves learning.

The JSON reports:

- `steps` and `global_steps`: the sum of actual environment steps across ranks,
  excluding warmup. `steps_per_second` divides this sum by the **maximum rank
  elapsed time**. This is not a count of tokens or replayed training samples.
- `agents`, `buffers`, `threads`, `minibatch`: per-rank configuration, preserving
  the original single-GPU fields. `global_agents`, `global_rollout_batch` and
  `global_minibatch` make aggregate batch sizes explicit.
- Startup and warmup separately, plus timed updates, steps and wall seconds.
  All ranks complete warmup before the timed region begins. Startup excludes
  process-fork overhead; optional replica checks are timed separately.
- Rollout and learner wall times. Learning includes backward, distributed
  communication and optimizer work. CUDA synchronization completes the final
  update before timing ends. Aggregate component times use the maximum across
  ranks **for each component**, so they do not necessarily add to total elapsed.
- Existing Puffer profile counters for inference, environment work, copies,
  training preparation and model/optimizer work. The environment counter
  includes serialization/tokenization for text adapters. With multiple
  collection buffers these counters are averaged across buffers and are not
  additive wall-time percentages. Distributed aggregate counters again use
  the maximum of each counter across ranks.
- Allocated policy parameters, including padding and any unused imported
  act/escalate head. `device_used_gib` is the maximum end-of-trial device-wide
  memory across ranks. The `ranks` array records each rank's GPU, memory, steps,
  timings and logical optimizer momentum bytes. Memory is **not peak memory**
  and can include other processes; use otherwise idle GPUs. Optimizer scratch
  is reported separately as element counts, since precision differs by build.
- `optimizer_momentum_bytes_max_rank`, `optimizer_momentum_bytes_total` and the
  replicated baseline bytes per rank distinguish actual logical momentum
  storage from a replicated optimizer. Whole-matrix ownership can be uneven.
- For imported decision policies, padded sequence length and sampled live
  observation token lengths aggregated across ranks. Lengths are sampled after
  each rollout, rather than for every intermediate state.

Accumulated PPO losses must be finite on every rank, and at least one update
must run. `--check-replicas=1` additionally compares every FP32 master-weight
byte against rank 0, including padding, before warmup and after the timed loop,
and checks weight finiteness. This uses bounded 4 MiB device scratch and runs
outside the throughput timing. A mismatch or failed rank produces a nonzero
exit status and no aggregate success result. This is a throughput and replica
consistency check; use integration tests for gradient/checkpoint correctness
and separate, seeded learning evaluations to judge task improvement.

Stock CartPole uses a tiny recurrent policy over four numerical features;
text CartPole uses a Transformer over tokens. Their throughput measures useful
system costs but is not a comparison of equivalent architectures or inputs.
The imported Transformer backend currently uses FP32, eager attention with
quadratic workspace, fixed padding to the bundle context limit and no CUDA
graphs. Native C++ alone does not make a large Transformer as fast as Puffer's
small policies. Batching and sequence length must be considered alongside
model size when interpreting results.
