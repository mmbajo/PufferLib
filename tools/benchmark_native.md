# Native training benchmark

Build an environment with `--benchmark` to time Puffer's real synchronous
collector and PPO/Muon updates. The benchmark warms up first, then runs for at
least the requested number of seconds, finishing the last complete update.
It emits one `BENCHMARK {...}` JSON line and does not write checkpoints or run
evaluation. It uses the same native build dependencies as training.

```sh
./build.sh cartpole build/bench_cartpole --benchmark --float
./build/bench_cartpole --seconds=30 --warmup=3

./build.sh decision_cartpole build/bench_decision_cartpole --benchmark
./build/bench_decision_cartpole --policy.bundle=bundles/laya \
    --seconds=30 --warmup=3

# Four simultaneous environments; one update uses all sixteen rollout steps.
./build/bench_decision_cartpole --policy.bundle=bundles/laya \
    --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=4 \
    --train.horizon=4 --train.minibatch_size=16 --seconds=30 --warmup=3
```

Ordinary `--section.key=value` overrides are supported. The benchmark forces
`base.async=0` and disables learning-rate/entropy schedules for duration-based
runs. It requires one GPU, one policy and no self-play; `base.gpu_offset`
selects the CUDA device. `--warmup` counts complete
rollout/update iterations and must be at least one; `--seconds` must be positive.
Use a separate invocation for each trial and keep hardware, build precision,
model, sequence budget, environment, minibatch and replay ratio recorded.

The JSON reports:

- Actual environment steps per elapsed second, including both rollout and
  learning; it is not a count of tokens or replayed training samples.
- Startup and warmup separately, plus timed updates, steps and wall seconds.
- Rollout and learner wall times. Device synchronization completes the last
  update before timing ends. Learning includes backward and optimizer work.
- Existing Puffer profile counters for model inference, environment work,
  copies, training preparation and model/optimizer work. The environment
  counter includes text serialization and tokenization for text adapters.
  With multiple collection buffers these component counters are averaged
  across buffers and are not additive wall-time percentages.
- Allocated policy parameters, including padding and any unused imported
  act/escalate head, and device-wide used GPU memory at the end. Memory is not
  peak memory and can include other processes; use an otherwise idle GPU.
- For imported decision policies, the padded sequence length and sampled live
  observation token lengths. Lengths are sampled after each rollout, rather
  than for every intermediate state.

Accumulated PPO losses must be finite and at least one PPO update must run.
This is a throughput check; use the integration tests for gradient/checkpoint
correctness and separate, seeded learning evaluations to judge task improvement.

Stock CartPole uses a tiny recurrent policy over four numerical features;
text CartPole uses a Transformer over tokens. Their throughput measures useful
system costs but is not a comparison of equivalent architectures or inputs.
The imported Transformer backend currently uses FP32, eager attention with
quadratic workspace, fixed padding to the bundle context limit and no CUDA
graphs. Native C++ alone does not make a large Transformer as fast as Puffer's
small policies. Batching and sequence length must be considered alongside
model size when interpreting results.
