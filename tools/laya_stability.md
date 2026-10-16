# Laya warm-start stability

`laya_stability.py` writes commands for six model-only warm starts from one chosen
competent checkpoint: Muon learning rates `1e-4` and `1e-5`, seeds `73`, `100073`
and `200073`, and 8,192 new global steps. Each uses four GPUs, four agents per
rank, horizon 16, minibatch 16, replay 1, 384 tokens and the coordinate Snake
prompt. The checkpoint restores the trained critic; optimizer, counters and RNG
start fresh. This is not full-state continuation.

```sh
python3 tools/laya_stability.py prepare \
  --working-directory=/path/to/PufferLib \
  --trainer=/path/to/puffer_laya --evaluator=/path/to/evaluate_snake \
  --bundle=/path/to/laya-native --initial-weights=/path/to/competent.bin \
  --training-precision=fp32 --output-dir=build/stability-fp32
```

Repeat with `--training-precision=tf32` and a separate output directory to compare
training arithmetic. Each output directory must be new. The tool does not launch
commands or contact a scheduler. `protocol.json` contains argv, cwd, stdout and
stderr for training and evaluation. The native binaries validate their inputs
when loaded. Record the code branch/commit, hardware, exit status and process
wall time in the executor; run only in an authorized allocation.

Each initial/final policy has **one 512-episode evaluation command**. All use
strict FP32, batch 16, cap 500, model-initialization seed 73 and environment/action
seed bases 6,140,001/6,240,001. Keep the same initial checkpoint for both precisions.
Evaluate only the final 8,192-step checkpoint.

After every command succeeds, analyze either one protocol or both together:

```sh
python3 tools/laya_stability.py analyze \
  build/stability-fp32/protocol.json build/stability-tf32/protocol.json \
  --output=build/stability-comparison.json
```

All contrasts call `compare_snake_learning.compare`: final minus initial, lower
minus higher learning rate, and TF32 minus FP32 at each rate/seed. It validates
complete episode records, summaries and pairing, then reports food changes with
10,000 paired percentile bootstrap resamples (seed 606001) and 95% intervals.
You can also compare any two evaluation files directly:

```sh
python3 tools/compare_snake_learning.py fp32-final.jsonl tf32-final.jsonl
```

Analysis requires a run record, at least one metrics event, and final
`metrics.epoch=32` / `metrics.agent_steps=8192`. It reports the number of logged
events and final metrics; logging cadence need not produce one event per epoch.
The run record contains the resolved configuration, including self-play.

The fixed classifications are:

- Preservation: every paired lower 95% bound is at least −0.5 food.
- Improvement: every paired lower 95% bound exceeds zero.
- Lower-rate advantage: all three lower-minus-higher-rate intervals exceed zero.

Preservation applies to the lower rate versus the initial policy and to TF32
versus FP32 at each rate. These episode intervals condition on saved policies;
they do not establish population-level training-seed significance. A warm-start
result does not establish learning from original pretrained weights. Speed needs
separate timing and hardware evidence.
