# Lights Out with a Transformer decision policy

`decision_lightsout` trains imported Laya, BERT or ModernBERT decision models
using Puffer's native CUDA collector, PPO and Muon optimizer. It shares the model
and tokenizer adapter with [CartPole](../decision_cartpole/README.md) and uses
the [audited campaign C engine](engine/SOURCE.md).

The board is 5 by 5. Each move toggles the selected cell and its orthogonal
neighbors; the goal is to turn every light off. All 25 moves are legal on live
states. Options `A` through `Y` identify cells in row-major order, corresponding
to action indices `0` through `24`. The text board includes those labels:

```text
A=1 B=0 C=0 D=1 E=1
F=0 G=1 H=0 I=0 J=1
K=0 L=0 M=1 N=0 O=0
P=1 Q=0 R=0 S=1 T=0
U=1 V=1 W=0 X=0 Y=0
```

This example illustrates the format. Observations also include the step count,
limit, last action and preceding action, since the reward penalizes immediate
repeats and ABA sequences. The compact option letters fit Laya's default
192-token question/option budget; all 25 options retain distinct token spans.
The complete state must fit the bundle's context limit or encoding fails.

## Build and train

Import a model and tokenizer using the [shared instructions](../decision_laya/README.md),
then build this environment:

```sh
./build.sh decision_lightsout build/puffer_decision_lightsout
./build/puffer_decision_lightsout train --policy.bundle=bundles/laya

# A short training/checkpoint smoke run:
./build/puffer_decision_lightsout train --policy.bundle=bundles/laya \
    --train.total_timesteps=8 --env.max_steps=2 --base.eval_episodes=2

./build/puffer_decision_lightsout eval /path/to/checkpoint.bin --headless \
    --policy.bundle=bundles/laya --base.eval_episodes=32
```

Defaults use one environment, horizon/minibatch 4, learning rate `1e-5`,
4,096 training steps and a 100-step episode limit. These are conservative
integration settings, not a tuned learning recipe. Training is synchronous,
FP32 and uses the imported native tokenizer; Python is unnecessary at runtime.
The shared adapter's [supported models, checkpoint semantics and other
limits](../decision_cartpole/README.md#limits-and-validation) apply here.

## Rules, rewards and reset behavior

Seeded resets start from an all-off board and independently press each cell
with probability `0.15`, resampling all-off results. Presses commute and undo
themselves, so every initial board is solvable. The adapter starts with Puffer's
instance seed and increments the episode seed by one on each reset, wrapping
as an unsigned 32-bit integer. A seed has the same meaning as in the vendored
campaign engine on the same C library (`rand_r` supplies its PRNG).

The engine preserves its float reward arithmetic:

- Each step starts at `-0.0288`.
- Pressing the immediately previous cell adds a `-0.03` penalty; otherwise
  returning to the cell pressed two moves ago adds `-0.02`.
- Each reduction in the number of lit cells adds `0.005`, and increases subtract
  the same amount.
- Solving replaces that step's entire reward with `+2`.
- Reaching the step cap without solving subtracts another `0.5` and truncates.

`score` and `perf` are both the solved-episode indicator, independently of shaped
episode return. Invalid actions are rejected. No-action and partial-board
variants are not introduced.

The last observation and mask are retained before autoreset, and both solving
and truncation stop the advantage trace. A pure timeout bootstraps from its
saved final state; solving does not bootstrap, including a solve on the final
allowed move. The engine's logged reward remains unchanged. Puffer's existing
PPO path clamps raw training rewards to `[-1, 1]`, so the terminal `+2` reaches
the learner as `+1`; this differs from the unmodified episode-return log.

## Validation

The preserved engine tests check shaping, repeat/ABA penalties, invalid-input
atomicity, solve/timeout precedence and solve 512 seeded boards using an
independent GF(2) solver. The adapter test compares 724 transitions across 128
seeded solving traces with the engine and checks option ordering, token budgets,
action masks, history, saved final observations, autoreset seeds and timeout
bootstrapping hooks. Both the full Laya bundle and the small BERT fixture pass;
the longest tested input is 218 tokens with their shared tokenizer.

```sh
cc -std=c11 -O2 ocean/decision_lightsout/engine/tests/test_lightsout.c \
    -o build/test_lightsout_engine
./build/test_lightsout_engine

# The adapter harness is CPU-executed but compiled with the CUDA toolchain.
./tests/build_pretrained_tests.sh lightsout-env
./build/test_decision_lightsout_env bundles/laya
```

These checks establish integration correctness, not improved solving performance.
