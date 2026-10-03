# Decision Snake

`decision_snake` wraps the pinned C Snake engine used by the Transformer decision
experiments. It is a separate environment from PufferLib's stock `snake`.
The [engine source and license](engine/SOURCE.md) retain their original bytes.

Training advances synchronously: observe the board, compute an action, then step
the C environment once. Time spent computing the action never advances the game.
Clocked evaluation and holding an earlier action during inference belong to the
separate benchmark evaluation protocol.

## Build and train with PufferLib

From the repository root, with a CUDA toolkit and PufferLib's native dependencies
available:

```sh
./build.sh decision_snake build/puffer_decision
./build/puffer_decision train --train.total_timesteps=4096 --base.eval_episodes=32
./build/puffer_decision eval latest --headless --base.eval_episodes=100
```

The build selects FP32 automatically. It uses `CUDA_HOME`/`CUDA_PATH`, `CUDACXX`,
or `nvcc` from `PATH`; `CC` can select the C compiler. NVHPC's sibling `math_libs`
and `comm_libs/nccl` directories are detected. An external NCCL installation can
be supplied through `NCCL_HOME`, or `NCCL_INCLUDE_DIR` and `NCCL_LIB_DIR`.
The default build includes `sm_80` machine code and PTX, plus `sm_90` machine code
when supported by the toolkit. Set `NVCC_ARCH` to override these targets. CPU/web policy
evaluation and the legacy `--profile` kernel harness are unsupported for this
Transformer. `--cu` selects GPU environments in PufferLib and does not apply here:
the environment remains C, while policy computation and training run on CUDA.

This is PufferLib's native rollout collector, PPO loss, Muon optimizer, logging,
and checkpoint path. The policy is a bidirectional Transformer with four attention
heads, head pooling, head-relative coordinate embeddings, and an ordered body
embedding. A stateless policy stage replaces MinGRU. `policy.hidden_size` controls
Transformer width and `policy.num_layers` controls its depth.

The default [config](../../config/decision_snake.ini) uses width 32, two layers,
32 environment instances, a rollout horizon of 32, minibatches of 256, and 512,000
training timesteps. Other optimization settings remain PufferLib's defaults;
this is not a reproduction of the Python campaign's optimizer or training recipe.
`base.async=0` and `base.cudagraphs=-1` are currently required. The example above
overrides the duration for a short smoke run; omit that option for the configured
duration. Native checkpoints contain policy weights, not complete optimizer and
environment state for an exact continuation.

Flat checkpoints also omit architecture metadata. If training changes
`--policy.hidden_size` or `--policy.num_layers`, pass the same settings to `eval`;
for example, a model trained with `--policy.hidden_size=64 --policy.num_layers=3`
requires both options when loading it. Native `eval` uses PufferLib's sampled
actions. The benchmark's greedy and clocked evaluations are separate protocols.

## Game and observation contract

- One snake on a 10 by 10 board, initially at cells 55, 54, 53 (head first).
- Four actions: 0 up, 1 down, 2 left, 3 right. The mask excludes only reversal
  into the neck. Fatal wall and body moves remain legal. Invalid actions reject;
  they are never redirected into a different move.
- Observations contain 100 unsigned bytes in row-major order. They encode the
  original signed board plus one: food 0, empty 1, head 2, neck 3, and ordered body
  segments through 101. This is a lossless encoding, with no one-hot expansion.
- Food gives reward +1 and grows the body. A collision gives -1 and terminates;
  its final observation retains the board before the collision. Other moves give
  zero. Entering the simultaneously departing tail is allowed.
- Filling the board terminates with outcome +1 and a total score of 97 food.
  Collision has outcome -1. A timeout has outcome 0, `truncated=1`, and no extra
  penalty. Natural termination takes priority on the final permitted tick.
- `[env] max_steps` is a positive int32, defaulting to 500 in the environment
  config. The seed selects food using the engine's private 32-bit LCG.

For example, the initial cells 53, 54, 55 are engine values `3, 2, 1` and policy
tokens `4, 3, 2`. They remain distinguishable body positions.

## Native adapter API

Include `decision_snake.h` in one translation unit and link `engine/snake.c`
compiled as C11. Call `puf_init` on a zero-initialized `Env`, with `env.rng` set to
the initial seed, then bind the observation, action, reward, terminal, and
four-byte action-mask buffers in `env.agents[0]`. The adapter owns the engine;
the caller owns these buffers. Call `puf_close` to free the engine.

Two explicit helpers support a collector that owns reset timing:

```c
decision_snake_reset(&env, seed); // 0 on success; explicit uint32 seed
decision_snake_step(&env, action); // 0 on success; does not autoreset
```

Both helpers return -1 on rejection and leave engine state and buffers unchanged.
After each accepted step, `env.transition` owns the observation and action mask,
action, reward, score, episode return, step count, episode seed, outcome, and
separate `terminated` and `truncated` flags. `transition.valid` becomes 1 after
an accepted action. Explicit reset clears this transition. The completed engine
rejects additional actions until reset. Its final mask is zero at both types of
episode boundary; a value-only bootstrap should consume the board without trying
to sample an action from that mask.

The PufferLib facade provides its usual autoreset behavior:

- `puf_reset` starts `env.next_seed`, initially the supplied `env.rng`.
- `puf_step` advances exactly one action and automatically starts a new episode
  after either boundary. The live Agent observation/mask then describe that new
  episode; `env.transition` retains the completed transition, including its final
  board. The Agent reward still belongs to that completed transition.
- `Agent.terminals[0]` is `terminated || truncated`, marking a reset boundary.
  It does **not** specify whether the value target may bootstrap.
- Each reset sets `next_seed = seed + 1` modulo 2^32. Instances own their seed
  sequences and do not use process-global `rand()`. This per-instance sequence
  differs from the Python campaign's shared, completion-ordered seed allocator.
  A collector reproducing that allocator must use the explicit helpers and supply
  its reset seeds in the reference order.
- `puf_step` treats an invalid or masked action as a caller error and exits with
  a diagnostic. Unlike the explicit helper, the PufferLib void interface cannot
  report a rejected step to its caller.

The log reports food as `score`, total reward as `episode_return`, and
`perf = score / 97`. `terminated` and `truncated` count their respective completed
episodes. All fields accumulate until the existing PufferLib log reducer reads
and clears them.

## Training boundary

The native rollout path copies the saved final board and timeout flag to the
GPU. It adds `gamma * V(final_board)` to timeout rewards before PufferLib's existing
advantage calculation, while the done flag stops propagation across resets.
Environment episode metrics retain the original game rewards.

Other collectors must also use the value of `transition.observations` when
`truncated=1`, set the future value to zero when `terminated=1`, and stop GAE
propagation across either reset. Bootstrapping from the autoreset observation or
silently treating every timeout as death changes the experiment. The header
defines `PUF_HAS_TRUNCATION` and supplies `puf_truncation_observation` to expose
that contract to the native collector, independently of the policy architecture.

The default build includes a simple Raylib board renderer. Define
`PUF_HEADLESS` when compiling a standalone test without rendering;
`puf_render` then does nothing and no window functions are linked.

## Offline reference checkpoint conversion

An optional PyTorch conversion tool exports existing `decisions` checkpoints
without importing that repository:

```sh
python tools/export_decision.py path/to/policy.pt build/reference.pufdt
```

The input must contain `policy_config` and `policy_state`, as produced by the
Snake campaign. The exporter uses `torch.load(..., weights_only=True)`, validates
every name and shape, rejects nonfinite weights and unsupported tail-distance
features, and writes named FP32 tensors without transposing linear weights.
Both CLS/head pooling and absolute/head-relative coordinates are supported.
PyTorch is required only for this conversion and numerical reference checks.

The versioned `PUFDT01` interchange file is described in
[`src/decision_checkpoint.cuh`](../../src/decision_checkpoint.cuh). It contains
architecture settings and model weights, with no optimizer or environment state.
It is used by the native numerical validation harness and is distinct from
PufferLib's flat policy checkpoint format. The
[native Snake evaluator](../../tools/evaluate_snake_learning.md#action-modes-and-board-checkpoint-imports)
also loads it with `--pufdt`, preserving the exported pooling and coordinate
settings. Build that runner with `PUFFER_EVAL_ENV=decision_snake`; select
`--sampling=sampled`, `greedy` or `random` explicitly when comparing protocols.

## Validation

The engine/adapter checks need only C and C++ compilers and Python's standard
library. They compare fixed traces captured from the campaign engine, including
complete games, seed independence, rejected actions, and timeout boundaries:

```sh
python3 -m unittest tests.test_decision_snake -v
```

Build the GPU checks with the same CUDA settings as the native trainer. Run them
on a GPU host. The numerical reference suite additionally needs PyTorch and
NumPy; it compares forward outputs and every parameter gradient against
PyTorch's Transformer, tests external Puffer parameter storage, and verifies that
native gradient updates reduce a small fixture loss.

```sh
./tests/build_decision_tests.sh transformer
python tests/test_decision_transformer.py --executable build/test_decision_transformer
./tests/build_decision_tests.sh puffer
./build/test_decision_puffer
```

The Puffer integration check uses the real rollout and training callbacks. Its
build uses the Raylib files fetched by `./build.sh decision_snake`; `RAYLIB_HOME`
can select an existing installation. Both test builds honor `CUDA_HOME`,
`CUDACXX`, and `NVCC_ARCH`. The native trainer and integration executable run
without Python.
