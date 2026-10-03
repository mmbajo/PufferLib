# CartPole with a pretrained Transformer policy

`decision_cartpole` trains an imported Laya, BERT or ModernBERT model through
Puffer's native rollout collector, PPO objective and Muon optimizer. It shares
the decision policy implementation with `decision_laya` (Snake): only state
serialization, action descriptions and the environment differ.

The environment uses the existing `ocean/cartpole/cartpole.h` integration
equations, initial-state sampling, rewards and rendering. It presents the
position, velocity, pole angle and angular velocity as text, along with the step
count and episode limit. Float formatting is locale-independent and preserves
each float's round-trip precision. Action 0 is **Push left** and action 1 is
**Push right**; both are always legal. Training runs sequentially: observe,
infer, then step. Model computation does not advance simulation time.

## Import, build and train

Use an existing imported bundle, or follow the
[model/tokenizer import instructions](../decision_laya/README.md). The same
bundle can initialize either environment; each task produces its own fine-tuned
checkpoint.

```sh
python tools/import_pretrained.py convaiinnovations/laya bundles/laya \
    --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
./build.sh decision_cartpole build/puffer_decision_cartpole

# Short mechanical validation, including checkpoint save and evaluation:
./build/puffer_decision_cartpole train --policy.bundle=bundles/laya \
    --train.total_timesteps=8 --base.eval_episodes=2

# The supplied default configuration runs 4,096 environment steps:
./build/puffer_decision_cartpole train --policy.bundle=bundles/laya

./build/puffer_decision_cartpole eval /path/to/checkpoint.bin --headless \
    --policy.bundle=bundles/laya --base.eval_episodes=32
```

Build dependencies are the same as the Snake text policy: CUDA/cuBLAS, Puffer's
native Raylib/OpenMP/NCCL dependencies, and a Rust/Cargo-built static tokenizer
library. Python is used for offline import/reference tests, not for the training
runtime. `CUDA_HOME`, `CUDACXX` and `CARGO` can select installed toolchains.

The default uses one environment, horizon 4, minibatch 4 and learning rate
`1e-5`. These are conservative integration settings, not a tuned CartPole
learning recipe. Use `--env.max_steps=N` to change the default 200-step limit;
`continuous=1` is rejected by this discrete decision adapter.

For multiple GPUs, see the [distributed training guide](../../tools/distributed_training.md).
It explains per-rank batch settings, optional Muon state sharding, and global
metrics and checkpoint step counts.

## Rewards and episode boundaries

Stock Puffer CartPole gives reward 1 on a continuing step and 0 on the step that
ends an episode, including its time limit. This adapter preserves that rule and
the stock physics, which should not be assumed identical to another library's
CartPole implementation. Its `score` is episode length and `perf` is accumulated
reward divided by the configured limit.

The adapter records the final tokenized state before autoreset. Physical failure
and timeout both set Puffer's done flag, stopping the advantage trace. A pure
timeout adds `gamma * V(final_state)` to the training reward; physical failure
has no bootstrap, even when it coincides with the time limit. Episode logs keep
the original environment rewards. Explicit reset clears the saved transition.
This fixes the boundary handling for the new decision environment; stock
`cartpole` retains its existing terminal-buffer behavior.

## Reuse the policy for another environment

The reusable API is in [`src/decision_policy.h`](../../src/decision_policy.h).
An adapter declares one discrete action count and includes that header directly:

```cpp
#define DECISION_ACTIONS 2
#include "../../src/decision_policy.h"
#include "pufferenv.h"

#define ACT_SIZES {DECISION_ACTIONS}
#define NUM_ATNS 1

// Called from the environment's observation writer:
decision_policy_encode(state_text, "Choose an action for this task.",
    {"First action", "Second action"}, agent->observations);
```

The snippet illustrates the policy contract; the environment still implements
Puffer's normal `Env`, `Log`, initialization, reset, step, logging and rendering
API. In particular:

- Supply exactly `DECISION_ACTIONS` distinct option descriptions in action-index
  order on every observation. Use the existing action-mask buffer to disable
  illegal actions, keeping at least one legal action on live states. Keep the
  option slots fixed instead of deleting masked options.
- Serialize all task-relevant state into `state_text`. The shared encoder rejects
  state truncation, lost options and options that collapse to identical token
  spans. Token budgets and tokenizer special tokens come from the bundle.
- Use the `obs_t` and `OBS_SIZE` defined by the shared header. It stores token
  IDs and marker positions losslessly in bytes; environments do not pack float
  token IDs themselves. Puffer initializes the shared model/tokenizer context
  before starting environment workers.
- For time limits, define `PUF_HAS_TRUNCATION` and implement
  `puf_truncation_observation(Env*, const Agent*)` as documented in
  [`pufferenv.h`](../../src/pufferenv.h). Return the saved pre-reset observation
  for a pure timeout and `NULL` otherwise, while marking either boundary done.
- Add the environment header and its normal configuration file. Copy the
  synchronous policy settings from `config/decision_cartpole.ini` and require
  `policy.bundle`. `build.sh` detects the direct shared-header include and adds
  the tokenizer/CUDA build dependencies automatically. No environment-specific
  CUDA policy file or Transformer backend change is required. Additional native
  dependencies of the environment itself still need their usual build wiring.

The same `src/decision_policy.cuh` handles the action-count-dependent tensors,
calibration, value output, backward pass and Puffer parameter registration.
`src/decision_passthrough.cuh` supplies the stateless network and decoder stages.
The old four-action Snake observation encoding and model parameter order are
preserved, so its existing weights remain loadable with the same bundle.

## Limits and validation

This adapter supports one discrete head with 2–255 fixed action slots, subject
to the model's sequence budget. Two-action CartPole and four-action Snake have
end-to-end integration coverage, alongside seven-action
[Connect Four](../decision_connect4/README.md), 25-action
[Lights Out](../decision_lightsout/README.md), and four-action
[2048](../decision_2048/README.md). The numeric upper bound does not guarantee
that 255 descriptions fit a particular bundle. Continuous actions and multiple
independent action heads require additional policy interfaces.

Training remains FP32, with `base.async=0` and `base.cudagraphs=-1`. Context is
limited to 2048 tokens, model width must be divisible by four, and attention uses
quadratic workspace. Model support remains BERT/ModernBERT including full Laya.
PPO trains the option policy and added critic, without an act/escalate loss.
The bundle's calibration is not refitted by PPO.
The [native benchmark](../../tools/benchmark_native.md) measures actual rollout
and learner throughput without checkpoint I/O; short train commands include
checkpoint saving and should not be used as steady-state speed measurements.

Keep the original matching bundle with every flat Puffer checkpoint, and record
the environment/configuration that produced it. Flat files contain weights,
without tokenizer, architecture, optimizer, RNG or environment state. Loading
one is not exact training resume. Since parameter shapes do not encode the
environment, a same-shape checkpoint can load in another task without implying
useful transfer. Evaluation samples actions under Puffer's normal protocol.

Validation passed for 2,249 matching stock/decision CartPole physics, reward and
reset snapshots, including timeout versus physical-terminal behavior. Shared
policy integration tests passed for both action counts, independent model
forward and all parameter gradients, actual PPO updates and exact checkpoint
reload. CPU packing checks also cover eight actions, where the token header
grows. The original small Snake integration and stock CartPole native/CPU builds
continue to pass.

The full published Laya model completed an eight-step CartPole run with forced
two-step timeouts. A small BERT encoder completed 256 steps with two environments
and two collection buffers. Both produced finite saved weights, changed encoder,
scorer and critic parameters, preserved the unused act head, and passed separate
checkpoint reload/evaluation. These establish a working training path; they do
not establish a tuned policy or improved CartPole performance.

```sh
./tests/build_pretrained_tests.sh cartpole-stock
./tests/build_pretrained_tests.sh cartpole-env
python tests/test_decision_cartpole_env.py \
    build/test_decision_cartpole_stock build/test_decision_cartpole_env bundles/laya

./tests/build_pretrained_tests.sh layout-8
./build/test_decision_policy_layout_8 bundles/laya

# Build on a CUDA-toolchain host, then run inside a GPU allocation.
# SMALL_BUNDLE is an imported small BERT/ModernBERT fixture whose tokenizer
# distinguishes the task descriptions and whose context fits the whole state.
./tests/build_pretrained_tests.sh policy-cartpole
./tests/build_pretrained_tests.sh policy-laya
./build/test_decision_policy_cartpole SMALL_BUNDLE
./build/test_decision_policy_laya SMALL_BUNDLE
```

The integration tests allocate independent reference models and training
workspaces; use a small encoder fixture rather than the full Laya checkpoint
for these tests. One environment is compiled into each executable; the new
adapter does not implement a mixed-task training scheduler.
