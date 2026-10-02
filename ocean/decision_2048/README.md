# 2048 with a pretrained Transformer policy

`decision_2048` uses the audited interactive-benchmarks C engine and the shared
native Laya/BERT/ModernBERT decision policy. It trains through Puffer's collector,
PPO and Muon. Each decision receives the full 4×4 board as text with actual tile
values, the current step and the episode limit. Simulation advances only when an
action is applied.

```sh
./build.sh decision_2048 build/puffer_decision_2048
./build/puffer_decision_2048 train --policy.bundle=bundles/laya
./build/puffer_decision_2048 eval /path/to/checkpoint.bin --headless \
    --policy.bundle=bundles/laya --base.eval_episodes=32
```

Use the [model/tokenizer import instructions](../decision_laya/README.md) to
prepare a bundle. This adapter inherits the shared policy's supported models,
tokenizer requirements, sequence budgets, FP32 computation and synchronous
collection limits. The initial configuration uses one environment, horizon 4,
minibatch 4, learning rate `1e-5`, 4,096 training steps and a 500-step episode
limit. These are integration defaults, not a tuned learning recipe.

## Actions and campaign rules

| Action | Option | Legal when |
| --- | --- | --- |
| 0 | Move up | The board changes |
| 1 | Move down | The board changes |
| 2 | Move left | The board changes |
| 3 | Move right | The board changes |

All four option slots remain present. The action mask excludes moves that leave
the board unchanged. For compatibility with the campaign engine, the explicit
`decision_2048_step` API accepts in-range no-ops: they consume one step, earn
`-0.05`, and neither spawn a tile nor advance the RNG. Invalid actions and steps
after completion are rejected without mutation. Puffer's normal policy respects
the mask.

Each episode starts with exactly two tiles; there is no adaptive reset
curriculum or lifetime-dependent cap. A changed board spawns one random tile:
2 with approximately 90% probability, otherwise 4. An original tile merges at
most once per action. Reaching 2048 does not end the episode. Tiles stop merging
at `2^30` to preserve the public `int32_t` board representation.

Every environment has private `rand_r` state. The first episode uses Puffer's
instance seed, subsequent episodes increment that seed modulo `2^32`, and
`decision_2048_reset(env, seed)` provides explicit replay control. Equal seeds
and actions reproduce the campaign on the same libc; libc-independent seeded
replay is not promised. These fixed-reset traces differ from stock Puffer's
curriculum RNG consumption. The vendored engine and original tests remain
unchanged; see [provenance](engine/SOURCE.md).

## Rewards, metrics and boundaries

`score` is the standard merge score, summing the resulting tile values.
`episode_return` sums the campaign's shaped rewards: `0.05` per merge, the
original large-tile bonus from tile 128 upward, and a `-1` game-over penalty.
For example, `[2,2,4,0]` moved left becomes `[4,4,0,0]` before spawning a tile:
the move adds **4** to score and **0.05** to reward. They are different metrics.
Puffer's training collector applies its usual reward clipping; logged returns
retain the environment reward. `max_tile` records the largest final tile and
`perf` is the fraction of episodes reaching at least 2048.

No remaining move is a true termination. The explicit step cap is a truncation
with no extra penalty. Natural termination takes precedence if both occur
together. Either boundary stops the advantage trace. A pure timeout bootstraps
from the saved final tokenized board; a true terminal receives no bootstrap.
Puffer autoresets after recording that final state and publishes the next
episode's board and legal mask. Explicit reset clears the saved transition.

## CPU validation

```sh
./tests/build_decision_2048_tests.sh
./build/test_decision_2048_engine
./build/test_decision_2048_env bundles/laya
```

The adapter test requires the CUDA toolchain for shared policy headers and the
native tokenizer library, but running these tests requires no GPU. It compares
6,144 seeded campaign/adapter transitions, including interleaved environments,
board serialization, rewards, scores, legal masks and autoresets. Separate
fixtures check timeout snapshots, game-over precedence, reset clearing, no-op
semantics, token markers, large tile values and invalid inputs. The original C
engine tests exercise merge order, RNG isolation and overflow boundaries.
