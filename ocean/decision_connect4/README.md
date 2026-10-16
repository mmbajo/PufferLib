# Connect Four with a pretrained Transformer

`decision_connect4` trains an imported Laya, BERT or ModernBERT decision model
using Puffer's native collector, PPO objective and Muon optimizer. The model sees
the complete 6-by-7 board as text and selects one of seven fixed column options.
It moves first against the campaign's pinned native opponent, preserving the
opening book, depth-three recursive evaluation and seeded tie-breaking.

The [C11 engine](engine/SOURCE.md) is unchanged from the interactive benchmark.
It fixes the upstream Puffer draw-mask bug and rejects illegal moves without
mutation. This is a fixed-opponent task; it does not train both players.

## Build and train

Follow the [model/tokenizer import instructions](../decision_laya/README.md),
then use the same imported bundle with this environment:

```sh
./build.sh decision_connect4 build/puffer_decision_connect4
./build/puffer_decision_connect4 train --policy.bundle=bundles/laya

# A short training/checkpoint/evaluation check:
./build/puffer_decision_connect4 train --policy.bundle=bundles/laya \
    --train.total_timesteps=8 --base.eval_episodes=2

./build/puffer_decision_connect4 eval /path/to/checkpoint.bin --headless \
    --policy.bundle=bundles/laya --base.eval_episodes=32
```

The default uses one environment, horizon 4, minibatch 4, learning rate `1e-5`,
and 4,096 environment decisions. These settings validate the training interface;
they are not a tuned learning recipe. Training is sequential and FP32, with
`base.async=0` and `base.cudagraphs=-1`. The shared native tokenizer is linked
statically; Python is unnecessary at training time. Keep the matching model and
tokenizer bundle with each flat Puffer checkpoint.

## Decision and episode contract

- Observation text contains all 42 cells, top row first and left to right,
  using `0` for empty, `1` for the player and `-1` for the opponent. It also
  includes the decision count and configured cap. Options `Column 0` through
  `Column 6` retain exactly that action order. Full columns are masked rather
  than removed, so action indices remain stable.
- One step makes the player's move and then the opponent's reply, unless the
  player's move already ends the game. Reward is zero on continuing steps,
  `+1` for a win, `-1` for a loss, and zero for a draw or time limit. `score` and
  `episode_return` preserve this signed result; `perf` reports win rate.
- `env.max_steps` defaults to 21, enough to finish any legal game. A smaller
  positive limit truncates after the complete turn. Natural termination takes
  precedence on the final allowed step.
- The adapter preserves final tokenized observations before autoreset. Both
  termination and truncation stop the advantage trace; only a pure timeout
  bootstraps from the final state. The next live observation and mask come from
  a fresh board while the previous reward/done remain available to the collector.
- The initial episode uses Puffer's instance seed. Each autoreset increments
  it modulo `2^32`. The engine owns its own `rand_r` state, independent of
  interleaved environments. Replay is deterministic on the same libc/platform;
  cross-libc tie-breaking is not guaranteed.

The complete board and all seven options must fit the imported bundle's token
budgets. The shared encoder rejects state truncation or options that become
identical under tokenization. Model support and checkpoint limitations match
the [shared decision policy](../decision_cartpole/README.md).

## Validation

The vendored engine tests use an independent grid win checker for all 69 winning
lines, plus 128 seeded replay/interleaving games, tactical fixtures, a full-board
draw regression, rejection behavior and terminal/time-limit precedence.

The adapter tests compare 455 transitions from 64 seeded games with direct C
engine execution. They decode the real native tokenized observations to verify
board orientation, action descriptions and count/cap text, while checking masks,
reward/done, saved final states, autoresets, seed wraparound and invalid inputs.
Replaying each game with a cap at its natural final move verifies that it remains
a terminal result without timeout bootstrapping. Tests pass with the full Laya
bundle. These checks establish correct mechanics, not improved playing strength.

```sh
# Build with a CUDA toolkit; these executables run entirely on the CPU.
./tests/build_decision_connect4_tests.sh
./build/test_connect4_engine
./build/test_decision_connect4_env bundles/laya

# Shared CUDA policy integration with a small imported encoder fixture:
./tests/build_decision_tests.sh policy-connect4
./build/test_decision_policy_connect4 SMALL_BUNDLE
```

The last test requires a GPU allocation. The small fixture must use a tokenizer
that distinguishes the board and options and fits their full sequence.
