# Paired native Snake learning evaluation

This runner evaluates imported Transformer policies through the existing
`decision_laya` Snake adapter. It uses the native campaign Snake engine,
serialized board, calibrated decision logits, reverse-only action mask and
Puffer categorical sampler. It allocates inference weights and workspaces;
there are no PPO or optimizer updates during evaluation.

**Food eaten is the primary metric.** Living longer without eating can improve
survival or avoid the collision penalty without improving the task score. The
comparison therefore reports food separately from return, episode length and
termination rates.

```sh
./tools/build_evaluate_snake_learning.sh

# Freeze checkpoint, episode seeds, cap and inference batch before evaluation.
./build/evaluate_snake_learning --policy.bundle=bundles/laya --base.seed=73 \
    --episodes=1024 --batch=16 --env-seed=670001 --action-seed=770001 \
    --env.max_steps=500 > before.jsonl
./build/evaluate_snake_learning --policy.bundle=bundles/laya --base.seed=73 \
    --weights=checkpoints/decision_laya/RUN/0000000000032768.bin \
    --episodes=1024 --batch=16 --env-seed=670001 --action-seed=770001 \
    --env.max_steps=500 > after.jsonl
python3 tools/compare_snake_learning.py before.jsonl after.jsonl \
    --output=paired-comparison.json
```

The example selects the final 32,768-step checkpoint in advance; it does not
assert that this training budget is sufficient to learn Snake. Run each policy
in a fresh process with the same imported bundle. Omitting `--weights` evaluates
the imported weights and the same initialized heads used by training. A flat
checkpoint must match the bundle's parameter count exactly and contain only
finite values. It contains weights only, not tokenizer or architecture metadata.
The protocol records `model_init_seed` (default 73), which initializes the added
critic and, for encoder-only imports, fresh decision heads. The critic is loaded
but does not select evaluation actions. Keep the initialization seed fixed.

Build requirements follow the native decision backend: CUDA/cuBLAS/NCCL,
Raylib from an existing `./build.sh decision_laya` build, and the tokenizer static
library built with Rust. Set `CUDA_HOME` or `CUDACXX` and optionally `NVCC_ARCH`;
the helper also accepts an output executable path. The executable requires no
Python runtime. Python's standard library is used only for paired statistical
analysis. Ordinary configuration overrides, including `--base.gpu_offset`,
remain available.

## Unchanged observation and decision interface

The evaluator calls
[`decision_laya_encode`](../ocean/decision_laya/decision_laya.h) through the same
observation path as native training. Its state text begins exactly as follows,
with the current step and configured cap substituted:

```text
Snake on a 10 by 10 grid. Rows run top to bottom; columns left to right. 0=empty, -1=food, 1=head, 2=neck, larger numbers follow the body toward the tail. Step 0 of 500. Board:
```

The prefix is followed by ten comma-separated rows of ten integer cells, each
ending in a newline. Body cells preserve their head-to-tail order. The exact
choice instruction is:

```text
Choose the next move. Eat food and avoid the walls and snake body.
```

The four option strings and action indices are unchanged:

| Action | Option |
| ---: | --- |
| 0 | `Move up` |
| 1 | `Move down` |
| 2 | `Move left` |
| 3 | `Move right` |

The 10×10 engine starts with head/neck/tail at row-major cells 55/54/53,
respectively, and one seeded food location. The initial mask is `[1, 1, 0, 1]`.
Only reversal into the neck is prohibited. Directions that collide with a wall
or body remain legal to sample; this evaluator does not add a safety planner.
The normal engine rule permits entry into the tail cell when that cell vacates.
Packing, tokenizer behavior, temperature calibration, rewards and dynamics are
unchanged. The imported bundle controls the token budget; no shorter prompt,
new shaping reward or greedy action selection is introduced. With the published
Laya bundle and tokenizer, inspected initial observations occupy **289 of 512
tokens**. This is a measured initial-state length, not a fixed length for every
board later in an episode.

## Episode identity and recorded outcomes

For episode ID `i`, the engine reset seed is `env_seed + i`, and the action
sampler uses Philox seed `action_seed + i`, subsequence 0 and offset 0. Both RNGs
reset for every episode, including episodes that reuse a batch slot. The
standard IDs are 0 through 1,023; `--episode-offset` selects a disjoint block.
Overflowing seed ranges are rejected. Exactly the requested number of episodes
is emitted, independent of completion order. Episode records are paired by ID,
not by output line order. The initial body is fixed, leaving only **97 possible
initial food placements**. Different held-out RNG streams can therefore repeat
initial boards, both within evaluation and across training/evaluation. Fresh
seeds do not establish generalization to unseen initial states.

The protocol records the bundle/checkpoint paths, episode and seed settings,
model initialization seed, inference batch, cap, padded token count, parameter
count, temperature, grid size, action count, sampling method and reverse-only
mask rule. Per-episode rows contain actual seeds, the initial ordered 100-cell
board and action mask, food score, raw return, length, action counts, average
masked action probabilities, outcome and boundary flags. The summary reports
means, food-score standard error, ending counts and inference-loop time.

| Event | Step reward | Outcome | Boundary behavior |
| --- | ---: | ---: | --- |
| Ordinary move | 0 | — | Continues |
| Eat food | +1 | — | Grows the snake |
| Wall/body collision | −1 | −1 | Natural termination |
| Fill all 100 cells | +1 for the final food | +1 | Natural termination; food score 97 |
| Reach cap without natural termination | 0, or +1 if food was eaten on that step | 0 | Pure timeout |

Raw episode return is therefore `food_score - 1` for a collision and
`food_score` otherwise. There is no per-step reward or penalty. `terminated`
means collision or full board. The engine sets `truncated` only for a pure
timeout. `time_limit_reached` records whether length equals the cap, so it can
also be true for a collision or full board on the last allowed step; those are
not pure timeouts. The evaluator retains the final transition without autoreset
before recording it and starting the next explicitly seeded episode.

Floating-point inference can vary with cuBLAS batch shape. Initial seeds and
boards are independent of batching, but a small probability change can alter a
sampled action. Use the **same batch size** for the primary before/after
comparison. The comparator requires identical complete protocols except for
the checkpoint path, including the bundle path, initialization seed and batch.
It rejects missing/duplicate episodes, changed seeds/initial boards/masks,
invalid action probabilities or counts, inconsistent terminal/reward accounting
and summaries that disagree with the raw episode rows.

## Interpreting the comparison

The comparator defaults to 10,000 paired episode bootstrap resamples using
seed 606001 and two-sided percentile 95% intervals. It reports:

- Primary mean food score before/after, paired mean food change, its standard
  error and interval, and counts of food-improved/worsened/tied episodes.
- Secondary mean return and length, plus collision, pure-timeout and full-board
  rates, with paired changes and intervals.

`positive_mean_food_score_change_interval` is true only when the lower bound of
`intervals_95.mean_food_score_change` is greater than zero. A longer episode or
higher return by itself does not satisfy this primary food criterion. Secondary
intervals are descriptive; they are not adjusted for multiple comparisons and
must not be used as alternate routes to declare primary success.

The uncertainty describes evaluation episodes for one trained checkpoint under
this Snake protocol. It does not estimate variability across training seeds,
establish transfer, prove a solved controller, or correct for selecting the best
of many checkpoints. Select the primary checkpoint prospectively and retain
inconclusive or negative results. Label any later tuning or independent
confirmation separately.

For a quick implementation check, evaluate the same 32 synthetic-BERT episodes
at batch sizes 1 and 8 and compare initial boards, masks, seeds and outcomes.
Also run with `--env.max_steps=1` to exercise cap accounting. Compare a JSONL
file with itself to verify zero paired changes and zero-width change intervals.
CPU validation tests run without CUDA or model weights:

```sh
python3 tests/test_compare_snake_learning.py
```
