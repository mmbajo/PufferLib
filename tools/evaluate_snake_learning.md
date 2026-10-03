# Paired native Snake learning evaluation

This runner evaluates imported text policies through `decision_laya` or numeric
board policies through `decision_snake`. Both use the native campaign Snake
engine and reverse-only action mask. Sampled, greedy and uniform-random action
modes are explicit. The runner allocates inference weights and workspaces;
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
Raylib from an existing native build, and, for text policies, the tokenizer static
library built with Rust. Set `CUDA_HOME` or `CUDACXX` and optionally `NVCC_ARCH`;
the helper also accepts an output executable path. The executable requires no
Python runtime. Python's standard library is used only for paired statistical
analysis. Ordinary configuration overrides, including `--base.gpu_offset`,
remain available.

## Action modes and board checkpoint imports

`--sampling=sampled` is the default. It uses Puffer's categorical sampler and
the per-episode Philox state. `--sampling=greedy` selects the first legal action
with the largest logit; ties follow the fixed up/down/left/right index order.
`--sampling=random` supplies zero logits to the same categorical sampler, giving
uniform legal-action probabilities and skipping policy forward computation.
Random mode accepts neither `--weights` nor `--pufdt`. All three modes retain
collision actions and the same environment dynamics. All require a GPU.

Build a separate executable for the numeric board policy:

```sh
PUFFER_EVAL_ENV=decision_snake ./tools/build_evaluate_snake_learning.sh \
    build/evaluate_snake_board

# Export a supported decisions/PyTorch checkpoint offline, then evaluate natively.
python tools/export_decision.py /path/to/policy.pt build/reference.pufdt
./build/evaluate_snake_board --pufdt=build/reference.pufdt --sampling=greedy \
    --base.seed=73 --episodes=1024 --batch=16 \
    --env-seed=670001 --action-seed=770001 --env.max_steps=500 > board.jsonl

# Matched random control: same episodes, seeds, cap and batch.
./build/evaluate_snake_board --sampling=random --base.seed=73 \
    --episodes=1024 --batch=16 --env-seed=670001 --action-seed=770001 \
    --env.max_steps=500 > random.jsonl
```

The board build accepts either `--weights=FILE` for Puffer's flat parameter
layout or `--pufdt=FILE` for named `PUFDT01` tensors, never both. PUFDT supplies
the architecture, including CLS/head readout and absolute/head-relative
coordinates, and validates names, shapes and finite values before loading.
Tail-distance features are unsupported. A flat board checkpoint requires the
matching width/layer configuration and current native board architecture.
Text bundles and PUFDT are separate formats; `--pufdt` is rejected by the text
build. See the [board import guide](../ocean/decision_snake/README.md#offline-reference-checkpoint-conversion).

Sampled and greedy scores answer different questions. Freeze the mode before
evaluation and compare like modes for a policy-improvement claim. Explicitly
label a model-versus-random or sampled-versus-greedy comparison as such.

## Observation and decision interface

The evaluator calls
[`decision_laya_encode`](../ocean/decision_laya/decision_laya.h) through the same
observation path as native training. With default `env.observation_format=0`,
its state text begins as follows, with the current step and cap substituted:

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

`--env.observation_format=1` prepends observed zero-based head/food coordinates
and signed food-minus-head row/column offsets, then retains the complete grid
and the same question/options. The first held-out episode in the example uses:

```text
Coordinates are zero-based (row, column). Head: (5, 5). Food: (5, 2). Food minus head: row +0, column -3. Negative row=up; positive row=down; negative column=left; positive column=right.
```

A full board reports `Food: none.` without offsets. This representation supplies
no recommended action or new legality rule. The numeric board build does not
use this text setting.

Imported policies also accept `--policy.zero_init_critic=1` for initialization
of the new critic and `--policy.sequence_length=N` for a reduced execution
padding length. Both default to `0`; sequence length `0` uses bundle `max_len`.
Packing and calibration still come from the bundle. An observation exceeding
the execution budget fails rather than losing state tokens. The same flat
checkpoint layout works at different supported execution lengths; finite
precision can still change predictions. Loading weights restores the saved
critic regardless of its initialization setting. See the
[training controls](../ocean/decision_laya/README.md#fine-tune-through-puffer).

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
mask rule. It also records model/representation, observation format, critic
initialization, execution and bundle token budgets, pooling and coordinates.
Per-episode rows contain actual seeds, the initial ordered 100-cell
board and action mask, food score, raw return, length, action counts, average
masked action probabilities, outcome and boundary flags. In greedy mode those
probabilities are the executed one-hot distribution, not the model's softmax.
The summary reports means, food-score standard error, ending counts and
inference-loop time.

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
comparison. By default the comparator requires identical complete protocols
except for the checkpoint path, including bundle, initialization seed and batch.
It rejects missing/duplicate episodes, changed seeds/initial boards/masks,
invalid action probabilities or counts, inconsistent terminal/reward accounting
and summaries that disagree with the raw episode rows.

An intentional ablation must declare each changed treatment field explicitly.
For example, changing the text format also changes its representation label:

```sh
python3 tools/compare_snake_learning.py grid.jsonl coordinates.jsonl \
    --allow-protocol-difference=observation_format \
    --allow-protocol-difference=representation \
    --output=representation-comparison.json
```

Only the model, representation, initialization option, execution token budget,
calibration and sampler fields listed in `--help` are eligible. The comparator
records both complete protocols, every actual difference, and the declared
allowances. It never waives matching episode IDs/counts, environment/action
seeds, model initialization seed, initial boards/masks, batch size, cap, action
space or game rules. A sampler treatment normally requires both `sampling` and
`sampling_mode`; a cross-model comparison may require several model fields.
Use allowances to describe a planned comparison, not to relabel unlike runs
as a controlled test of one change. Older records without the new metadata are
still readable, but mixing old/new records does not silently infer equivalence.

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
