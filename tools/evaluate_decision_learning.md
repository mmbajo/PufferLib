# Paired native CartPole learning evaluation

This runner evaluates imported Transformer decision policies with the existing
`decision_cartpole` engine, text packing, calibrated policy forward pass and
Puffer's stochastic categorical sampler. It allocates inference workspaces and
weights only; it performs no PPO or optimizer updates.

```sh
./tools/build_evaluate_decision_learning.sh

# Freeze these settings before inspecting the trained checkpoint's results.
./build/evaluate_decision_learning --policy.bundle=bundles/laya \
    --episodes=256 --batch=16 --env-seed=170001 --action-seed=270001 \
    > before.jsonl
./build/evaluate_decision_learning --policy.bundle=bundles/laya \
    --weights=checkpoints/decision_cartpole/RUN/0000000000032768.bin \
    --episodes=256 --batch=16 --env-seed=170001 --action-seed=270001 \
    > after.jsonl
python3 tools/compare_decision_learning.py before.jsonl after.jsonl \
    --output=paired-comparison.json
```

Use one fresh process for each checkpoint. A missing `--weights` evaluates the
imported model and the same initialized heads used by native training. Flat
checkpoints must exactly match the registered parameter count; all loaded
values are checked for finiteness. The protocol records `model_init_seed` from
`--base.seed` (default 73), before initialization advances its RNG. This seed
determines fresh decision heads for encoder-only imports and the added critic;
keep it fixed along with the evaluation seeds. `--policy.bundle` supplies architecture,
tokenizer and calibration. Ordinary configuration overrides, including
`--base.gpu_offset`, remain available. This tool is specific to the two-action
CartPole decision adapter, not a generic environment evaluator.

The imported-policy controls `--policy.zero_init_critic=1` and
`--policy.sequence_length=N` also apply here; see the
[decision training guide](../ocean/decision_laya/README.md#fine-tune-through-puffer).
The protocol records the effective execution length as `padded_tokens`, the
bundle limit separately as `bundle_max_tokens`, and `zero_init_critic` explicitly.
Hold these settings fixed between paired evaluations. A shorter execution
budget rejects overlength inputs instead of truncating the state; a loaded
checkpoint restores its saved critic. Old default-only JSONL files remain
readable, but comparisons reject missing or different control metadata rather
than assuming equivalent settings.

For episode identifier `i`, the initial environment RNG seed is
`env_seed + i` and the action-sampler Philox seed is `action_seed + i`, with
subsequence and offset zero. These seeds are reset independently for every
episode, including episodes assigned to a reused batch slot. The default
identifiers are 0 through 255; `--episode-offset` supports explicit disjoint
identifier blocks. Overflowing seed ranges are rejected. Environment and action
seed assignments, initial states and episode quotas therefore do not depend on
completion order or batch-slot assignment. Floating-point model calculations
can vary slightly with cuBLAS batch size; use the **same batch size** for the
primary before/after comparison. The ordinary training CLI's historical
single-GPU environment seeding is not used for this protocol.

Every run emits one JSON protocol record, exactly the requested number of
per-episode records, and one summary record. Episode rows include actual seeds,
initial physical state, length, raw return, physical termination flags, whether
the time limit was reached, pure timeout, counts of left/right actions and mean
policy probability of pushing right. `max_abs_theta_observed` is the maximum
absolute **pre-action policy-observed** angle, excluding the terminal state that
the environment immediately autoresets. Pole-angle/cart-position failure flags
come from the engine's final-state log and include physical failures on the
last step. A simultaneous physical failure and time limit is marked as both,
but is not counted as a pure timeout.

Stock Puffer CartPole awards 1 on continuing steps and 0 on the episode's last
step, including a timeout. Consequently score/episode length is in `[1, 200]`
and raw return is `length - 1` at the default cap. The runner preserves this
convention. It does not substitute Gym dynamics, modify rewards, add shaping,
change the environment cap or replace stochastic sampling with greedy actions.

The comparator requires identical protocols apart from checkpoint path,
identical paired seeds/initial states, complete episode identifiers and valid
reward/action accounting. It reports mean episode length/return before and
after, paired mean length change, improved/worsened/tied episodes and pure
survival-to-cap rate. Its default confidence intervals use 10,000 paired
bootstrap resamples of episode rows with seed 606001 and percentile 95% bounds.
A positive lower bound for the mean-change interval is evidence of an increase
under this evaluation protocol. The uncertainty concerns evaluated episodes
for **one trained checkpoint**; it does not estimate variation across training
seeds, establish task transfer or correct for selecting the best of many
checkpoints. Preselect the primary checkpoint and report exploratory
intermediate-checkpoint results separately.

For a quick implementation check, run a small imported BERT fixture over the
same 32 episode seeds at batch sizes 1 and 8. Compare records by episode ID:
initial states and seeds must match exactly, and lengths/actions will normally
match unless a small floating-point probability difference changes a sampled
action. Compare a JSONL file with itself to verify zero paired change. The
statistical utility's CPU tests run with:

```sh
python3 tests/test_compare_decision_learning.py
```
