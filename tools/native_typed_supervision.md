# Native typed supervision

`pretrained_decision` trains the same imported BERT, ModernBERT, or Laya model
that its JSONL serving mode uses. Each training row supplies one question and
an integer target. Choice and Score use categorical cross-entropy; Noul uses
binary cross-entropy over its two option logits. Frozen bundle temperatures
apply in both training and serving. This entrypoint does not use PPO.

Build with the CUDA/toolchain environment used by `tools/build_pretrained.sh`:

```sh
tools/build_pretrained.sh build/pretrained_decision
```

## Input and targets

Training and validation files contain one JSON object per line (expanded here
for readability):

```json
{
  "state": "North is empty; east is food.",
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose a move toward food.",
      "criteria": {"north": "move north", "east": "move east"},
      "option_order": [1, 0]
    }
  },
  "target": 1
}
```

`target` is the **semantic index**, before `option_order` permutes the packed
candidates. In this example, target 1 means `east`, which is packed in slot 0.
Choice indices follow the criteria object's insertion order. Score indices
follow its criteria array. Noul targets are integer 0 for false and 1 for true,
including when custom labels or an option permutation are supplied; JSON
booleans are not integer labels.

Different rows may have different candidate counts. Each minibatch pads token
sequences and candidate positions and masks both kinds of padding. Training
uses serving's state rendering, typed question parsing, tokenizer, and packing.
States that exceed the token budget are rejected for supervised training.
Keep every question for a state in the same split: the trainer rejects exact
rendered states shared by training and validation.

```sh
build/pretrained_decision --bundle build/bert-native-fixture \
  --train build/typed-supervision/train.jsonl \
  --validation build/typed-supervision/heldout.jsonl \
  --epochs 20 --batch-size 32 --learning-rate 0.001 --seed 1 \
  --max-tokens 256 --save-weights build/typed-supervision/trained.bin \
  > build/typed-supervision/training.jsonl
```

The optimizer is FP32 Adam with beta1 0.9, beta2 0.999, epsilon 1e-8,
no weight decay, and no gradient clipping. Supervision updates the encoder,
typed head, and option scorer; it leaves act-head and PPO critic weights
unchanged. Existing PPO optimizer settings and defaults are independent.

Stdout contains JSONL evaluation records at epoch 0 and the final epoch,
plus one training record per epoch. Each validation row also emits its epoch,
zero-based row index, correctness and log loss as a `validation_row` record.
Evaluation summaries name the split, epoch,
number of distinct rendered states, question count, and per-type question
count, accuracy, and mean natural-log loss. Training `mean_log_loss` averages
the online losses before each update; it is not a final-checkpoint evaluation.
Validation does not select checkpoints or stop training.

`--epochs 0` evaluates and exports initialized weights. `--weights FILE` starts
from an existing flat checkpoint with fresh Adam moments. `--save-weights`
writes the existing Puffer FP32 tensor layout, including its unused critic,
so ordinary serving can load it directly:

```sh
build/pretrained_decision --bundle build/bert-native-fixture \
  --weights build/typed-supervision/trained.bin \
  --input build/typed-supervision/heldout.jsonl --max-tokens 256 --raw
```

The serving path ignores training targets. Its `raw` output exposes full
precision option logits and temperature, while public probabilities are
rounded. Compute checkpoint log loss from the raw values for comparison to
training evaluation records.

## Exact Snake labels

The dataset generator calls the existing C Snake engine. It uses direct board
comparisons to keep the two state sets disjoint and puts all three questions
for each board in one split.

```sh
mkdir -p build/typed-supervision
cc -std=c11 -O2 tools/generate_snake_supervision.c \
  -o build/typed-supervision/generate_snake_supervision
build/typed-supervision/generate_snake_supervision \
  build/typed-supervision/train.jsonl build/typed-supervision/heldout.jsonl \
  512 128 73
```

Output paths must be new. The optional positional arguments are training
state count, held-out state count, and seed. The defaults above produce 1536
training questions and 384 held-out questions, covering 512 and 128 boards.

Each board asks:

- **Choice:** which of two or three offered legal first moves reaches food
  soonest within four attempted moves. The offered set has one unique,
  reachable minimum. Worse moves are randomly ordered before placing the
  best move in its assigned target slot.
- **Noul:** whether a named legal attempted move collides, determined by one
  exact engine transition. Integer 1 means a collision.
- **Score:** moves to food after a named first move, with ordered levels
  `1 move`, `2 moves`, `3–4 moves`, and `not within 4`. Counts include the
  attempted first move; collision and unreachable cases use the last level.

States expose ordered body and food coordinates plus neighboring cell kinds.
They do not include search distances. Retained states are early enough in the
episode that the four-move search does not encounter the hidden time limit.
Replay metadata records the reset seed and action history for independent
label and state-text checks. Score classes and Noul labels are balanced within
each split; their queried move is sampled uniformly among moves matching that
row's target using the random state's high bits. Report training-fitted
board-free baselines beside model accuracy: offered directions can have different
chances of being best even when their order is randomized.

The small `bert-native-fixture` has a random width-16 encoder and a real full
vocabulary tokenizer. Learning on it checks native supervision and held-out
generalization on this dataset; it is not a pretrained-language quality test.

## Checks

The CPU dataset test independently replays generated boards through the C
engine and checks labels, balancing, deterministic output, and disjointness:

```sh
python3 -m unittest discover -s tests -p 'test_snake_supervision.py' -v
python3 -m unittest discover -s tests -p 'test_analyze_typed_supervision.py' -v
python3 tests/test_pretrained_supervision.py \
  --executable build/pretrained_decision --bundle build/bert-native-fixture \
  --output-dir build/typed-supervision/integration-cpu --cpu-only
```

Inside a GPU allocation, run the same integration runner without `--cpu-only`
and with a new output directory. It checks mixed candidate masks, semantic
target permutations, nonunit temperatures, final partial batches, exact
initialized checkpoint reload, frozen act/critic weights, and batched
evaluation against ordinary serving. The full learning run should report
initial and final held-out accuracy and log loss for each question type,
together with the distinct held-out state count.

Compare the initial and final per-row validation records from the training
process, using the same training and held-out files passed to the trainer:

```sh
python3 tools/analyze_typed_supervision.py \
  build/typed-supervision/train.jsonl \
  build/typed-supervision/heldout.jsonl \
  build/typed-supervision/training.jsonl
```

The comparison pairs native correctness/loss by row and bootstraps whole boards.
Its intervals describe these held-out boards; they do not measure variation
across training seeds or independent game trajectories.

Each type also reports a training-label majority and a direction-based rule.
For Choice, the latter predicts the most frequent best direction for the offered
set, ignoring its order; an additional `ordered_options` rule fits the target
index for each full ordered list. For Noul and Score, `directions` fits the label
for the queried move. Ties choose the first label observed in training. Unseen
features fall back to the training majority among valid candidates, or the first
candidate if none was observed (sorted directions for the direction rule).
Held-out labels are used only to score these rules. The analyzer reports each
rule's accuracy and a paired interval for final-model accuracy minus the rule.
An ordered-list rule can capture a direction prior, so beating the initial model
alone does not establish shortest-path reasoning.
