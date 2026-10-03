# Pretrained Transformer decisions in native Puffer

`decision_laya` trains a pretrained Transformer policy with Puffer's native
collector, PPO objective and Muon optimizer. A raw-text tokenizer runs on the CPU;
the encoder, typed decision heads and gradients run in CUDA C++. Each training
step observes the game, computes an action, then advances the environment.
Python is used only for optional offline imports and reference tests.

The policy adapter is shared with
[`decision_cartpole`](../decision_cartpole/README.md), which uses two actions
and the stock CartPole physics. The reusable interface accepts an environment's
state text, instruction and fixed action descriptions; see that guide for adding
another discrete-action environment without changing the Transformer backend.

There are two import modes:

| Source | Imported parameters | Newly initialized parameters |
| --- | --- | --- |
| Full Laya checkpoint | BERT/ModernBERT encoder, type embeddings, decision Transformer, option scorer, act/escalate head; calibration metadata is preserved | Puffer value head |
| Hugging Face BERT or ModernBERT checkpoint | Encoder; known original pooler/classifier/MLM heads are explicitly excluded | Laya decision heads and Puffer value head |

An encoder-only import needs decision training before its new heads produce
useful answers. Supported encoder families are an explicit registry: classic
BERT with absolute positions and exact GELU, and ModernBERT with its local/global
attention, default RoPE and GEGLU. Unsupported architectures and configuration
variants fail validation. This is not a universal Transformer weight converter.

## Import a model and tokenizer

The offline importer accepts a local Hugging Face safetensors directory or a
Hugging Face model ID. Local imports need Python's standard library; remote
imports additionally need `huggingface_hub`. FP32, FP16 and BF16 safetensors,
including sharded checkpoints, retain their original disk representation and
are converted to FP32 during native loading.
Remote references are resolved to immutable source revisions recorded in the
bundle manifest; an explicit commit makes the requested revision clear.

```sh
python tools/import_pretrained.py convaiinnovations/laya bundles/laya \
    --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851

# Use an existing download without network access:
python tools/import_pretrained.py /path/to/local/laya/snapshot bundles/laya-local

# Encoder-only import from an existing BERT or ModernBERT checkpoint:
python tools/import_pretrained.py /path/to/hf/encoder bundles/encoder \
    --head-layers 2 --max-length 512 --head-max-length 192
```

The destination must not already exist. Token budgets must fit the encoder's
`max_position_embeddings`. A local encoder directory needs `config.json`,
`model.safetensors` or its shard index, and tokenizer files. Laya directories
instead contain `encoder/config.json`, `rl_agent_config.json` and `tokenizer/`.

Use a separately supplied tokenizer with `--tokenizer /path/to/tokenizer` or
`--tokenizer ORGANIZATION/TOKENIZER --tokenizer-revision COMMIT`. Supply
`tokenizer.json` and `tokenizer_config.json`; typed decisions require CLS, SEP,
MASK and PAD token definitions. The importer copies both files and validates
all vocabulary IDs against the encoder embedding size. This establishes valid
ID bounds; the tokenizer's vocabulary must still match the model's training.

The native engine supports tokenizer JSON graphs accepted by Hugging Face
`tokenizers` 0.23.2, including BPE, WordPiece and Unigram. It preserves Unicode
normalization, added tokens, pre-tokenization and post-processing. Custom Python
tokenizer code and conversion from a standalone SentencePiece binary are outside
this import path. Persisted padding/truncation is cleared so each request controls
its own budgets. Model-specific special-token IDs come from the tokenizer,
while the encoder's embedding padding index stays in its model configuration.

Bundle loading checks configuration, tensor names, shapes, extents and finite
values before uploading weights. Read-only file mappings and bounded conversion
buffers avoid a second full-size FP32 host copy. Manifest SHA-256 values record
import provenance; the native loader does not recompute those hashes.

## Build and run typed decisions

Building needs a C/C++ compiler, a CUDA toolkit with cuBLAS, and Rust/Cargo for
the tokenizer static library. Rust 1.90.0 was used for validation. Set `CUDA_HOME`
or `CUDACXX` when CUDA is not on `PATH`; `CARGO`, `CARGO_HOME` and `RUSTUP_HOME`
can select an existing isolated Rust installation. Build scripts do not install
toolchains. Cargo fetches the dependencies pinned in `Cargo.lock` on its first
build. See the [tokenizer build instructions](../../tools/native_tokenizer/README.md).

```sh
./tools/build_native_tokenizer.sh
./tools/build_pretrained.sh
./build/pretrained_decision --bundle bundles/laya --input requests.jsonl

# Reload weights produced by native Puffer training, keeping the same bundle:
./build/pretrained_decision --bundle bundles/laya \
    --weights /path/to/trained-puffer-checkpoint.bin --input requests.jsonl
```

The resulting executable uses native code only. It does not import Python or
PyTorch. The tokenizer exposes a C ABI and C++ wrapper around the Rust static
library. Default CUDA binaries include `sm_80` and, when supported by the toolkit,
`sm_90`, plus a PTX fallback. `NVCC_ARCH=sm_90` can select a specific architecture.

Input is one JSON object per line. Each object has a shared `state` and named
`questions`; several questions can share the same state. These are valid example
lines for the three question types:

```jsonl
{"state":"The road ahead is blocked, but the left lane is clear.","questions":{"move":{"type":"choice","instructions":"Choose the next move.","criteria":{"left":"Turn left","straight":"Continue ahead","right":"Turn right"}}}}
{"state":"Question: What is 2 + 2? Reply: Four.","questions":{"quality":{"type":"score","instructions":"Rate the answer.","criteria":["incorrect","partially correct","correct"]}}}
{"state":"Rain started at noon.","questions":{"supported":{"type":"noul","instructions":"The rain began in the morning."}}}
```

Choice criteria preserve their object order. Score criteria are ordered levels
whose returned score is the probability-weighted level index. Noul represents
false/true options and returns the probability of true; optional criteria and
labels use `false` and `true` keys. The aliases `t`, `ins` and `crit` are accepted.
`option_order` is an optional permutation of option indices; output probabilities
are restored to the original order. String states pass through directly;
structured states and criteria are serialized as JSON with spaced separators.

Output includes answers, option probabilities, confidence and the preserved
act-head probability. Add `--raw` for logits, token IDs, marker positions and
the applied temperature. Omit `--input` to read JSONL from standard input.
`--weights` replaces the imported parameters with a native Puffer flat checkpoint,
including its added critic, while preserving this typed question/answer interface.
The same matching bundle is required because flat weights contain no tokenizer
or architecture metadata. The critic is loaded but is not a typed answer.
Question packing follows pinned Laya source at
[`fa9a2a7`](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/laya/common.py):
literal mask-token text is sanitized, options have token caps, and the remaining
budget holds the state. Truncation is reported; collapsed or missing options
are rejected. Laya's current inference temperature clamp is `[0.5, 5]`, including
its option-count overrides. The source temperature tensor and configuration
calibration remain separate: the English checkpoint tensor is `[1,1,1]`, while
its inference configuration supplies fitted temperatures.

## Fine-tune through Puffer

The included task uses the same benchmark Snake engine as
[`decision_snake`](../decision_snake/README.md), serialized into a textual board
and a choice question with four movement options. Reverse-direction legality is
applied to the option policy. A new value head supplies Puffer's critic. The
imported act/escalate head remains available for typed inference; this Snake PPO
task supplies no act-head supervision.

The exact instruction is:

```text
Choose the next move. Eat food and avoid the walls and snake body.
```

The four options, in action-index order, are `Move up`, `Move down`, `Move left`
and `Move right`. The state explains that rows run top to bottom and columns
left to right, with `0=empty`, `-1=food`, `1=head`, `2=neck`, and larger integers
ordering the remaining body toward the tail. It then supplies the current step,
episode cap and all ten comma-separated board rows. The model receives this
question/options sequence on every decision; it does not generate a text answer.

Use the [paired Snake evaluator](../../tools/evaluate_snake_learning.md) to
compare the imported and trained policies on matched episodes. Food collected
is its primary outcome; survival, raw reward and termination rates are reported
separately.

```sh
./build.sh decision_laya
./build/puffer_decision_laya train --policy.bundle=bundles/laya

# A short smoke run; post-training evaluation also has a small episode budget:
./build/puffer_decision_laya train --policy.bundle=bundles/laya \
    --train.total_timesteps=8 --env.max_steps=2 --base.eval_episodes=2

./build/puffer_decision_laya eval latest --headless \
    --policy.bundle=bundles/laya --base.eval_episodes=32
```

The native Puffer build also uses its usual Raylib/OpenMP/NCCL dependencies;
the standalone inference build does not. This environment requires FP32 CUDA
training. Its initial configuration uses one agent, rollout horizon 4, minibatch
4, learning rate `1e-5`, synchronous collection and disabled CUDA graphs.
The packed training adapter supports bundle `max_len` up to 2048 and encoder
widths divisible by four.
It retains Puffer's training algorithm rather than reproducing Laya's original
training recipe. Sampling, optimizer choice and rewards therefore define a new
fine-tuning experiment. The act head has no task loss, and imported calibration
has not been refitted after fine-tuning. State truncation is rejected in this
training adapter so it cannot quietly omit board information.

Three optional controls support explicit training experiments. All default to
`0`; specify the same settings when evaluating the corresponding policy.

| Setting | Default (`0`) | Optional behavior |
| --- | --- | --- |
| `env.observation_format` | Original complete Snake grid | `1` prepends zero-based head/food coordinates and signed food-minus-head row/column offsets, retaining every grid cell and the original question/options |
| `policy.zero_init_critic` | Historical random initialization of the new value head | `1` initializes its weights and bias to zero while preserving the RNG sequence and every other initial parameter |
| `policy.sequence_length` | Pad execution to the bundle's `max_len` | An integer from `1` through `max_len` selects a smaller execution workspace; an input exceeding it is rejected, never truncated |

For example, opt into all three controls explicitly:

```sh
./build/puffer_decision_laya train --policy.bundle=bundles/laya \
    --env.observation_format=1 --policy.zero_init_critic=1 \
    --policy.sequence_length=384
```

The coordinate summary describes observed geometry; it supplies no recommended
action, collision filter or new reward. A full board reports that food is absent.
The critic setting changes initialization only: loading a trained flat checkpoint
restores its saved critic. Encoder-only imports retain the same initialization of
their other fresh heads. These settings do not change parameter names, shapes or
flat checkpoint layout.

The sequence setting preserves the bundle's packing and calibration. Reducing
padding can alter floating-point results and sampled trajectories, so record it
as part of the protocol and hold it fixed in a primary policy comparison. Token
counts depend on the tokenizer and state; a setting that fits an initial board
must also fit later boards. The CPU observation test checks both Snake formats
and rejects insufficient budgets. These controls are experiment options, not a
claim that any combination improves learning.

Use the same imported bundle when evaluating a trained checkpoint: the bundle
defines architecture, tokenizer, packing and calibration. Puffer's flat weight
files contain model parameters only. They do not contain architecture metadata,
optimizer state, collector state or tokenizer files, and cannot provide an exact
training resume. Puffer evaluation samples actions; benchmark-specific greedy
or clocked evaluation is a separate protocol.

## Validation and current limits

The native encoder and complete decision model have forward/backward parity
tests against Hugging Face and Laya, including every parameter gradient, masks,
repeated embeddings, the added critic and detached act-head confidence features.
Ten small-model numerical tests passed. The actual 421M-parameter English Laya
checkpoint also matched reference option and act logits on three real text
requests covering choice, score and noul. Those checks establish numerical
agreement for the tested cases, not task accuracy or fine-tuning quality.

A full-checkpoint Puffer smoke run completed eight steps and two PPO updates with
`max_steps=2` forcing timeout transitions. Losses and saved weights were finite;
the encoder, option scorer and added critic changed, while the unused act-head
parameters remained exactly unchanged. Checkpoint alignment padding stayed zero.
This verifies training and checkpoint plumbing; it does not establish learning
progress or improved Snake performance. An encoder-only BERT smoke run also
updated its encoder and new decision/critic heads while preserving the unused
act head and finite checkpoint values.

Tokenizer tests cover 130 encodings and 260 decodes across BPE, WordPiece,
Unigram and the actual Laya tokenizer. Bundle tests cover exact dtype conversion,
shards, seven malformed-checkpoint cases and incompatible attention dimensions.
Sequence tests compare eight exact
packing/calibration cases and eight rejected requests with pinned Laya source.
Four production JSONL tests also passed: complete typed outputs and raw model
inputs/logits match Laya, including structured numeric values, option ordering,
custom labels and truncation reporting; invalid requests are rejected, and
reloading the eight-step Puffer checkpoint produces finite, changed predictions.
Seven importer tests cover local imports, required parameter validation and
immutable revision resolution for remote model/tokenizer references.

```sh
./tests/build_pretrained_tests.sh all

# CPU reference checks:
python -m unittest tests.test_import_pretrained -v
python tests/test_native_tokenizer.py --executable build/test_native_tokenizer
python tests/test_pretrained_bundle.py --executable build/test_pretrained_bundle
python tests/test_pretrained_input.py --executable build/test_pretrained_input \
    --bundle bundles/laya
./build/test_decision_laya_observation bundles/laya

# Audit an existing native training checkpoint; use that run's initial seed:
./build/test_laya_checkpoint bundles/laya /path/to/trained-puffer-checkpoint.bin 73

# If the run explicitly zero-initialized its new critic:
./build/test_laya_checkpoint bundles/laya /path/to/trained-puffer-checkpoint.bin 73 \
    --zero-init-critic

# Run these inside an existing GPU allocation:
python tests/test_pretrained_encoder.py --executable build/test_pretrained_encoder
python tests/test_pretrained_decision.py --executable build/test_pretrained_decision

# Optional initialization/padding regression checks. The full gradient test
# requires a small test bundle with max_len=512; use forward-only for full Laya.
./tests/build_decision_policy_options.sh
./build/test_decision_policy_options --host-only
./build/test_decision_policy_options /path/to/tiny-bert-test-bundle
./build/test_decision_policy_options bundles/laya --forward-only
./build/test_decision_policy_options bundles/laya --forward-only --coordinates

# Optional actual-model test; uses existing files without downloading:
python tests/test_pretrained_bundle.py --executable build/test_pretrained_bundle \
    --laya-snapshot /path/to/local/laya/snapshot --laya-bundle bundles/laya

# End-to-end JSONL and trained-weight reload, using the production executable:
python tests/test_pretrained_cli.py --executable build/pretrained_decision \
    --bundle bundles/laya --laya-snapshot /path/to/local/laya/snapshot \
    --weights /path/to/trained-puffer-checkpoint.bin
```

The reference environment needs PyTorch, NumPy, Transformers, `safetensors`,
`tokenizers==0.23.2` and Laya from the pinned source commit above. Tests report
skips when optional reference dependencies or snapshots are absent. Individual
test executables can be built with `encoder`, `decision`, `bundle`, `input`,
`tokenizer`, `observation` or `checkpoint` instead of `all`; an optional second
argument selects the output file. `all` also builds the reusable adapter's
CartPole, layout and Puffer integration harnesses, requiring the same native
Raylib/OpenMP/NCCL dependencies as the trainer. Their individual targets and
commands are in the [CartPole guide](../decision_cartpole/README.md#limits-and-validation).
The tokenizer harness needs only the C++ compiler and tokenizer archive. The
other harnesses need CUDA headers/toolchain to compile; CPU bundle, input,
observation and checkpoint checks do not allocate GPU memory. The observation
test checks packed IDs, option markers, and preservation of timeout observations
across autoreset. The checkpoint audit accepts full Laya and encoder-only bundles;
it requires a trained checkpoint and the run's initial seed to reconstruct fresh
head parameters and verify that the unused act head stayed unchanged. It is not
a generic weight viewer.

The CUDA backend uses FP32 eager attention with quadratic sequence workspace.
Large batches and long contexts have not been validated; start with the supplied
small training configuration. The full Python Laya SDK's HTTP serving, routing,
long-document windowing and other orchestration APIs are outside this native
typed-decision path. Model and tokenizer licenses continue to apply to imports;
the pinned public English Laya checkpoint and Hugging Face tokenizers use
Apache-2.0.
