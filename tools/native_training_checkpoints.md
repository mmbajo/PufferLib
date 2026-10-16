# Native training checkpoints

Native training has two distinct loading modes. They are mutually exclusive.

| Option | Restored state | Intended use |
| --- | --- | --- |
| `base.load_model_path=model.bin` | FP32 model weights, including critic | Start a new optimization run from an existing model |
| `base.resume_path=STEP.train` | Model, optimizer, counters, RNG, environment and recurrent carry | Continue a saved run at its next rollout |

Previously `load_model_path` was honored by evaluation but ignored by the native
training entry point. Training now applies it before the first rollout and before
initializing selfplay opponents. The asynchronous actor snapshot is updated too.
Flat files must have exactly the expected parameter count and only finite values.
They still contain no architecture or tokenizer metadata: use the matching policy
settings and, for text policies, the matching imported bundle.

## Warm start

Build the same policy/environment, then use an existing flat checkpoint:

```sh
./build/puffer_decision train --headless --base.run_id=warm \
    --base.load_model_path=checkpoints/decision_snake/old/0000000000512000.bin
```

This starts fresh optimizer moments, schedules, action/environment RNG and episode
state. Old `.bin` files cannot reconstruct those states and cannot be used as full
resumes. A warm start may change the seed, training budget or optimization recipe.

## Save and resume training

Full-state saves are opt-in. Flat weight checkpoints continue to be written for
evaluation. With `base.save_training_state=1`, each scheduled checkpoint also
creates a `.train` directory. The final completed rollout always gets a checkpoint.

This example uses the board Snake defaults (32 agents, horizon 32, one GPU):

```sh
./build.sh decision_snake build/puffer_decision

# Plan a 512K-step schedule but stop safely at its 256K rollout boundary.
./build/puffer_decision train --headless --base.eval_episodes=0 \
    --base.run_id=part1 --train.total_timesteps=512000 \
    --base.save_training_state=1 --base.stop_after_steps=256000

# The same executable resumes the original schedule, using a new output run ID.
./build/puffer_decision train --headless --base.eval_episodes=0 \
    --base.run_id=part2 --train.total_timesteps=512000 \
    --base.save_training_state=1 \
    --base.resume_path=checkpoints/decision_snake/part1/0000000000256000.train
```

`stop_after_steps` is an absolute global step, not an additional budget. It requires
state saving and must be a positive multiple of
`train.gpus * vec.total_agents * train.horizon`, at most `total_timesteps`. Zero
runs the complete planned budget. Keep `total_timesteps` unchanged across a resume:
it determines the learning-rate and entropy schedules. A resume at or beyond the
requested stopping step is rejected. Full resume accepts an explicit directory;
`latest` remains a convenience only for flat model loading.

The same controls work with `decision_laya`, including encoder-only BERT imports,
full Laya bundles and optional coordinate observations. Pass the original bundle,
policy options, training settings, environment settings and per-rank batch layout
on both commands. Data parallelism and Muon's whole-matrix state sharding are
supported when those settings remain identical. Adam resumes restore both FP32
moment buffers and its bias-correction step; its moments remain replicated and
require `train.distributed_optimizer=0`. The optimizer cannot change on resume.

## Compatibility and failure handling

Version 2 supports Linux little-endian LP64, FP32, `base.async=0`,
`base.cudagraphs=-1`, one policy, and CPU environments with explicit state codecs.
Currently those environments are board Snake and text Snake. Selfplay, CUDA
environments, BF16, asynchronous collection and changing world size are rejected.
Other environments can implement the `PUF_ENV_STATE` codec; no raw Env structs
or pointer values are persisted.

The loader compares the producing executable's canonical path, CUDA RNG record
size, world size, and resolved `env`, `policy`, `train` and `vec` settings. Relevant base seed/collection
controls are included. Configuration values are compared in their parsed string
representation, so use the same spelling for numeric settings too. Text policies
also record the canonical bundle path, parsed encoder and decision configuration,
packing budgets, special tokens and the active choice temperature. These direct
values catch architecture changes even when the flat parameter count is unchanged.
Output, logging, checkpoint and stopping paths may change.

Keep the producing executable and imported bundle at their recorded paths. The
saved model replaces the initial imported weights; the bundle still constructs
the architecture and tokenizer. Content replacement at the same path is not
validated. Record code by branch and commit and binaries/bundles by path with the
run; retaining those assets is required for compatible CUDA RNG and tokenizer
behavior. Older flat weights remain usable for warm starts.

The version 2 markers are `PUFFER-TRAINING-COMMIT-2` and
`PUFFER-TRAINING-RANK-2`. Version 1 records are refused before interpreting their
payload with `record version 1 requires feat/typed-supervision at 148add30; use
its producing executable`. That branch retains the old reader, including Adam.
There is no migration of version 1 records. The unpublished version 2 review
artifacts used a larger schema; they are scratch artifacts and are not supported
by this final version 2 reader.

Every rank saves model weights, its optimizer moments/counter, action and minibatch
RNG, recurrent carry, device environment buffers, pending timeout observations,
and environment continuation state. Snake retains its engine, log and next seed
in a 520-byte record for both observation formats. Epoch/global-step counters are
common metadata; the next update derives LR and entropy coefficient from the epoch. Parameter and RNG buffers restore byte-for-byte. Temporary
forward/backward workspaces are overwritten by the next rollout/update and are
not saved. Loss and environment accumulators are retained; dashboard histories,
profiling totals and wall-clock throughput start a new process session.

Every save creates a random 32-byte coordination value, shared through a temporary
`ID` file and copied into the manifest and rank headers. The temporary file is
removed before publication. Rank headers also carry their rank number,
configuration metadata and common epoch/step counters. Records contain length-prefixed
strings, little-endian 64-bit counts and the raw named buffers, with no payload
trailer. Readers check the expected buffer names and sizes before allocating,
then require exact end of file.

All ranks validate their complete state before any rank applies it. Snake records
are decoded once into temporary states and applied after the collective barrier;
failed validation discards the temporary states without changing live state. Missing or
misplaced ranks, incomplete files, trailing bytes, incompatible metadata,
nonfinite float state, invalid Adam moments/counters and malformed environment
records fail loading. A rank from another save fails the direct coordination-value
comparison even when its configuration and counters match. Arbitrary changes to
otherwise valid finite payload bytes are not detected; checkpoints are trusted
local artifacts.

Saving uses an exclusive `.partial` directory and fsynced rank files. Rank zero
writes and fsyncs `COMMITTED` there, syncs the staging directory, then renames it
to the destination and syncs the parent directory. Existing destinations are
refused before saving. A crash before publication can leave only the staging
directory. Its name does not determine readability: a complete `COMMITTED`
record admits it; a missing or incomplete record refuses it. Preserve abandoned
staging directories for inspection and use a new output run ID to avoid collisions.
All ranks must see the same shared checkpoint filesystem. A failed rank causes
the native supervisor to terminate its peers, including peers waiting at a
checkpoint collective.

Full snapshots are larger and slower than flat weight saves. Every rank stores a
copy of the model, while optimizer shards and environment/RNG state are rank-local.
Loading stages each rank's saved state in host memory for validation before GPU
mutation. Budget disk space and host RAM accordingly.

## Reproducibility and tests

Exact state restoration does not make subsequent Transformer optimization
bitwise reproducible. Existing embedding-gradient kernels use floating-point
atomic additions; separate training runs, including runs with identical seeds,
can diverge after backward passes. Checkpoint tests require byte-exact restored
state, identical next rollout, and one identical fixed-gradient optimizer update
before another backward pass. The fixed LR isolates optimizer-state restoration
from the schedule, and the update exposes a missing momentum or moment buffer. They also run the
actual split/resume CLI, check counters and cosine schedules, and report final
split-versus-uninterrupted weight differences without treating them as equality.

```sh
c++ -std=c++17 -O2 tests/test_native_checkpoint_io.cpp -o build/test_native_checkpoint_io
./build/test_native_checkpoint_io
python3 -m unittest tests.test_decision_snake tests.test_decision_snake_state -v
./tests/build_native_checkpoint.sh decision_snake

# On a host with two CUDA GPUs; output directory must not exist.
python3 tests/run_native_checkpoint_tests.py \
    --probe build/test_native_checkpoint_decision_snake \
    --trainer build/puffer_decision --output build/checkpoint-validation --gpus=2

# Repeat with Adam and its replicated moments.
python3 tests/run_native_checkpoint_tests.py \
    --probe build/test_native_checkpoint_decision_snake \
    --trainer build/puffer_decision --output build/checkpoint-adam-validation \
    --gpus=2 --optimizer=adam
```

The GPU runner checks one and two ranks, replicated Adam or replicated/sharded Muon, fresh-process
rollout and fixed-gradient update replay by comparing the raw `.next-save-RANK`
and `.next-load-RANK`
files, malformed/incompatible checkpoint rejection, real training warm-start
behavior and full resume. Rejected loads also compare every live buffer and both
training counters with their pre-load values. The adversarial cases include
an invalid later environment after successful earlier validation, nonfinite weights,
negative Adam second moments,
wrong Adam steps, mixed saves, misplaced ranks and incomplete publication. The text probe uses the same CUDA test with
`./tests/build_native_checkpoint.sh decision_laya`; pass its imported bundle after
the `save|load PATH GPUS SHARD` arguments.
