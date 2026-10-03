# Vendored Snake engine

`snake.c`, `snake.h`, and `LICENSE` were copied from
[sbintuitions/decisions, revision 42cf77279324f75b238bd857576c4dff90a024ec](https://github.com/sbintuitions/decisions/tree/42cf77279324f75b238bd857576c4dff90a024ec/envs/snake).
The adapter lives outside this directory; the engine is compiled as C11 and
linked through the header's C ABI.

The local engine now appends a versioned snapshot API to `snake.c` and declares
it in `snake.h` (with the required standard-library includes). The original
gameplay, seed/RNG, reset, step, observation, and legal-action functions are
unchanged. `LICENSE` remains unchanged. The hashes below identify the original
upstream files, before the serialization addition.

| File | SHA-256 |
| --- | --- |
| `snake.c` | `ea33f88b2389ac3a367eec5c72d880051e32ffe3fcc7087f4b5ebcac7902795d` |
| `snake.h` | `b2733e2b329a8ce448f20513c690e840c258d51b34969e016ed975cb5188646a` |
| `LICENSE` | `24dae53fbe9da4b15693499b043783c7572b894ac9a5dbc6889856a409c24dee` |

The source project records that this engine was copied from
[sbintuitions/interactive-benchmarks, revision f145ef872e41aaf4970ffda39b0e21cda59cc947](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/snake).
It adapts [PufferLib Snake at 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/snake/snake.h),
under the [MIT license](LICENSE), copyright (c) 2022 PufferAI.

This benchmark variant has different rules from PufferLib's stock Snake.
See the [adapter contract](../README.md) for its observation and episode semantics.

## Exact state snapshots

`ib_snake_state_size()`, `ib_snake_state_save()`, and `ib_snake_state_load()`
serialize all gameplay state, including the food RNG. Version 1 is a fixed
476-byte record with a magic identifier, version, byte count, grid size,
little-endian integers, IEEE-754 binary64 reward/return, ordered active body
cells, canonical unused cells, and CRC32. Load accepts exactly one complete
record and rejects incompatible versions, bad checksums, nonfinite values, and
invalid game domains before changing the destination. It stores no pointers or
compiler structure padding. Save/load return 0 on success and -1 on rejection.

The shared adapter exposes `PUF_ENV_STATE` and the static callbacks
`puf_state_size`, `puf_state_save`, `puf_state_validate`, and `puf_state_load`.
These wrap the engine record with a separately versioned/checksummed record of
the seed schedule, logs, last accepted pre-autoreset transition, and live host
observation, action mask, action, reward, and done buffers. They work for both
raw Snake and packed Laya observations. Load requires matching observation size,
episode cap, observation format, and agent policy; Laya additionally validates
packed token bounds against the configured model/tokenizer. The caller must
bind the destination buffers before loading. Agent buffer bindings and render
resources remain owned by the destination process. Validation never mutates it.
These callbacks cover environment state only; model/optimizer state, collector
buffers, and action-sampling RNG must be checkpointed by the training core.

Run the CPU checks with:

```sh
python3 -m unittest tests.test_decision_snake tests.test_decision_snake_state -v
tests/build_decision_state_tests.sh
build/test_decision_laya_state /path/to/imported/laya-bundle
```

The first command needs only C and C++ compilers. The Laya harness needs the CUDA
toolchain for existing model declarations and a tokenizer bundle, but creates no
CUDA context and uses no GPU. Tests cover exact continuation through all food
spawns to a full board, timeout/collision autoreset, pending transitions, wrapped
episode seeds, and transactional rejection of damaged/domain-invalid records.
