# Vendored Snake engine

`snake.c`, `snake.h`, and `LICENSE` were copied from
[sbintuitions/decisions, revision 42cf77279324f75b238bd857576c4dff90a024ec](https://github.com/sbintuitions/decisions/tree/42cf77279324f75b238bd857576c4dff90a024ec/envs/snake).
The adapter lives outside this directory; the engine is compiled as C11 and
linked through the header's C ABI.

The local engine now appends a versioned snapshot API to `snake.c` and declares
it in `snake.h` (with the required standard-library includes). The original
gameplay, seed/RNG, reset, step, observation, and legal-action functions are
unchanged. `LICENSE` remains unchanged. The revision links identify the original
upstream files, before the serialization addition.

The source project records that this engine was copied from
[sbintuitions/interactive-benchmarks, revision f145ef872e41aaf4970ffda39b0e21cda59cc947](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/snake).
It adapts [PufferLib Snake at 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/snake/snake.h),
under the [MIT license](LICENSE), copyright (c) 2022 PufferAI.

This benchmark variant has different rules from PufferLib's stock Snake.
See the [adapter contract](../README.md) for its observation and episode semantics.

## Exact state snapshots

`ib_snake_state_size()`, `ib_snake_state_save()`, and `ib_snake_state_load()`
serialize all gameplay state, including the food RNG. Version 2 is a fixed
472-byte record (`IBSNAK02`) with a version, byte count, grid size,
little-endian integers, IEEE-754 binary64 reward/return, ordered active body
cells, and canonical unused cells. Load accepts exactly one complete
record and rejects incompatible versions, nonfinite values, and
invalid game domains before changing the destination. It stores no pointers or
compiler structure padding. Save/load return 0 on success and -1 on rejection.
The outer training record diagnoses version 1 checkpoints with the compatible
old branch, `feat/typed-supervision` at `148add30`; no conversion is performed.

The adapter's version 2 (`PUFSNK02`) record is 520 bytes for both raw and text
Snake: a 16-byte header, the next seed, seven log floats, and the 472-byte engine
record starting at byte 48. `puf_state_validate` produces an `EnvState` containing
the decoded engine and log without touching the live environment. The loader
then calls `puf_state_load` to apply it, or `puf_state_discard` after failed
validation elsewhere. Engine domains and log ranges are checked only on load.
Configuration belongs to the outer training metadata. Host transitions and Agent
buffers are replaced by the next step; collector state and sampling RNG are saved
separately by the training core. Old unpublished version 2 review artifacts are
not a compatibility target for the smaller record.

Run the CPU checks with:

```sh
python3 -m unittest tests.test_decision_snake tests.test_decision_snake_state -v
tests/build_decision_state_tests.sh
build/test_decision_laya_state /path/to/imported/laya-bundle
```

The first command needs only C and C++ compilers. The Laya harness needs the CUDA
toolchain for existing model declarations and a tokenizer bundle, but creates no
CUDA context and uses no GPU. Tests cover exact continuation through all food
spawns to a full board, timeout/collision autoreset, wrapped episode seeds,
side-effect-free validation, and transactional rejection of malformed records.
