# Vendored Snake engine

`snake.c`, `snake.h`, and `LICENSE` are copied unchanged from
[sbintuitions/decisions, revision 42cf77279324f75b238bd857576c4dff90a024ec](https://github.com/sbintuitions/decisions/tree/42cf77279324f75b238bd857576c4dff90a024ec/envs/snake).
The adapter lives outside this directory; the engine is compiled as C11 and
linked through the header's C ABI.

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
