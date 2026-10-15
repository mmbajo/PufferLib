# Vendored Snake engine

`snake.c`, `snake.h`, and `LICENSE` are copied unchanged from
[sbintuitions/decisions, revision 42cf77279324f75b238bd857576c4dff90a024ec](https://github.com/sbintuitions/decisions/tree/42cf77279324f75b238bd857576c4dff90a024ec/envs/snake).
The adapter lives outside this directory; the engine is compiled as C11 and
linked through the header's C ABI.

The source project records that this engine was copied from
[sbintuitions/interactive-benchmarks, revision f145ef872e41aaf4970ffda39b0e21cda59cc947](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/snake).
It adapts [PufferLib Snake at 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/snake/snake.h),
under the [MIT license](LICENSE), copyright (c) 2022 PufferAI.

This benchmark variant has different rules from PufferLib's stock Snake.
See the [adapter contract](../README.md) for its observation and episode semantics.
