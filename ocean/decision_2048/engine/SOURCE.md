# Vendored 2048 engine

`g2048.c`, `g2048.h`, `LICENSE`, and `tests/test_g2048.c` are copied unchanged
from [interactive-benchmarks revision f145ef872e41aaf4970ffda39b0e21cda59cc947](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/g2048).
The engine is compiled as C11 and linked through its C ABI; the decision adapter
lives outside this directory.

The source project adapts [PufferLib 2048 at revision 6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/g2048/g2048.h)
under its [MIT license](LICENSE), copyright (c) 2022 PufferAI.
See the [adapter contract](../README.md) for campaign rules and their differences
from stock Puffer 2048.
