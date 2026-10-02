# Vendored Connect Four engine

`connect4.c`, `connect4.h`, `tests/test_connect4.c`, and `LICENSE` are copied
unchanged from [sbintuitions/interactive-benchmarks, revision
f145ef872e41aaf4970ffda39b0e21cda59cc947](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/connect4).
The adapter lives outside this directory. The engine compiles as C11 and exposes
the header's C ABI.

| File | SHA-256 |
| --- | --- |
| `connect4.c` | `7e6b9be183eb35b25bae5bfd65135991beb2b21c7604f09bf2c462f66d57d016` |
| `connect4.h` | `bb169e753b62df3136c5e2d70fc146ffb372793069ae3e02fc40b8e3bb747d76` |
| `LICENSE` | `24dae53fbe9da4b15693499b043783c7572b894ac9a5dbc6889856a409c24dee` |

The source adapts [PufferLib Connect Four at
6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/connect4/connect4.h)
under the [MIT license](LICENSE), copyright (c) 2022 PufferAI. It preserves the
fixed native opponent, including its opening book, depth-three recursive
evaluation, move order, and `rand_r` tie-breaking. The campaign port rejects
illegal moves without mutation, preserves final boards, adds explicit time
limits, and fixes the upstream full-board draw mask. See the
[adapter contract](../README.md).
