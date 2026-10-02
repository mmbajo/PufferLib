# Lights Out engine provenance

This directory vendors the audited standalone Lights Out engine from
[`sbintuitions/interactive-benchmarks`, revision
`f145ef872e41aaf4970ffda39b0e21cda59cc947`](https://github.com/sbintuitions/interactive-benchmarks/tree/f145ef872e41aaf4970ffda39b0e21cda59cc947/envs/lightsout).
The C source, public header, license and engine tests are unchanged.

The original implementation is
[`PufferAI/PufferLib` Lights Out at
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/ocean/lightsout/lightsout.h),
licensed under the [MIT license](LICENSE), Copyright (c) 2022 PufferAI.

The standalone campaign version fixes the board at 5 by 5, applies seeds before
scrambling, excludes initially solved boards, rejects invalid actions without
mutation and separates solving from time limits. The decision adapter uses this
same version so its game rules match the campaign environment.

SHA-256 of the vendored files:

```text
f46cfa631558801a5fc9f140cd149bf25173097a9382c90faa1f3bb2a1ce43fc  lightsout.c
af5a849f4797525a7a774b4dd514127332c50fadbfdff63f8872a51c0ac77bf9  lightsout.h
```
