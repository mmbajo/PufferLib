# rAthena client

A small Ragnarok Online client in C for bots on our own rAthena servers. It speaks the client version
and rules its generated packet table is for, with packet obfuscation off: 2013-06-18 pre-renewal and
2022-10-05 renewal (this branch's table) are tested. It compiles as C11 and as C++17, since the
trainer compiles the env with nvcc.

| File | What |
|---|---|
| `rathena_client.h` | The client: login, character creation, a world model (own state, units, items on the ground, inventory, skills, the map's walkable cells), actions, and a real-time or lockstep clock |
| `rathena_packets.h` | Generated, do not edit: packet ids (`RO_HEADER_*`, `RO_TYPE_*`, `RO_CZ_*`), lengths, C structs and which fields they have (`RO_HAS_*`), for one client version and rule set |
| `rathena_mobs.h` | Generated, do not edit: monster level, HP, attack, defense, element, aggressiveness and boss flag, so the env can describe monsters by their properties |
| `scripted_bot.c` | Hand-written bots that level novices: the baseline, and the end-to-end test |
| `gen/gen_packets.py`, `gen/gen_mobs.py`, `gen/gen_atlas.py` | Write the generated headers from rAthena's own source and database: packets, monsters, and the travel region's maps and warps (`rathena_atlas.h`) |

## Build

```
gcc -std=gnu11 -O2 -Wall -Wextra -o scripted_bot scripted_bot.c -lz -lm -lpthread
python3 gen/gen_packets.py [rathena_dir] [packetver] [pre-re|re] [out.h]   # after rAthena's packets change
python3 gen/gen_mobs.py [rathena_dir] [pre-re|re] [out.h]
python3 gen/gen_atlas.py [rathena_dir] [pre-re|re] [map ...]   # the travel region: RG_TRAVEL_MAPS in slurm/world.sh
```

Both generators default to the version and rules the rAthena checkout was configured with (its
`src/map/Makefile`), so after a build `python3 gen/gen_packets.py ~/ragnarok/rathena-re` writes the
matching table. `gen_packets.py` compiles small programs against rAthena's headers; run it on a
compute node. Every mirrored struct gets a size check at compile time, so a layout that drifts from
rAthena fails to build. To build against another table without replacing these, pass
`-DRO_PACKETS_H='"path/rathena_packets.h"'` and `-DRO_MOBS_H='"path/rathena_mobs.h"'`. The training
jobs stop (`world_check_client` in `slurm/world.sh`) when the table and the server's build differ.

## Things the protocol taught us

- Client 2013-06-18 shuffles the action packet ids: walk `0x088e`, attack `0x0889`, pick up
  `0x0962`, keepalive `0x0930`, map enter `0x095b`. The generator takes them from
  `clif_shuffle.hpp`. Clients after 2018-03-07 no longer shuffle (or obfuscate) them.
- Newer versions move ids and layouts, and the client follows the table's names:
  - From 2013-12 on, the unit packets carry a unit's id in `AID`; `GID` is a player's character id
    and 0 for monsters.
  - From 2017-08 on, exp comes in `ZC_LONGLONGPAR_CHANGE` (`0x0acb`, 64-bit).
  - From 2018-10 on, the inventory lists have a byte saying whose list it is (cart and storage
    share their ids).
- rAthena's packet db only parses what clients send, and some of its lengths for server packets
  are stale: from 2018-11 on, an item drop (`0x0add`) is 24 bytes, not 22, and a sprite change
  (`0x01d7`) 15, not 11. The generator sizes such packets by the struct the server sends.
- The server never walks a player into attack range: it answers `0x0139` ("too far"), and the
  client has to approach first. Monsters chase on their own.
- The map server doesn't resend what the character list already said (levels, exp, zeny), so the
  client keeps those from the char server.
- Login passwords hold 23 characters at most.
- Lockstep: a bot's first READY can miss the step that is running, so `ro_lockstep_join()` keeps
  steps going until every bot has had a DONE. `ro_lockstep_step_all()` then targets the step after
  the newest DONE any bot has seen, which cannot deadlock.
- zlib's header clashes with a type the trainer defines (`Byte`), so the client declares the one
  zlib function it needs.
