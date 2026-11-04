# rAthena client

A small Ragnarok Online client in C for bots on our own rAthena servers. It speaks client version
2013-06-18, pre-renewal, with packet obfuscation off, and compiles as C11 and as C++17, since the
trainer compiles the env with nvcc.

| File | What |
|---|---|
| `rathena_client.h` | The client: login, character creation, a world model (own state, units, items on the ground, inventory, skills, the map's walkable cells), actions, and a real-time or lockstep clock |
| `rathena_packets.h` | Generated, do not edit: packet ids, lengths and C structs for this client version |
| `rathena_mobs.h` | Generated, do not edit: monster level, HP, attack, defense, element, aggressiveness and boss flag, so the env can describe monsters by their properties |
| `scripted_bot.c` | Hand-written bots that level novices: the baseline, and the end-to-end test |
| `gen/gen_packets.py`, `gen/gen_mobs.py` | Write the two generated headers from rAthena's own source and database |

## Build

```
gcc -std=gnu11 -O2 -Wall -Wextra -o scripted_bot scripted_bot.c -lz -lm -lpthread
python3 gen/gen_packets.py [rathena_dir]   # after rAthena's packets change
python3 gen/gen_mobs.py [rathena_dir]
```

`gen_packets.py` compiles small programs against rAthena's headers. Every mirrored struct gets a
size check at compile time, so a layout that drifts from rAthena fails to build.

## Things the protocol taught us

- This client version shuffles the action packet ids: walk `0x088e`, attack `0x0889`, pick up
  `0x0962`, keepalive `0x0930`, map enter `0x095b`. The generator takes them from
  `clif_shuffle.hpp`.
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
