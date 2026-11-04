# ragnarok: bots on a real Ragnarok Online server

A PufferLib env whose agents play Ragnarok Online on throwaway [rAthena](https://github.com/rathena/rathena)
servers. rAthena is the open-source RO server, and it ships its own maps and data, so no game files
are needed. The bots speak the real client protocol (client 2013-06-18, pre-renewal), so a policy
trained here can also play on a normal server next to people.

## How it works

```
one GPU job on one node
├── MariaDB                          node-local, deleted when the job ends
├── world 0: login, char, map        own ports and database; map server in lockstep
├── world 1 … world N-1
└── ./puffer train ragnarok
      env i = world i with num_agents bots; each bot is a TCP client (client/rathena_client.h)
```

- **Lockstep.** A patch to rAthena (`server/patches/`) lets the clients step the map server's clock.
  - The clock stands still until every bot has sent its actions and a READY packet.
  - The server then runs 100 ms of game time, one timer at a time, and answers DONE.
  - So the world runs as fast as the CPU allows, and it waits while the trainer is busy.
  - With the patch switched off (the default), the same build serves a normal real-time world.
- **Episodes.** A GM-level script in the training worlds (`server/npc/rl_train.txt`, `@rlreset`)
  resets a character at the start of each episode. Real-time servers have no such script.
- **The env** (`ragnarok.h`).
  - It turns actions into requests: walk, attack, pick up, potion, sit.
  - Attacks and pick-ups carry on over the next steps: rAthena never walks a player into range, so
    the env does it.
  - Observation: the bot's own state, the walkable cells around it, the nearest monsters and the
    nearest items on the ground.
  - Reward: exp gained divided by the exp the level needs.
  - `perf`: exp per step relative to a hand-written bot (1.0 = as good as that bot).
- **One env per world**, since a world steps only when all of its bots are ready.

## Layout

| Path | What |
|---|---|
| `ragnarok.h` | The env |
| `ragnarok_check.c` | Drives one env without the trainer: random actions, or a scripted policy that uses only the env's actions |
| `client/` | The C client, its generated packet and monster tables, the scripted bot (see its README) |
| `server/patches/` | rAthena patches: packet obfuscation off, lockstep stepping, lockstep fixes |
| `server/npc/` | Scripts the training worlds load: the episode reset and extra monsters |
| `slurm/` | Jobs that start training worlds, test them, benchmark them and train (see its README and `TOOLCHAIN.md`) |

## Running it

1. Build rAthena with the patches:
   - `git am server/patches/*.patch` on rAthena `d4b8e7b`;
   - then `./configure --enable-epoll --enable-packetver=20130618 --enable-prere=yes --with-maxconn=8192`
     and `make server` (`slurm/build-train.sbatch`).
2. Build the toolchain image once (`slurm/build_image.sbatch`): CUDA 12.2 and clang 15.
3. Train: `slurm/submit.sh slurm/train.sbatch NWORLDS NBOTS TIMESTEPS [RUNS_FILE]`. The job
   starts the worlds, builds the env and runs `./puffer train` (or each line of `RUNS_FILE`).

`slurm/env-check.sbatch` and `slurm/bench-worlds.sbatch` test a world and the env without a GPU.

## Results so far

| Setup | Bot steps/s | Note |
|---|---|---|
| Real time, 4 scripted bots | 40 | 226 kills in 10 minutes |
| Lockstep, 1 world, 4 scripted bots | 26,800 | 230 kills in 10 game minutes: same game, 670x faster |
| Lockstep, 1 world, 64 scripted bots | 63,800 | |
| Lockstep, 8 lean worlds of 64 bots, 20 CPUs | 470,000 to 550,000 | |
| Training: 16 worlds of 64 bots, one H100, 64 CPUs | about 1,000,000 | 500M steps in 9 minutes |

First training runs: the policy reaches 0.88 of the scripted bot's exp per step at 30M steps, then
drifts back toward random, and its entropy rises while its KL falls to zero. Fixing that drift
(reward scale, entropy bonus, learning rate) comes before anything else.
