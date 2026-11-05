# Training worlds

Throwaway rAthena worlds for bots, started inside Slurm jobs and deleted when the jobs end. They
never touch the play server. Submit every job with `./submit.sh`, which keeps off the nodes this
account already uses; job output and server logs go to `~/ragnarok/train/logs`.

| File | What |
|---|---|
| `world.sh` | Library sourced by the jobs: one MariaDB per job, then any number of worlds, each with its own ports (7000+10W, 7001+10W, 7002+10W on 127.0.0.1), database and bot accounts. Passwords are random per job and never printed. The jobs take the rAthena build from `RATHENA` (default `~/ragnarok/rathena-re`, client 20221005 with renewal rules), lean worlds take their monsters from its rules (`npc/pre-re` or `npc/re`), and `world_check_client` stops a job whose bot client speaks another version or rule set. |
| `test-bot.sbatch NBOTS MINUTES DIE_AFTER` | One world with scripted bots; the first bot is a GM and types `@die` once. `MODE=lockstep` (pass `--export=ALL,MODE=lockstep`) uses the lockstep build. |
| `bench-worlds.sbatch NWORLDS NBOTS MINUTES RESET` | Lean lockstep worlds side by side, for throughput. Give it 2 CPUs per world plus a few (`./submit.sh -c 20 ...` for 8 worlds). |
| `build-train.sbatch` | Builds `~/ragnarok/rathena-re` (branch `local/lockstep`): client 20221005 with renewal rules, plus the lockstep patch and 8192 connections per server. `--export=ALL,RATHENA=<dir>,PACKETVER=<date>,PRERE=yes` builds another checkout, version or pre-renewal (`~/ragnarok/rathena-train`: 20130618 pre-renewal). |
| `env-check.sbatch NBOTS STEPS [ACTION_REPEAT]` | The PufferLib env without the trainer (`ocean/ragnarok/ragnarok_check`): random actions, then the scripted policy on each task through the env's own actions. |
| `world-age.sbatch NBOTS ROUNDS [TASK]` | Does a world change as it ages? The scripted policy plays ROUNDS rounds of 10 game minutes on one world, each on fresh accounts, and the field's monster census is printed. |
| `build-env.sbatch` | Builds the env's trainer (for H100) and `--cpu` binary in the toolchain container on a CPU node, to catch compile errors before a GPU job. |
| `train.sbatch NWORLDS NBOTS TIMESTEPS [RUNS_FILE]` | Training on one GPU: the worlds on the node, the trainer in the toolchain container; checkpoints and logs in `/lustre/users/user02198/ragnarok/runs/<job>`. A runs file holds one run per line (extra `./puffer train` arguments); every run gets freshly started worlds. `BUILD_FLAGS=--float` builds with float32 rollout buffers. Runs on `002-partition-default` with a 24 h limit (training always uses the default partition, with limits over an hour); `-t` on `submit.sh` changes the limit. |
| `build_image.sbatch`, `smoke.sbatch`, `probe.sbatch` | The toolchain image (CUDA 12.2, clang 15) and its checks; see `TOOLCHAIN.md`. |

The scripts that lean worlds load are in `../server/npc/`:

- `rl_train.txt`: `@rlreset <task>`, the episode reset, and the list of task maps (`RG_TASK_MAPS` in
  `world.sh` is the same list, for the maps a lean world loads);
- `rl_spawns.txt`: extra monsters on the task maps (the training field, and Spores on `pay_fild01`);
- `rl_census.txt`: every 10 game minutes, each task map's monster count and its monsters in a 4x4
  grid, in the map server's log;
- `rl_spread.txt`: every 5 game minutes, a fifth of each task map's idle monsters move to random
  cells, so the maps stay evenly stocked (see the lessons below).

The rAthena patches are in `../server/patches/`.

## World settings

- New characters start on `prt_fild08` (200,200) with a Main Gauche worn, a Cotton Shirt and 30
  Red Potions; rates are rAthena's defaults (1x), as on your server.
- Flood protection lets 127.0.0.1 through (it bans an address after 5 connections in 3 s), idle
  connections last 300 s, and SQL logging is off.
- `WORLD_LEAN=1` loads only the training field, its monsters and `npc/*.txt`: a world starts in
  about 2 s and its map server uses about 40 MB.
- Lockstep worlds (`lockstep: yes`) keep the clock still until every bot has sent READY, then run
  100 ms of game time. Autosave is every hour of game time.

## Results so far (2026-11-05)

| Run | Bots | Game time | Wall time | Bot steps/s | Notes |
|---|---|---|---|---|---|
| Real time, one world | 4 | 10 min | 10 min | 40 | 226 kills, 1,180 base exp per bot-hour |
| Lockstep, one world | 4 | 10 min | 0.9 s | 26,800 | 230 kills, 1,086 base exp per bot-hour (same game, within noise) |
| Lockstep, one world | 64 | 10 min | 6.0 s | 63,800 | |
| Lockstep, 8 lean worlds, 20 CPUs | 8 x 64 | 10 min | about 5.6 s | 549,000 in total | `@rlreset` every 2 game minutes |
| Same, with 2.7x the easy monsters | 8 x 64 | 10 min | about 6.5 s | 473,000 in total | 1,100 base exp per bot-hour again, as with 4 bots |
| Env, scripted policy, one world | 64 | 10 min | 6.1 s | 62,600 | 33 base exp per 2-minute episode, as the scripted bot |
| Training, 16 worlds, 1 H100, 64 CPUs | 16 x 64 | | 100M steps in about 100 s | about 1,000,000 | 0 to 14 kills per 5-minute episode by 30M steps, then flat (job 5004735) |
| Same, potions only below 70% HP | 16 x 64 | | 500M steps in 9 min | about 1,000,000 | perf 0.88 of the scripted bot at 30M steps, then down to 0.40 by 500M; deaths and potion use fall too (job 5004782) |

## Lockstep lessons

- A bot's first READY can miss the step that is running; `ro_lockstep_join()` handles that, and
  `ro_lockstep_step_all()` targets the step after the newest DONE any bot has seen.
- A world whose bots are all gone steps itself in real time, or logouts would never finish.
- rAthena keeps a logged-out account "online" for a while; a second run on the same world should
  use other accounts (`RG_ACCOUNT_OFFSET`).
- A world played by a learning policy changes as it ages, though the scripted bot leaves it as it
  was (`world-age.sbatch`: 50 rounds over 8 game hours, perf 0.87 to 1.08, 345 to 359 monsters on
  the field). Under a learned policy the monsters gather into a few clumps: in job 5005388 the
  busiest 100x100 square of `prt_fild08` grew from about 50 to up to 195 of its 360 monsters in 8
  game hours, and most of the map emptied. The likely cause: a monster that is hit chases the bot
  that hit it and stays where it loses track when that bot is reset elsewhere, so a policy that
  starts fights and walks off drags monsters around. A later run on
  the same worlds then started on an empty field and never learned (9 runs of 9). Hence fresh
  worlds for every run and `rl_spread.txt`.
