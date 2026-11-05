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
- **The env** (`ragnarok.h`) trains one goal-conditioned model on several tasks at once.
  - Each episode, a bot draws a task (`taskN_weight`), and `@rlreset <task>` sets up its character:
    job, levels, stats, skills, gear and map.
  - With `curriculum = 1`, one sampler per trainer process draws the tasks instead. A task unlocks
    once the tasks it builds on reach perf 0.8, or have had 20,000 episodes (the class skills and
    navigation build on field leveling, survival on the Swordman skills). Unlocked tasks are drawn by learning progress
    (how fast their perf is changing), each keeping at least 5% of the episodes. The logs show each
    task's chance of being drawn as `<task>_chance`.
  - The observation starts with a goal vector (task, difficulty, job) and then describes:
    - the bot itself: HP, SP, levels, exp, weight, casting, skills;
    - the 15×15 walkable cells around it;
    - the 8 nearest monsters, by their properties (level, HP, element, aggressive, boss) rather
      than their ids;
    - the 4 nearest items;
    - the effects on the bot: status icons such as Blessing, with the share of their time left, and
      ailments (stone, frozen, stunned, asleep, poisoned, other), which each monster shows too.
    - It reserves room for party members, dialogue and more, so later tasks don't change its size.
  - Actions:
    - a verb: wait, move, attack, pick up, potion, sit, or skill;
    - a direction;
    - a monster or item slot;
    - one of 24 first-class skills.
    - Masks rule out what can't be used right now.
  - Attacks and pick-ups carry on over the next steps: rAthena never walks a player into range, so
    the env does it. A skill on a monster does not start an attack, so a Mage keeps its distance.
  - Reward: exp gained divided by the exp the level needs, normalized by the task's scripted
    baseline so that every task pays the same at that level.
  - `perf`: exp per step relative to the task's scripted bot (1.0 = as good as that bot), logged
    per task as `<task>_perf`. Navigation pays the progress toward its target plus 1 on arrival,
    and its perf is the speed on the way there.
- **One env per world**, since a world steps only when all of its bots are ready.

Every class task is a first class at levels 25/25 with its skills, weapon and 30 Red Potions, on a
renewal field (client 20221005) whose monsters are near its level. The baselines are the scripted bots
in `ragnarok_check.c` on renewal (job 5007175).

| Task | Character | Map | Baseline (exp per step) |
|---|---|---|---|
| 0 `field` | Novice, level 1, Main Gauche, 30 Red Potions | `prt_fild08`: Porings, Lunatics, Fabres | 2.31 |
| 1 `sword_skills` | Swordman: Bash 10, Magnum Break 5, Blade | `pay_fild01`: Spores (added), Willows | 2.29 |
| 2 `survival` | Swordman 18/18, the same skills, 10 potions | `moc_fild02`: Peco Pecos (level 25), which help each other | 2.38 (0.25 deaths per episode) |
| 3 `mage_skills` | Mage: Fire, Cold and Lightning Bolt 5, Soul Strike 3, Napalm Beat 3, Fire Ball 1, Frost Diver 1, Rod | `mjolnir_01`: mostly Caramels (earth), some wind | 2.25 |
| 4 `archer_skills` | Archer: Double Strafe 10, Arrow Shower 5, Owl's and Vulture's Eye 5, Bow, 2,000 Arrows | `pay_fild07`: Bigfoot, Caramel, Stainer, Creamy | 3.54 |
| 5 `acolyte_skills` | Acolyte: Heal 5, Blessing 5, Increase AGI 5, Club | `pay_fild01` | 2.65 |
| 6 `thief_skills` | Thief: Double Attack 10, Envenom 5, Steal 3, Hiding 2, Main Gauche | `pay_fild07` | 2.94 |
| 7 `merchant_skills` | Merchant: Mammonite 10 with 200,000 zeny, Axe | `mjolnir_01` | 2.96 |
| 8 `navigation` | The field's Novice, walking to a cell 10–120 cells away; the goal gives the way there | `prt_fild08` | 0.394 cells per step (about 70% arrive within 1,200 steps) |

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
   - then `./configure --enable-epoll --enable-packetver=20221005 --with-maxconn=8192` (renewal) and
     `make server`, as `slurm/build-train.sbatch` does in `~/ragnarok/rathena-re`. The bot client's
     packet table speaks that build, and the world jobs use it by default.
   - The earlier pre-renewal build is `~/ragnarok/rathena-train` (client 20130618, `PRERE=yes`).
     World jobs use it with `RATHENA=~/ragnarok/rathena-train`, after regenerating the client's
     tables for it (`client/README.md`).
2. Build the toolchain image once (`slurm/build_image.sbatch`): CUDA 12.2 and clang 15.
3. Train: `slurm/submit.sh slurm/train.sbatch NWORLDS NBOTS TIMESTEPS [RUNS_FILE]`. The job
   starts the worlds, builds the env and runs `./puffer train` (or each line of `RUNS_FILE`).

`slurm/env-check.sbatch` and `slurm/bench-worlds.sbatch` test a world and the env without a GPU.

## Results so far

Everything in this section ran on pre-renewal worlds (client 20130618), before the tasks moved to
renewal; the gates are being repeated on renewal.

| Setup | Bot steps/s | Note |
|---|---|---|
| Real time, 4 scripted bots | 40 | 226 kills in 10 minutes |
| Lockstep, 1 world, 4 scripted bots | 26,800 | 230 kills in 10 game minutes: same game, 670x faster |
| Lockstep, 1 world, 64 scripted bots | 63,800 | |
| Lockstep, 8 lean worlds of 64 bots, 20 CPUs | 470,000 to 550,000 | |
| Training: 16 worlds of 64 bots, one H100, 64 CPUs | about 1,000,000 | 500M steps in 9 minutes |

Training, field task alone (lr 0.015, reward_scale 1, death −1; perf at a number of world steps):

| Setup | 30M | 100M | 200M | 300M | final eval |
|---|---|---|---|---|---|
| Shared worlds, old masks, ent_coef 0.001 | 0.86 | 0.81 | 0.53 | 0.28 | 0.16 |
| Fresh worlds per run, monsters kept spread, new masks | 0.88 | 1.00 | 0.92 | 0.92 | 0.91 |
| Same, ent_coef 0.0001 (two seeds), the defaults now | 1.04–1.08 | 1.02–1.07 | 1.00–1.04 | 1.04–1.05 | 1.06–1.12 |

- **The early runs declined because the learned policy gathered the field's monsters into clumps.**
  - Over hours of game time the field's monster count stays the same, but most of the map empties
    (measured with `server/npc/rl_census.txt`); under the scripted bot it does not.
  - The likely cause: a monster that is hit chases the bot that hit it, and stays where it loses
    track when that bot is reset elsewhere, so a policy that starts fights and walks off drags
    monsters around.
  - A later run on the same worlds started on that empty field and never learned.
  - Every run now gets fresh worlds, and `server/npc/rl_spread.txt` keeps the task maps evenly
    stocked (see `slurm/README.md`).
- **G0 passes:** with ent_coef 0.0001 the field policy beats the scripted bot and holds it
  through 400M steps, with about 40 kills and 0.01 deaths per episode against the bot's 36 and 0.1.
- **G1 passes:** one model on all three tasks (900M steps, a third of the episodes each) does as
  well as each task alone (300M steps each): field 1.18 against 1.09, Swordman skills 0.79 against
  0.77, survival 0.89 against 0.93 (one training run each). The Swordman tasks are still below
  their scripted bots.
