"""Capture golden traces from an independently checked-out decisions Snake engine.

This is a manual fixture generator, never part of test execution. For example:
    python3 tests/fixtures/decision_snake_reference.py /path/to/decisions/envs/snake/snake.c

Only the action policy is implemented here; all expected state and reward values
come from the supplied C engine. Review the provenance and diff before replacing
the checked-in fixture. Normal tests need neither ctypes nor the original checkout.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def cycle_actions():
    path = [row * 10 for row in range(10)]
    for row in range(9, -1, -1):
        path.extend(row * 10 + (col if row % 2 else 10 - col) for col in range(1, 10))
    successors = dict(zip(path, path[1:] + path[:1]))
    position = 55
    while True:
        following = successors[position]
        yield {-10: 0, 10: 1, -1: 2, 1: 3}[following - position]
        position = following


def capture(source, library):
    ptr = ctypes.c_void_p
    library.ib_snake_create.argtypes = [ctypes.c_uint32, ctypes.c_int]
    library.ib_snake_create.restype = ptr
    library.ib_snake_destroy.argtypes = [ptr]
    library.ib_snake_step.argtypes = [ptr, ctypes.c_int]
    library.ib_snake_observe.argtypes = [ptr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    names = ["steps", "terminated", "truncated", "outcome", "score", "episode_return", "reward", "legal_actions"]
    for name in names:
        function = getattr(library, "ib_snake_" + name)
        function.argtypes = [ptr]
        function.restype = ctypes.c_double if name in {"score", "episode_return", "reward"} else ctypes.c_int

    def snapshot(env):
        board = (ctypes.c_int32 * 100)()
        assert library.ib_snake_observe(env, board, 100) == 0
        fields = [getattr(library, "ib_snake_" + name)(env) for name in names]
        # This reward scheme uses exact integers; avoid a float serialization oracle.
        assert all(value == int(value) for value in fields)
        return [int(value) for value in fields] + list(board)

    def episode(name, seed, cap, actions):
        env = library.ib_snake_create(seed, cap)
        assert env
        try:
            first = snapshot(env)
            rows = [first]
            taken = []
            for action in actions:
                assert library.ib_snake_step(env, action) == 0
                taken.append(str(action))
                rows.append(snapshot(env))
                if rows[-1][1] or rows[-1][2]:
                    break
            assert rows[-1][1] or rows[-1][2], name
            serialized = "".join(",".join(map(str, row)) + "\n" for row in rows)
            return {
                "name": name,
                "seed": seed,
                "max_steps": cap,
                "actions": "".join(taken),
                "trace_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "rows": len(rows),
                "first": first,
                "final": rows[-1],
            }
        finally:
            library.ib_snake_destroy(env)

    cases = [
        episode("wall_up_at_cap", 42, 6, [0] * 6),
        episode("wall_down", 0, 100, [1] * 5),
        episode("wall_right", 1, 100, [3] * 5),
        episode("wall_left", 0xFFFFFFFF, 100, [0] + [2] * 6),
    ]
    for seed in (0, 1, 42, 0xFFFFFFFF):
        cases.append(episode(f"timeout_{seed}", seed, 32, cycle_actions()))
        cases.append(episode(f"complete_{seed}", seed, 10000, cycle_actions()))
    complete = next(case for case in cases if case["name"] == "complete_0")
    cases.append(episode("full_board_at_cap", 0, complete["final"][0], cycle_actions()))
    # Find a seed whose initial food is directly above the head, so a one-step
    # timeout must retain both the growth and its +1 reward before resetting.
    for seed in range(1000):
        env = library.ib_snake_create(seed, 1)
        state = snapshot(env)
        library.ib_snake_destroy(env)
        if state[8 + 45] == -1:
            cases.append(episode("food_at_timeout", seed, 1, [0]))
            break
    else:
        raise AssertionError("No fixture seed found")
    cases.append(episode("empty_at_timeout", 0xFFFFFFFF, 1, [0]))
    resets = {}
    for seed in sorted({case["seed"] for case in cases} | {(case["seed"] + 1) % 2**32 for case in cases}):
        env = library.ib_snake_create(seed, 10000)
        resets[str(seed)] = snapshot(env)
        library.ib_snake_destroy(env)

    return {
        "description": "Reference C engine reset and accepted-step traces; CSV integers with trailing newline, fields followed by 100 raw row-major board cells.",
        "fields": names + ["board[100]"],
        "source": {
            "repository": "https://github.com/sbintuitions/decisions",
            "revision": subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip(),
            "path": "envs/snake/snake.c",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "header_sha256": hashlib.sha256(source.with_suffix(".h").read_bytes()).hexdigest(),
        },
        "resets": resets,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    source = args.source.resolve()
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "reference.so"
        subprocess.run(["cc", "-std=c11", "-shared", "-fPIC", "-O2", str(source), "-o", str(target)], check=True)
        fixture = capture(source, ctypes.CDLL(str(target)))
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Captured {len(fixture['cases'])} reference traces into {args.output}")


if __name__ == "__main__":
    main()
