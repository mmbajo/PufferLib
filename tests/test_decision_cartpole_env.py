#!/usr/bin/env python3
"""Compare the decision adapter against the stock environment without a GPU.

Usage: python tests/test_decision_cartpole_env.py STOCK_BINARY DECISION_BINARY BUNDLE
The decision binary additionally asserts its tokenizer, timeout and rejection
contracts before emitting the same seeded traces as the stock binary.
"""
import subprocess
import sys


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    def run(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stderr}")
        return result.stdout

    stock = run([sys.argv[1]])
    decision = run(sys.argv[2:])
    stock_lines, decision_lines = stock.splitlines(), decision.splitlines()
    if stock_lines != decision_lines:
        for index, (before, after) in enumerate(zip(stock_lines, decision_lines)):
            if before != after:
                raise AssertionError(f"CartPole trace {index} differs:\nstock: {before}\ndecision: {after}")
        raise AssertionError(f"Trace lengths differ: {len(stock_lines)} != {len(decision_lines)}")
    assert len(stock_lines) == 2249
    print("PASS: 2,249 identical stock/decision physics, reward and reset snapshots; "
          "timeout, termination, token packing, locale and invalid-input checks passed")


if __name__ == "__main__":
    main()
