#!/usr/bin/env python3
"""GPU integration: fresh-process state/rollout replay and actual trainer CLI.

Run from the repository root in a CUDA allocation. The output must be new.
Final split/uninterrupted weights are compared diagnostically: Transformer
atomic gradient accumulation already makes separate training runs non-bitwise.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess


def corrupt_state(directory, rank, field, replacement, payload_offset=0):
    """Replace a named payload directly to test semantic, transactional validation."""
    path = directory / f"rank-{rank}.state"
    data = bytearray(path.read_bytes())

    def number(buffer, offset):
        assert offset + 8 <= len(buffer)
        return struct.unpack_from("<Q", buffer, offset)[0], offset + 8

    def string(buffer, offset):
        length, offset = number(buffer, offset)
        assert offset + length <= len(buffer)
        return bytes(buffer[offset:offset+length]).decode(), offset + length

    marker, offset = string(data, 0)
    assert marker == "PUFFER-TRAINING-RANK-2"
    size, offset = number(data, offset)  # raw checkpoint coordination value
    offset += size
    _, offset = string(data, offset)  # executable/config metadata
    actual_rank, offset = number(data, offset)
    assert actual_rank == rank
    _, offset = number(data, offset)  # epoch
    _, offset = number(data, offset)  # local step
    count, offset = number(data, offset)
    changed = False
    for _ in range(count):
        name, offset = string(data, offset)
        size, offset = number(data, offset)
        assert offset + size <= len(data)
        if name == field:
            assert not changed and payload_offset + len(replacement) <= size
            start = offset + payload_offset
            data[start:start+len(replacement)] = replacement
            changed = True
        offset += size
    assert changed and offset == len(data)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=2, choices=(1, 2))
    parser.add_argument("--optimizer", default="muon", choices=("muon", "adam"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe, trainer = map(lambda p: str(p.resolve()), (args.probe, args.trainer))
    outcomes = {"configuration": {"optimizer": args.optimizer, "gpus": args.gpus}}

    def probe_command(mode, path, world, shard, optimizer=None):
        return [probe, mode, str(path), str(world), str(shard),
                f"--optimizer={optimizer or args.optimizer}"]

    def run(label, command, success=True):
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=90)
        except subprocess.TimeoutExpired as error:
            text = error.stdout or ""
            if isinstance(text, bytes): text = text.decode(errors="replace")
            (output / f"{label}.log").write_text(text)
            outcomes[label] = {"timeout_seconds": 90, "expected_success": success,
                               "command": command}
            (output / "progress.json").write_text(json.dumps(outcomes, indent=2)+"\n")
            raise
        (output / f"{label}.log").write_text(result.stdout)
        outcomes[label] = {"returncode": result.returncode, "expected_success": success,
                           "command": command}
        (output / "progress.json").write_text(json.dumps(outcomes, indent=2)+"\n")
        # A crash or killed/hung child is not successful malformed-input
        # validation: the native launcher must report a controlled error.
        assert result.returncode == (0 if success else 1), (label, result.stdout[-4000:])
        return result.stdout

    cases = [(1, 0)]
    if args.gpus == 2:
        cases += [(2, 0)] if args.optimizer == "adam" else [(2, 0), (2, 1)]
    for world, shard in cases:
        label = f"world{world}-shard{shard}"
        state = output / label
        run(label + "-save", probe_command("save", state, world, shard))
        run(label + "-load", probe_command("load", state, world, shard))
        replay = []
        for rank in range(world):
            before = Path(str(state) + f".next-save-{rank}")
            after = Path(str(state) + f".next-load-{rank}")
            left, right = before.read_bytes(), after.read_bytes()
            assert left == right, (label, rank, "next rollout/fixed-update state differs")
            replay.append({"rank": rank, "save": str(before), "load": str(after), "bytes": len(left)})
        outcomes[label + "-fresh-process-replay"] = replay
        if (world, shard) == cases[-1]:
            for fault in ("rank-missing", "rank-truncated", "rank-trailing", "rank-v1", "rank-misplaced",
                          "manifest-missing", "manifest-truncated", "manifest-v1",
                          "world", "shard"):
                if fault == "rank-misplaced" and world == 1:
                    continue
                bad = output / ("bad-" + fault)
                shutil.copytree(state, bad)
                rank = bad / f"rank-{world - 1}.state"
                data = rank.read_bytes()
                if fault == "rank-missing": rank.unlink()
                elif fault == "rank-truncated": rank.write_bytes(data[:-1])
                elif fault == "rank-v1":
                    rank.write_bytes(data.replace(b"PUFFER-TRAINING-RANK-2", b"PUFFER-TRAINING-RANK-1", 1))
                elif fault == "rank-trailing": rank.write_bytes(data + b"x")
                elif fault == "rank-misplaced": shutil.copyfile(bad / "rank-0.state", rank)
                elif fault == "manifest-missing": (bad / "COMMITTED").unlink()
                elif fault.startswith("manifest-"):
                    manifest = bad / "COMMITTED"
                    contents = manifest.read_bytes()
                    if fault == "manifest-truncated":
                        contents = contents[:-1]
                    elif fault == "manifest-v1":
                        contents = contents.replace(b"PUFFER-TRAINING-COMMIT-2",
                            b"PUFFER-TRAINING-COMMIT-1", 1)
                    manifest.write_bytes(contents)
                wrong_world = 1 if fault == "world" and world == 2 else world
                wrong_shard = 1-shard if fault == "shard" else shard
                if fault == "world" and world == 1: continue
                text = run("reject-" + fault,
                    probe_command("load", str(bad) + "/", wrong_world, wrong_shard), False)
                if fault == "shard" and args.optimizer == "adam":
                    assert "Adam requires FP32 and train.distributed_optimizer=0" in text
                else:
                    assert "rejected load left rank" in text, "transactional rejection not verified"
                if fault.endswith("-v1"):
                    assert "feat/typed-supervision at 148add30" in text
            complete = output / "complete.partial"
            shutil.copytree(state, complete)
            run("load-complete-staging", probe_command("load", complete, world, shard))
            for rank in range(world):
                assert Path(str(complete) + f".next-load-{rank}").read_bytes() == \
                    Path(str(state) + f".next-save-{rank}").read_bytes()
            original = {p.name: p.read_bytes() for p in state.iterdir() if p.is_file()}
            run("reject-existing-committed", probe_command("save", state, world, shard), False)
            assert original == {p.name: p.read_bytes() for p in state.iterdir() if p.is_file()}, \
                "existing checkpoint changed"
            empty = output / "existing-empty"
            empty.mkdir()
            run("reject-existing-empty", probe_command("save", empty, world, shard), False)
            assert not list(empty.iterdir()), "existing empty destination changed"
            # A valid peer record from another same-config checkpoint must
            # never join this cohort, even when shape, rank and counters match.
            donor = output / "another-cohort"
            run("another-cohort-save", probe_command("save", donor, world, shard))
            marker_bytes = 8 + len("PUFFER-TRAINING-COMMIT-2")
            id_start = marker_bytes + 8  # world size precedes the raw 32-byte ID
            assert (donor / "COMMITTED").read_bytes()[id_start:id_start+32] != \
                (state / "COMMITTED").read_bytes()[id_start:id_start+32]
            mixed = output / "bad-mixed-rank"
            shutil.copytree(state, mixed)
            name = f"rank-{world - 1}.state"
            shutil.copyfile(donor / name, mixed / name)
            text = run("reject-mixed-rank", probe_command("load", mixed, world, shard), False)
            assert "incompatible metadata" in text, "did not reject the different checkpoint ID"
            assert "rejected load left rank" in text
            for fault, field, replacement, offset, message in (
                ("nonfinite-weights", "weights", struct.pack("<f", float("nan")), 0,
                    "nonfinite state in weights"),
                ("invalid-environment", "environment.1", struct.pack("<I", 0), 48+32,
                    "invalid environment.1"),
            ):
                bad = output / ("bad-" + fault)
                shutil.copytree(state, bad)
                corrupt_state(bad, world-1, field, replacement, offset)
                text = run("reject-" + fault, probe_command("load", bad, world, shard), False)
                assert message in text and "rejected load left rank" in text
            other = "muon" if args.optimizer == "adam" else "adam"
            text = run("reject-different-optimizer",
                probe_command("load", state, world, 0, other), False)
            assert "rejected load left rank" in text
            if args.optimizer == "adam":
                text = run("reject-sharded-adam-save", probe_command("save", output/"invalid-sharded-adam", world, 1), False)
                assert "Adam requires FP32 and train.distributed_optimizer=0" in text
                assert not (output/"invalid-sharded-adam").exists()
                for fault, field, replacement, message in (
                    ("adam-step-zero", "adam.step", struct.pack("<Q", 0), "Adam step does not match training epoch"),
                    ("adam-step-overflow", "adam.step", struct.pack("<Q", (1 << 64)-1), "Adam step does not match training epoch"),
                    ("adam-negative-second", "adam.second_moment", struct.pack("<f", -1), "negative Adam second moment"),
                    ("adam-nonfinite-first", "adam.first_moment", struct.pack("<f", float("nan")), "nonfinite state in adam.first_moment"),
                ):
                    bad = output / ("bad-"+fault)
                    shutil.copytree(state, bad)
                    corrupt_state(bad, world-1, field, replacement)
                    text = run("reject-"+fault, probe_command("load", bad, world, shard), False)
                    assert message in text, (fault, "did not reach semantic validation", text)
                    assert "rejected load left rank" in text

    world = args.gpus
    total, half = 256*world, 128*world
    common = [trainer, "train", "--headless", "--base.eval_episodes=0", "--base.async=0",
              "--base.cudagraphs=-1", "--base.seed=73", "--vec.total_agents=4",
              "--vec.num_buffers=2", "--vec.num_threads=1", "--env.max_steps=9",
              "--train.horizon=8", "--train.minibatch_size=16", "--train.learning_rate=0.001",
              "--train.anneal_lr=1", "--train.anneal_ent_coef=1", "--train.replay_ratio=1",
              f"--train.gpus={world}", f"--train.total_timesteps={total}",
              f"--train.optimizer={args.optimizer}",
              f"--train.distributed_optimizer={int(world > 1 and args.optimizer == 'muon')}", "--base.checkpoint_interval=4",
              f"--base.checkpoint_dir={output}/checkpoints", f"--base.log_dir={output}/logs"]
    run("cli-zero-epochs", common + ["--base.run_id=zero-epochs", "--train.total_timesteps=1"])
    run("cli-full", common + ["--base.run_id=full", "--base.save_training_state=1"])
    run("cli-split", common + ["--base.run_id=split", "--base.save_training_state=1",
                              f"--base.stop_after_steps={half}"])
    root = output / "checkpoints/decision_snake"
    resume = root / f"split/{half:016d}.train"
    run("cli-resume", common + ["--base.run_id=resumed", "--base.save_training_state=1",
                               f"--base.resume_path={resume}"])
    full = (root / f"full/{total:016d}.bin").read_bytes()
    split = (root / f"resumed/{total:016d}.bin").read_bytes()
    assert len(full) == len(split) and len(full) % 4 == 0
    a, b = struct.unpack(f"<{len(full)//4}f", full), struct.unpack(f"<{len(split)//4}f", split)
    import math
    assert all(math.isfinite(x) for x in a+b)
    outcomes["split_vs_uninterrupted"] = {
        "full_path": str(root / f"full/{total:016d}.bin"),
        "resumed_path": str(root / f"resumed/{total:016d}.bin"),
        "bitwise_equal": full == split, "max_abs_difference": max(abs(x-y) for x,y in zip(a,b)),
        "rms_difference": math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a)),
        "interpretation": "diagnostic; exact restoration, next-rollout and fixed-gradient update replay are the acceptance checks",
    }
    # Both prefixes ran independently without restoring a checkpoint. Record
    # their variation before interpreting any final split/full difference.
    prefix_full = (root / f"full/{half:016d}.bin").read_bytes()
    prefix_split = (root / f"split/{half:016d}.bin").read_bytes()
    assert len(prefix_full) == len(prefix_split) == len(full)
    pa = struct.unpack(f"<{len(prefix_full)//4}f", prefix_full)
    pb = struct.unpack(f"<{len(prefix_split)//4}f", prefix_split)
    assert all(math.isfinite(x) for x in pa+pb)
    outcomes["independent_training_prefixes"] = {
        "bitwise_equal": prefix_full == prefix_split,
        "max_abs_difference": max(abs(x-y) for x,y in zip(pa,pb)),
        "rms_difference": math.sqrt(sum((x-y)**2 for x,y in zip(pa,pb))/len(pa)),
        "interpretation": "independent prefixes before any resume; diagnostic of baseline run variation",
    }
    source = root / f"full/{total:016d}.bin"
    run("cli-warm-start", common + ["--base.run_id=warm", f"--base.load_model_path={source}",
                                   "--train.learning_rate=0", "--train.anneal_lr=0"])
    assert (root / f"warm/{total:016d}.bin").read_bytes() == full, "trainer ignored model-only warm start"
    for fault, content in (("short", full[:-4]), ("extra", full+b"xxxx"),
                           ("nonfinite", struct.pack("<f", float("nan"))+full[4:])):
        bad = output / (fault + ".bin"); bad.write_bytes(content)
        run("warm-reject-"+fault, common + [f"--base.run_id=bad-{fault}", f"--base.load_model_path={bad}"], False)
    for label, flags in (
        ("conflicting-load", [f"--base.resume_path={resume}", f"--base.load_model_path={source}"]),
        ("unaligned-stop", ["--base.save_training_state=1", "--base.stop_after_steps=1"]),
        ("stop-without-state", [f"--base.stop_after_steps={half}"]),
        ("different-budget", [f"--base.resume_path={resume}", f"--train.total_timesteps={2*total}"]),
        ("malformed-stop", ["--base.stop_after_steps=garbage"]),
    ):
        run("reject-"+label, common + [f"--base.run_id=bad-{label}"] + flags, False)
    assert not list(output.glob("checkpoints/**/*.partial")), "successful saves left staging directories"
    (output / "results.json").write_text(json.dumps(outcomes, indent=2)+"\n")
    print(json.dumps(outcomes["split_vs_uninterrupted"], indent=2))
    print(f"PASS: {args.optimizer} exact fresh-process replay, corrupt/incompatible state rejection, trainer resume/warm start")


if __name__ == "__main__":
    main()
