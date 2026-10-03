#!/usr/bin/env python3
"""GPU integration: fresh-process state/rollout replay and actual trainer CLI.

Run from the repository root in a CUDA allocation. The output must be new.
Final split/uninterrupted weights are compared diagnostically: Transformer
atomic gradient accumulation already makes separate training runs non-bitwise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe, trainer = map(lambda p: str(p.resolve()), (args.probe, args.trainer))
    outcomes = {}

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

    for world, shard in [(1, 0)] + ([(2, 0), (2, 1)] if args.gpus == 2 else []):
        label = f"world{world}-shard{shard}"
        state = output / label
        before = run(label + "-save", [probe, "save", str(state), str(world), str(shard)])
        after = run(label + "-load", [probe, "load", str(state), str(world), str(shard)])
        pattern = r"rank=(\d+) next_rollout_sha=([a-f0-9]{64})"
        left, right = dict(re.findall(pattern, before)), dict(re.findall(pattern, after))
        assert len(left) == world and left == right, (label, left, right)
        outcomes[label + "-fresh-process-replay"] = left
        if world == args.gpus and shard == (1 if world == 2 else 0):
            for fault in ("rank-missing", "rank-truncated", "rank-corrupt", "rank-trailing",
                          "manifest-missing", "manifest-truncated", "partial", "world", "shard"):
                bad = output / ("bad-" + fault + (".partial" if fault == "partial" else ""))
                shutil.copytree(state, bad)
                rank = bad / f"rank-{world - 1}.state"
                data = rank.read_bytes()
                if fault == "rank-missing": rank.unlink()
                elif fault == "rank-truncated": rank.write_bytes(data[:-1])
                elif fault == "rank-corrupt":
                    changed = bytearray(data); changed[len(changed)//2] ^= 1; rank.write_bytes(changed)
                elif fault == "rank-trailing": rank.write_bytes(data + b"x")
                elif fault == "manifest-missing": (bad / "COMMITTED").unlink()
                elif fault == "manifest-truncated":
                    manifest = bad / "COMMITTED"; manifest.write_bytes(manifest.read_bytes()[:-1])
                wrong_world = 1 if fault == "world" and world == 2 else world
                wrong_shard = 1-shard if fault == "shard" else shard
                if fault == "world" and world == 1: continue
                # The trailing slash checks normalized rejection of staging dirs.
                run("reject-" + fault, [probe, "load", str(bad) + "/", str(wrong_world), str(wrong_shard)], False)
            original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in state.iterdir() if p.is_file()}
            run("reject-existing-committed", [probe, "save", str(state), str(world), str(shard)], False)
            assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in state.iterdir() if p.is_file()}, "existing checkpoint changed"
            empty = output / "existing-empty"
            empty.mkdir()
            run("reject-existing-empty", [probe, "save", str(empty), str(world), str(shard)], False)
            assert not list(empty.iterdir()), "existing empty destination changed"
            # A valid peer record from another same-config checkpoint must
            # never join this cohort, even when shape, rank and counters match.
            donor = output / "another-cohort"
            run("another-cohort-save", [probe, "save", str(donor), str(world), str(shard)])
            assert (donor / "ID").read_bytes() != (state / "ID").read_bytes()
            mixed = output / "bad-mixed-rank"
            shutil.copytree(state, mixed)
            name = f"rank-{world - 1}.state"
            shutil.copyfile(donor / name, mixed / name)
            text = run("reject-mixed-rank", [probe, "load", str(mixed), str(world), str(shard)], False)
            assert "incompatible metadata" in text, "did not reject the different checkpoint ID"

    world = args.gpus
    total, half = 256*world, 128*world
    common = [trainer, "train", "--headless", "--base.eval_episodes=0", "--base.async=0",
              "--base.cudagraphs=-1", "--base.seed=73", "--vec.total_agents=4",
              "--vec.num_buffers=2", "--vec.num_threads=1", "--env.max_steps=9",
              "--train.horizon=8", "--train.minibatch_size=16", "--train.learning_rate=0.001",
              "--train.anneal_lr=1", "--train.anneal_ent_coef=1", "--train.replay_ratio=1",
              f"--train.gpus={world}", f"--train.total_timesteps={total}",
              f"--train.distributed_optimizer={int(world > 1)}", "--base.checkpoint_interval=4",
              f"--base.checkpoint_dir={output}/checkpoints", f"--base.log_dir={output}/logs"]
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
        "full_sha256": hashlib.sha256(full).hexdigest(), "resumed_sha256": hashlib.sha256(split).hexdigest(),
        "bitwise_equal": full == split, "max_abs_difference": max(abs(x-y) for x,y in zip(a,b)),
        "rms_difference": math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a)),
        "interpretation": "diagnostic; exact restoration and next-rollout replay are the acceptance checks",
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
    print("PASS: exact fresh-process replay, corrupt/incompatible state rejection, trainer resume/warm start")


if __name__ == "__main__":
    main()
