"""GPU integration checks for training metrics and evaluator precision labels.

Pass a board trainer, both text evaluators, a small imported BERT bundle and a
new output directory. Uses two GPUs by default; launches no scheduler jobs.
"""

import argparse
import configparser
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_decision_learning
import compare_snake_learning


class PretrainedRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "board_trainer"):
            raise unittest.SkipTest("pass binary, bundle and output paths on the command line")

    def invoke(self, name, command, rejection=None):
        stdout = self.output / (name + ".stdout")
        stderr = self.output / (name + ".stderr")
        with stdout.open("x") as out, stderr.open("x") as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
                env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=180)
        if rejection is None:
            self.assertEqual(result.returncode, 0, stderr.read_text())
        else:
            self.assertNotEqual(result.returncode, 0, f"{name} unexpectedly succeeded")
            self.assertIn(rejection, stderr.read_text())
        return stdout

    def records(self, path):
        return [json.loads(line, parse_constant=lambda value: self.fail(value))
            for line in path.read_text().splitlines()]

    def training_command(self, name, metrics, epochs=1, max_steps=500):
        return [self.board_trainer, "train", "--headless", "--base.seed=73",
            "--base.async=0", "--base.cudagraphs=-1", "--base.gpu_offset=0",
            f"--train.gpus={self.gpus}", "--vec.total_agents=1", "--vec.num_buffers=1",
            "--vec.num_threads=1", "--vec.num_policies=1", "--selfplay.enabled=0",
            "--train.horizon=4", "--train.minibatch_size=4", "--train.replay_ratio=1",
            f"--train.total_timesteps={4 * self.gpus * epochs}",
            "--base.eval_episodes=0", "--base.save_training_state=0",
            "--base.resume_path=None", "--base.load_model_path=None",
            "--base.checkpoint_interval=0", f"--env.max_steps={max_steps}",
            "--policy.gemm_precision=fp32", f"--base.metrics_path={metrics}",
            f"--base.run_id={name}", f"--base.checkpoint_dir={self.output / 'checkpoints'}",
            f"--base.log_dir={self.output / 'logs'}"]

    def test_metrics_are_current_logging_events(self):
        for name, epochs, max_steps in (("no-episodes", 1, 500), ("timeouts", 2, 1)):
            with self.subTest(run=name):
                path = self.output / (name + ".jsonl")
                self.invoke(name, self.training_command(name, path, epochs, max_steps))
                run, *events = self.records(path)
                self.assertEqual(set(run), {"schema", "event", "world_size", "config"})
                self.assertEqual(run["schema"], "puffer.training_metrics.v1")
                self.assertEqual(run["event"], "run")
                self.assertEqual(run["world_size"], self.gpus)

                saved = configparser.ConfigParser(interpolation=None)
                saved.read(self.output / "logs/decision_snake" / (name + ".ini"))
                self.assertTrue(saved.has_section("metrics"))
                saved.remove_section("metrics")
                self.assertEqual(set(run["config"]), set(saved.sections()))
                for section in saved.sections():
                    self.assertEqual(set(run["config"][section]), set(saved[section]), section)
                for section, key, expected in (
                    ("base", "metrics_path", str(path)), ("base", "run_id", name),
                    ("train", "gpus", str(self.gpus)), ("policy", "gemm_precision", "fp32"),
                    ("env", "max_steps", str(max_steps)), ("selfplay", "enabled", "0"),
                ):
                    self.assertEqual(run["config"][section][key], expected)

                self.assertEqual(len(events), epochs)
                previous_uptime = -1
                for epoch, event in enumerate(events, 1):
                    self.assertEqual(set(event), {"schema", "event", "metrics"})
                    self.assertEqual(event["schema"], "puffer.training_metrics.v1")
                    self.assertEqual(event["event"], "metrics")
                    values = event["metrics"]
                    self.assertEqual(values["epoch"], epoch)
                    self.assertEqual(values["agent_steps"], 4 * self.gpus * epoch)
                    self.assertEqual(values["distributed/world_size"], self.gpus)
                    self.assertGreater(values["uptime"], previous_uptime)
                    previous_uptime = values["uptime"]
                    self.assertTrue(all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in values.values()))
                    for key in ("loss/policy", "loss/value", "loss/entropy", "loss/total",
                        "loss/old_kl", "loss/kl", "loss/clipfrac"):
                        self.assertIn(key, values)
                    if max_steps == 500:
                        self.assertEqual({key: value for key, value in values.items()
                            if key.startswith("env/")}, {"env/n": 0})
                    else:
                        self.assertEqual(values["env/n"], 4 * self.gpus)
                        self.assertEqual(values["env/episode_length"], 1)

                original = path.read_bytes()
                self.invoke(name + "-existing", self.training_command(name, path),
                    rejection="training metrics: cannot create")
                self.assertEqual(path.read_bytes(), original)

    def test_sweep_rejects_shared_metrics_path_before_launch(self):
        path = self.output / "sweep.jsonl"
        checkpoint_dir = self.output / "sweep-checkpoints"
        log_dir = self.output / "sweep-logs"
        self.invoke("sweep-metrics", [self.board_trainer, "sweep",
            f"--base.metrics_path={path}", f"--base.checkpoint_dir={checkpoint_dir}",
            f"--base.log_dir={log_dir}", "--sweep.max_runs=1", "--sweep.gpus=1"],
            rejection="sweep does not support base.metrics_path; "
                "use a separate path per training run")
        for created in (path, checkpoint_dir, log_dir):
            self.assertFalse(created.exists(), f"sweep launched a child that created {created}")

    def test_board_rejects_tf32(self):
        command = self.training_command("board-tf32", "None")
        command[command.index("--policy.gemm_precision=fp32")] = "--policy.gemm_precision=tf32"
        self.invoke("board-tf32", command,
            rejection="policy.gemm_precision is supported only by imported Transformer policies")

    def test_evaluator_labels_and_paired_comparisons(self):
        for environment, executable, comparator in (
            ("decision_cartpole", self.cartpole_evaluator, compare_decision_learning),
            ("decision_laya", self.snake_evaluator, compare_snake_learning),
        ):
            outputs = {}
            for precision in ("fp32", "tf32"):
                with self.subTest(environment=environment, precision=precision):
                    path = self.invoke(environment + "-" + precision, [executable,
                        f"--policy.bundle={self.bundle}", f"--policy.gemm_precision={precision}",
                        "--policy.sequence_length=384", "--policy.zero_init_critic=1",
                        "--base.seed=73", "--base.gpu_offset=0", "--env.max_steps=1",
                        "--episodes=16", "--batch=16", "--env-seed=6140001",
                        "--action-seed=6240001"])
                    emitted = self.records(path)
                    self.assertEqual(emitted[0]["gemm_precision"], precision)
                    protocol, summary, episodes = comparator.load(path)
                    for key, expected in (
                        ("environment", environment), ("bundle", self.bundle), ("episodes", 16),
                        ("batch", 16), ("max_steps", 1), ("padded_tokens", 384),
                        ("model_init_seed", 73), ("zero_init_critic", True),
                    ):
                        self.assertEqual(protocol[key], expected, key)
                    self.assertEqual(len(episodes), 16)
                    self.assertTrue(all(episode["length"] == 1 for episode in episodes.values()))
                    self.assertEqual(summary["pure_timeouts"], 16)
                    outputs[precision] = path
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(ValueError,
                    "evaluation protocols differ beyond checkpoint path"):
                    comparator.compare(outputs["fp32"], outputs["tf32"])
                if environment == "decision_laya":
                    result = comparator.compare(outputs["fp32"], outputs["tf32"], resamples=1000,
                        allow_protocol_differences=("gemm_precision",))
                    self.assertEqual(set(result["protocol_differences"]), {"gemm_precision"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-trainer", type=Path, required=True)
    parser.add_argument("--cartpole-evaluator", type=Path, required=True)
    parser.add_argument("--snake-evaluator", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=2)
    args, remaining = parser.parse_known_args()
    for name in ("board_trainer", "cartpole_evaluator", "snake_evaluator", "bundle"):
        setattr(PretrainedRuntimeTest, name, str(getattr(args, name).resolve()))
    PretrainedRuntimeTest.output = args.output_dir.resolve()
    PretrainedRuntimeTest.output.mkdir(parents=True, exist_ok=False)
    PretrainedRuntimeTest.gpus = args.gpus
    unittest.main(argv=[__file__, *remaining], verbosity=2)
