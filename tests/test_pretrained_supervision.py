"""Native typed trainer boundaries and GPU serving/checkpoint integration.

Uses a local tiny imported encoder bundle and writes all process outputs below a
new output directory. --cpu-only checks malformed input without a visible GPU.
The separate C-engine dataset campaign measures held-out learning quality.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def serving_probabilities(logits, temperature):
    # Match serving's FP32 softmax: unequal logits can round to tied probabilities.
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    scaled = [f32(f32(value) / f32(temperature)) for value in logits]
    maximum = max(scaled)
    values = [f32(math.exp(f32(value - maximum))) for value in scaled]
    total = 0.0
    for value in values:
        total = f32(total + value)
    return [f32(value / total) for value in values]


def examples(split):
    rows = []
    for index in range(3):
        state = {"split": split, "index": index,
            "board": "north wall; east body; south empty; west food. " * (index + 1),
            "values": [1.0, 1000000000000000001, False, None]}
        count = index + 2
        order = list(range(1, count)) + [0]
        for kind in ("choice", "score", "noul"):
            question = {"type": kind, "instructions": "Select the outcome of moving west."}
            if kind == "choice":
                question["criteria"] = {f"move{n}": f"direction {n}" for n in range(count)}
            elif kind == "score":
                question["criteria"] = [f"outcome {n}" for n in range(count)]
            else:
                question["instructions"] = "Moving west collides."
                question["labels"] = {"false": " no ", "true": " yes "}
            question["option_order"] = [1, 0] if kind == "noul" else order
            rows.append({"state": state, "questions": {"answer": question},
                "target": index % 2 if kind == "noul" else index % count})
    return rows


class PretrainedSupervisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "executable"):
            raise unittest.SkipTest("pass --executable, --bundle and --output-dir")
        # Real tokenizer/weights stay untouched. Nonunit calibration catches a
        # trainer that silently optimizes/evaluates raw or batch-max-K logits.
        cls.bundle = cls.output / "bundle"
        cls.bundle.mkdir()
        manifest = json.loads((cls.source_bundle / "manifest.json").read_text())
        decision_name = manifest["decision_config"]
        for source in cls.source_bundle.iterdir():
            if source.name != decision_name:
                (cls.bundle / source.name).symlink_to(source)
        config = json.loads((cls.source_bundle / decision_name).read_text())
        config["temperature"] = [0.7, 1.9, 1.4]
        config["temperature_by_options"] = {"choice:2": 2.7, "choice:3-5": 0.6,
            "score:2": 0.8, "score:3-5": 2.2, "noul:2": 1.6}
        (cls.bundle / decision_name).write_text(json.dumps(config) + "\n")
        cls.train_rows, cls.validation_rows = examples("train"), examples("validation")
        cls.train_path = cls.write_rows("train", cls.train_rows)
        cls.validation_path = cls.write_rows("validation", cls.validation_rows)

    @classmethod
    def write_rows(cls, name, rows):
        path = cls.output / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return path

    def invoke(self, name, arguments, rows=None, reject=False, cpu=False):
        command = [str(self.executable), "--bundle", str(self.bundle), *map(str, arguments)]
        environment = {**os.environ, "OMP_NUM_THREADS": "1"}
        if cpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
            input=None if rows is None else "".join(json.dumps(row) + "\n" for row in rows),
            env=environment, timeout=180)
        for extension, contents in (("stdout", result.stdout), ("stderr", result.stderr)):
            with (self.output / f"{name}.{extension}").open("x") as stream:
                stream.write(contents)
        if reject:
            self.assertNotEqual(result.returncode, 0, name)
            self.assertIn("pretrained_decision:", result.stderr)
            self.assertNotIn("CUDA", result.stderr, "input validation must precede CUDA allocation")
            self.assertNotIn("cuBLAS", result.stderr,
                "input validation must precede cuBLAS creation")
            self.assertFalse(result.stdout.strip())
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()] if not reject else []

    def train(self, name, epochs=0, train=None, validation=None, weights=None,
            reject=False, cpu=False, extra=()):
        checkpoint = self.output / (name + ".bin")
        arguments = ["--train", train or self.train_path,
            "--validation", validation or self.validation_path,
            "--epochs", epochs, "--batch-size", 4, "--learning-rate", 0.003,
            "--seed", 1, "--max-tokens", 256, "--save-weights", checkpoint, *extra]
        if weights:
            arguments += ["--weights", weights]
        records = self.invoke(name, arguments, reject=reject, cpu=cpu)
        if reject:
            self.assertFalse(checkpoint.exists(), "invalid input must not create a checkpoint")
        else:
            self.assertTrue(checkpoint.is_file())
        return checkpoint, records

    def test_invalid_rows_and_split_overlap_before_cuda(self):
        invalid = []
        for label in (-1, 2, 0.5, True, None, "0"):
            row = copy.deepcopy(self.train_rows[0])
            row["target"] = label
            invalid.append((f"target-{len(invalid)}", [row]))
        for index in (1, 2):
            row = copy.deepcopy(self.train_rows[index])
            row["target"] = 2
            invalid.append((f"typed-target-{index}", [row]))
        row = copy.deepcopy(self.train_rows[0])
        del row["target"]
        invalid.append(("missing-target", [row]))
        for count in (0, 2):
            row = copy.deepcopy(self.train_rows[0])
            row["questions"] = {f"question{n}": self.train_rows[0]["questions"]["answer"]
                for n in range(count)}
            invalid.append((f"questions-{count}", [row]))
        row = copy.deepcopy(self.train_rows[0])
        row["questions"]["answer"]["option_order"] = [0, 0]
        invalid.append(("invalid-permutation", [row]))
        row = copy.deepcopy(self.train_rows[0])
        row["state"] = "The state continues. " * 1000
        invalid.append(("truncated-state", [row]))
        invalid.append(("empty-training", []))
        for name, rows in invalid:
            with self.subTest(case=name):
                self.train(name, train=self.write_rows(name, rows), reject=True, cpu=True)
        self.train("split-overlap", validation=self.train_path, reject=True, cpu=True)
        self.train("empty-validation", validation=self.write_rows("empty-validation", []),
            reject=True, cpu=True)

    def test_invalid_training_options_before_cuda(self):
        for option, value in (("--epochs", "-1"), ("--batch-size", "0"),
                ("--learning-rate", "nan"), ("--learning-rate", "0"),
                ("--epochs", "1junk"), ("--batch-size", "2.5")):
            with self.subTest(option=option, value=value):
                self.train(f"option-{option[2:]}-{value}", extra=[option, value],
                    reject=True, cpu=True)

    def serving_metrics(self, name, rows, checkpoint=None):
        arguments = ["--raw", "--max-tokens", 256]
        if checkpoint:
            arguments += ["--weights", checkpoint]
        responses = self.invoke(name, arguments, rows=rows)
        self.assertEqual(len(responses), len(rows))
        totals = {}
        per_row = []
        for row, response in zip(rows, responses):
            question = row["questions"]["answer"]
            answer = response["answers"]["answer"]
            raw = answer["raw"]
            logits = [value / raw["temperature"] for value in raw["logits"]]
            maximum = max(logits)
            log_denominator = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
            order = question["option_order"]
            target_slot = order.index(row["target"])
            probabilities = serving_probabilities(raw["logits"], raw["temperature"])
            prediction = max(range(len(order)), key=lambda index: probabilities[order.index(index)])
            correct = int(prediction == row["target"])
            loss = log_denominator - logits[target_slot]
            per_row.append({"correct": correct, "log_loss": loss})
            total = totals.setdefault(question["type"], {"questions": 0,
                "accuracy": 0.0, "log_loss": 0.0})
            total["questions"] += 1
            total["accuracy"] += correct
            total["log_loss"] += loss
            self.assertEqual(len(raw["marker_positions"]), len(order))
            self.assertTrue(all(math.isfinite(value) for value in raw["logits"]))
        for total in totals.values():
            for key in ("accuracy", "log_loss"):
                total[key] /= total["questions"]
        return totals, responses, per_row

    def check_evaluation(self, record, expected, split, epoch):
        self.assertEqual(record["event"], "evaluation")
        self.assertEqual(record["split"], split)
        self.assertEqual(record["epoch"], epoch)
        self.assertEqual(record["states"], 3)
        self.assertEqual(record["questions"], 9)
        self.assertEqual(set(record["types"]), set(expected))
        for kind, values in expected.items():
            self.assertEqual(record["types"][kind]["questions"], values["questions"])
            self.assertAlmostEqual(record["types"][kind]["accuracy"], values["accuracy"],
                delta=1e-7)
            self.assertAlmostEqual(record["types"][kind]["log_loss"], values["log_loss"],
                delta=2e-5)

    def check_validation_rows(self, records, expected_by_epoch):
        records = [record for record in records if record["event"] == "validation_row"]
        self.assertEqual(set(record["epoch"] for record in records), set(expected_by_epoch))
        for epoch, expected in expected_by_epoch.items():
            actual = sorted((record for record in records if record["epoch"] == epoch),
                key=lambda record: record["row"])
            self.assertEqual([record["row"] for record in actual], list(range(len(expected))))
            for record, reference in zip(actual, expected):
                self.assertEqual(set(record), {"event", "epoch", "row", "correct", "log_loss"})
                self.assertIs(type(record["correct"]), int)
                self.assertEqual(record["correct"], reference["correct"])
                self.assertAlmostEqual(record["log_loss"], reference["log_loss"], delta=2e-5)

    def test_mixed_batches_match_serving_and_checkpoint_reload(self):
        if self.cpu_only:
            self.skipTest("GPU integration disabled by --cpu-only")
        initial, before = self.train("initial")
        reloaded, same = self.train("reloaded", weights=initial)
        self.assertEqual(initial.read_bytes(), reloaded.read_bytes())
        self.assertEqual(before, same)
        evaluations = [record for record in before if record["event"] == "evaluation"]
        self.assertEqual(len(evaluations), 2, "zero epochs must evaluate each split once")
        initial_outputs = {}
        for split, rows in (("train", self.train_rows), ("validation", self.validation_rows)):
            expected, outputs, per_row = self.serving_metrics(
                f"initial-serve-{split}", rows, initial)
            initial_outputs[split] = outputs
            record = next(item for item in evaluations if item["split"] == split)
            self.check_evaluation(record, expected, split, 0)
            if split == "validation":
                initial_rows = per_row
                self.check_validation_rows(before, {0: initial_rows})
        _, ordinary, _ = self.serving_metrics("ordinary-serving", self.validation_rows)
        self.assertEqual(ordinary, initial_outputs["validation"],
            "adding an unused critic must preserve ordinary serving initialization")

        # Serving breaks exact ties by semantic index, not packed position.
        zeros = self.output / "zero-weights.bin"
        zeros.write_bytes(bytes(initial.stat().st_size))
        tied_train = copy.deepcopy(self.train_rows)
        tied_validation = copy.deepcopy(self.validation_rows)
        for row in tied_train + tied_validation:
            row["target"] = 0
        _, tied = self.train("tied", weights=zeros,
            train=self.write_rows("tied-train", tied_train),
            validation=self.write_rows("tied-validation", tied_validation))
        expected, _, per_row = self.serving_metrics("tied-serving", tied_validation, zeros)
        for values in expected.values():
            self.assertEqual(values["accuracy"], 1.0)
        for record in (record for record in tied if record["event"] == "evaluation"):
            self.check_evaluation(record, expected, record["split"], 0)
        self.check_validation_rows(tied, {0: per_row})

        trained, records = self.train("trained", epochs=1, weights=initial)
        self.assertNotEqual(initial.read_bytes(), trained.read_bytes())
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        encoder = json.loads((self.bundle / manifest["encoder_config"]).read_text())
        decision = json.loads((self.bundle / manifest["decision_config"]).read_text())
        width = encoder["hidden_size"]
        n_act = decision.get("n_act", 1 + len(decision.get("act_costs", {})))
        # Existing flat layout ends with act-head tensors and the PPO critic.
        inactive = [256 * (width + 4), 256, n_act * 256, n_act, width, 1]
        inactive_bytes = 4 * sum((count + 3) // 4 * 4 for count in inactive)
        self.assertEqual(initial.read_bytes()[-inactive_bytes:],
            trained.read_bytes()[-inactive_bytes:],
            "supervision must leave act/critic weights untouched")
        updates = [record for record in records if record["event"] == "train"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["epoch"], 1)
        self.assertEqual(updates[0]["updates"], 3)
        self.assertEqual(updates[0]["questions"], 9)
        self.assertTrue(math.isfinite(updates[0]["mean_log_loss"]))
        evaluations = [record for record in records if record["event"] == "evaluation"]
        self.assertEqual(len(evaluations), 4)
        changed = False
        for split, rows in (("train", self.train_rows), ("validation", self.validation_rows)):
            expected, outputs, per_row = self.serving_metrics(
                f"trained-serve-{split}", rows, trained)
            record = next(item for item in evaluations
                if item["split"] == split and item["epoch"] == 1)
            self.check_evaluation(record, expected, split, 1)
            if split == "validation":
                self.check_validation_rows(records, {0: initial_rows, 1: per_row})
            for previous, current in zip(initial_outputs[split], outputs):
                previous = previous["answers"]["answer"]["raw"]
                current = current["answers"]["answer"]["raw"]
                for key in ("input_ids", "marker_positions", "temperature"):
                    self.assertEqual(previous[key], current[key])
                changed |= any(abs(a - b) > 1e-6
                    for a, b in zip(previous["logits"], current["logits"]))
        self.assertTrue(changed, "training must change predictions through the serving path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args, remaining = parser.parse_known_args()
    PretrainedSupervisionTest.executable = args.executable.resolve()
    PretrainedSupervisionTest.source_bundle = args.bundle.resolve()
    PretrainedSupervisionTest.output = args.output_dir.resolve()
    PretrainedSupervisionTest.output.mkdir(parents=True, exist_ok=False)
    PretrainedSupervisionTest.cpu_only = args.cpu_only
    unittest.main(argv=[__file__, *remaining], verbosity=2)
