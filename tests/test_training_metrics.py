"""CPU checks for the production native training JSONL writer; no GPU required."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        (ROOT / "build").mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="training-metrics-tests-", dir=ROOT / "build")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        cls.binary = cls.work / "metrics-test"
        command = compiler + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-Wno-unused-function", "-I", str(ROOT / "src"),
            str(ROOT / "tests/test_training_metrics.cpp"), "-o", str(cls.binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def invoke(self, mode, path, expected=0):
        result = subprocess.run([str(self.binary), mode, str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def read_events(self, name, mode="finite"):
        path = self.work / name
        self.invoke(mode, path)
        raw = path.read_text()
        self.assertTrue(raw.endswith("\n"))
        rows = [json.loads(line, parse_constant=lambda value: self.fail(value)) for line in raw.splitlines()]
        self.assertEqual(len(rows), 3)
        return rows

    def test_full_precision_zero_episode_event_and_provenance(self):
        run, zero, complete = self.read_events("finite.jsonl")
        self.assertEqual(run["schema"], "puffer.training_metrics.v1")
        self.assertEqual(run["event"], "run")
        self.assertEqual(run["config"]["base"]["seed"], "73")
        self.assertEqual(run["config"]["train"]["optimizer"], "muon")
        self.assertEqual(run["world_size"], 4)
        self.assertEqual(run["config"]["policy"]["gemm_precision"], "tf32")
        self.assertEqual(run["config"]["env"]["max_steps"], "500")
        self.assertEqual(run["config"]["selfplay"]["num_policies"], "2")
        self.assertEqual(zero["metrics"]["epoch"], 9)
        self.assertEqual(zero["metrics"]["agent_steps"], 9216)
        self.assertEqual(zero["metrics"]["uptime"], 12.345678901234567)
        self.assertEqual(zero["metrics"]["env/n"], 0)
        self.assertEqual(zero["metrics"]["loss/kl"], 1.2345678901234567e-9)
        self.assertEqual(zero["metrics"]["loss/entropy"], 1.0986122886681098)
        self.assertEqual(complete["metrics"]["env/n"], 3)
        self.assertEqual(complete["metrics"]["env/score"], 2.5)

    def test_json_escaping(self):
        run, metric, _ = self.read_events("escaped.jsonl")
        self.assertEqual(run["config"]["base"]["run_id"], 'line\nquote"slash\\tab\t\x01')
        self.assertEqual(run["config"]["policy"]["bundle"], "models/日本語")
        self.assertEqual(metric["metrics"]['loss/escaped"\\\n\x01'], -0.25)

    def test_nonfinite_values_are_null(self):
        _, row, _ = self.read_events("nonfinite.jsonl", "nonfinite")
        for key in ("loss/value", "loss/policy", "loss/total"):
            self.assertIsNone(row["metrics"][key])
    def test_existing_file_is_not_overwritten_or_appended(self):
        path = self.work / "existing.jsonl"
        original = b"previous evidence\n"
        path.write_bytes(original)
        result = self.invoke("create", path, expected=1)
        self.assertIn("training metrics: cannot create", result.stderr)
        self.assertEqual(path.read_bytes(), original)

    def test_disabled_writer_and_missing_parent(self):
        self.invoke("disabled", "None")
        self.invoke("disabled", "")
        path = self.work / "missing-parent" / "metrics.jsonl"
        self.invoke("create", path, expected=1)
        self.assertFalse(path.exists())



if __name__ == "__main__":
    unittest.main(verbosity=2)
