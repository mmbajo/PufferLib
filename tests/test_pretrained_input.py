"""CPU parity with public Laya sequence construction and calibrated outputs.

Pass an already imported Laya --bundle; no checkpoint download or GPU is used.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    from laya.common import (QTYPES, answer_confidence, build_sequence, clamp_temperature,
                             confidence_from_probs, render_options, temp_bucket)
    from transformers import AutoTokenizer
except ImportError:
    np = None


@unittest.skipIf(np is None, "input oracle needs laya, NumPy and Transformers")
class PretrainedInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable, bundle = (os.environ.get(name) for name in
                              ("PUFFER_PRETRAINED_INPUT_TEST_BINARY", "PUFFER_LAYA_BUNDLE"))
        if not executable or not bundle:
            raise unittest.SkipTest("pass --executable and --bundle")
        cls.executable, cls.bundle = str(Path(executable).resolve()), Path(bundle).resolve()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.bundle, local_files_only=True)
        cls.config = json.loads((cls.bundle / "decision_config.json").read_text())

    def invoke(self, requests):
        with tempfile.TemporaryDirectory(prefix="puffer-input-oracle-") as directory:
            path = Path(directory) / "requests.json"
            path.write_text(json.dumps(requests, ensure_ascii=False))
            result = subprocess.run([self.executable, str(self.bundle), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return [json.loads(line) for line in result.stdout.splitlines()]

    def test_sequence_truncation_markers_and_calibration_match_laya(self):
        cases = []
        for kind in ("choice", "score", "noul"):
            question = {"t": kind, "ins": "Choose café Cafe\u0301 日本語 [MASK] honestly",
                        "crit": {"up": "Go north", "down": "Go south", "left": "Turn west"} if kind == "choice"
                        else ["bad", "okay", "good"] if kind == "score" else {}}
            for left in (False, True):
                cases.append((question, dict(state="日本語 [MASK] Cafe\u0301 " + "long state words. " * 180,
                                              max_len=64, head_max_len=32, truncate_left=left)))
        cases.append(({"t": "choice", "ins": "Pick the right option.",
                       "crit": {f"option{i}": f"distinct {i}" for i in range(12)}},
                      dict(state="42", max_len=512, head_max_len=192)))
        cases.append(({"t": "choice", "ins": "[MASK] instructions " + "long " * 100,
                       "crit": {"A": "first " + "description " * 80, "B": "second " + "text " * 100}},
                      dict(state="abc", max_len=64, head_max_len=32, order=[1, 0])))
        requests, expected = [], []
        for question, settings in cases:
            ids, markers, stats, truncated = build_sequence(
                self.tokenizer, settings["state"], question, max_len=settings["max_len"],
                head_max_len=settings["head_max_len"], truncate_left=settings.get("truncate_left", False),
                option_order=settings.get("order"), return_stats=True, return_truncation_stats=True)
            options = render_options(question)
            logits = np.linspace(-1.25, 1.0, len(options), dtype=np.float32)
            requests.append(dict(settings, type=question["t"], instructions=question["ins"],
                                 options=options, logits=logits.tolist()))
            qt = QTYPES[question["t"]]
            temperature = clamp_temperature(self.config.get("temperature_by_options", {}).get(
                temp_bucket(qt, len(options)), self.config["temperature"][qt]))
            probabilities = np.exp(logits / temperature - (logits / temperature).max())
            probabilities /= probabilities.sum()
            restored = np.empty_like(probabilities)
            restored[settings.get("order", list(range(len(options))))] = probabilities
            expected.append(dict(ids=ids, markers=markers, options_distinct=stats["options_distinct"],
                                 tokens_per_option=stats["tokens_per_option"] or -1,
                                 state_tokens=truncated["state_tokens"], state_tokens_used=truncated["state_tokens_used"],
                                 state_tokens_dropped=truncated["state_tokens_dropped"], temperature=temperature,
                                 probabilities=restored, prediction_index=int(restored.argmax()),
                                 answer_confidence=answer_confidence(restored, len(restored)),
                                 confidence=answer_confidence(restored, len(restored)) if qt == 2 else
                                 confidence_from_probs(restored, len(restored)),
                                 expected_score=float(np.arange(len(restored)) @ restored),
                                 act_probabilities=np.exp([0.5, -0.5]) / np.exp([0.5, -0.5]).sum()))
        actual = self.invoke(requests)
        self.assertEqual(len(actual), len(expected))
        for index, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(case=index):
                self.assertNotIn("error", got)
                for name, value in want.items():
                    if name in ("ids", "markers", "options_distinct", "tokens_per_option", "state_tokens",
                                "state_tokens_used", "state_tokens_dropped", "prediction_index"):
                        self.assertEqual(got[name], value, name)
                    else:
                        np.testing.assert_allclose(got[name], value, atol=2e-6, rtol=2e-6, err_msg=name)

    def test_invalid_or_ambiguous_requests_are_rejected(self):
        baseline = dict(state="short state", type="choice", instructions="Pick one.",
                        options=["first", "second"], logits=[0., 1.], max_len=64, head_max_len=32)
        variants = [dict(type="invalid"), dict(options=[]), dict(order=[0, 0]), dict(order=[1]),
                    dict(options=["identical", "identical"]), dict(max_len=8, head_max_len=32),
                    dict(type="noul", options=["only"]),
                    dict(state="long state " * 300, reject_truncated_state=True)]
        actual = self.invoke([dict(baseline, **changes) for changes in variants])
        self.assertEqual(len(actual), len(variants))
        for result in actual:
            self.assertIn("error", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--bundle")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_PRETRAINED_INPUT_TEST_BINARY"] = args.executable
    if args.bundle:
        os.environ["PUFFER_LAYA_BUNDLE"] = args.bundle
    unittest.main(argv=[__file__, *remaining], verbosity=2)
