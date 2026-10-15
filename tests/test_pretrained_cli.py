"""Black-box JSONL inference against public Laya, including trained Puffer weights.

Uses an existing local snapshot and bundle; never downloads a checkpoint. Run
inside a GPU allocation with enough host memory for the full CPU reference.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import unittest

try:
    import numpy as np
    import torch
    from laya.common import (QTYPES, build_model, build_sequence, clamp_temperature,
                             confidence_from_probs, render_criterion, temp_bucket)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
except ImportError:
    torch = None


def simple_request():
    return {"state": "A snake faces east. Food is north and a wall is to the east.", "questions": {
        "move": {"type": "choice", "instructions": "Choose a safe move toward the food.",
                 "criteria": {"left": "turn north", "forward": "continue east", "right": "turn south"},
                 "option_order": [2, 0, 1]},
        "safe": {"type": "noul", "instructions": "Continuing east is safe."},
    }}


@unittest.skipIf(torch is None, "CLI oracle needs public laya, Transformers, Torch, NumPy and safetensors")
class PretrainedCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable, bundle, snapshot = (os.environ.get(name) for name in (
            "PUFFER_PRETRAINED_CLI_BINARY", "PUFFER_LAYA_BUNDLE", "PUFFER_LAYA_SNAPSHOT"))
        if not executable or not bundle or not snapshot:
            raise unittest.SkipTest("pass --executable, --bundle and --laya-snapshot")
        cls.executable = str(Path(executable).resolve())
        cls.bundle, cls.snapshot = Path(bundle).resolve(), Path(snapshot).resolve()
        cls.weights = os.environ.get("PUFFER_PRETRAINED_CLI_WEIGHTS")
        cls.config = json.loads((cls.bundle / "decision_config.json").read_text())
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.snapshot / "tokenizer", local_files_only=True)
        torch.set_num_threads(4)
        cls.model = build_model(cls.config, encoder_dir=str(cls.snapshot / "encoder"), pretrained=False)
        cls.model.load_state_dict(load_file(str(cls.snapshot / "model.safetensors")), strict=True)
        cls.model.encoder.set_attn_implementation("eager")
        cls.model.eval()

    def invoke(self, requests, weights=None):
        command = [self.executable, "--bundle", str(self.bundle), "--raw", "--max-tokens", "256"]
        if weights:
            command.extend(["--weights", weights])
        return subprocess.run(command, input="".join((request if isinstance(request, str)
                                                       else json.dumps(request, ensure_ascii=False)) + "\n"
                                                     for request in requests),
                              capture_output=True, text=True)

    def reference(self, request, question):
        q = {"t": question.get("type", question.get("t")),
             "ins": question.get("instructions", question.get("ins")),
             "crit": question.get("criteria", question.get("crit"))}
        if "labels" in question:
            q["labels"] = question["labels"]
        ids, markers, _stats, truncated = build_sequence(
            self.tokenizer, request["state"], q, max_len=256, head_max_len=self.config["head_max_len"],
            option_order=question.get("option_order"), return_stats=True, return_truncation_stats=True)
        tensors = [torch.tensor([values], dtype=torch.long) for values in (ids, markers)]
        qt = QTYPES[q["t"]]
        with torch.no_grad():
            logits, acts = self.model(tensors[0], torch.ones_like(tensors[0]), tensors[1],
                                     torch.ones_like(tensors[1], dtype=torch.bool), torch.tensor([qt]))
        temperature = clamp_temperature(self.config.get("temperature_by_options", {}).get(
            temp_bucket(qt, len(markers)), self.config["temperature"][qt]))
        probabilities = torch.softmax(logits[0] / temperature, -1).numpy()
        restored = np.empty_like(probabilities)
        restored[question.get("option_order", list(range(len(markers))))] = probabilities
        return q, ids, markers, truncated, logits[0].numpy(), acts[0].numpy(), restored, temperature

    def test_original_checkpoint_matches_public_laya_for_typed_json(self):
        requests = [simple_request(), {
            "state": {"numbers": [1.0, 1e6, 1000000000000000001, -0.0],
                      "flags": [True, False, None], "text": "café 東京 [MASK] with \"quotes\" and\\slashes"},
            "questions": {"numeric": {"type": "choice", "instructions": "Select the most faithful description.",
                                      "criteria": {"first": {"value": 1.0}, "second": {"value": 1e6},
                                                   "third": {"value": 1000000000000000001}},
                                      "option_order": [1, 2, 0]}},
        }, {
            "state": "The answer says four. " * 200,
            "questions": {"quality": {"type": "score", "instructions": "Grade the reply.",
                                      "criteria": [{"value": 1.0, "good": False},
                                                   {"value": 1e6, "good": True}, {"value": 1000000000000000001}]},
                          "supported": {"t": "noul", "ins": "The answer includes a number.",
                                        "labels": {"false": " no ", "true": " yes "},
                                        "crit": {"false": {"evidence": False, "weight": 1.0},
                                                 "true": {"evidence": True, "weight": 1e6}}}},
        }]
        # These wire spellings differ from Python's canonical json.dumps output.
        scientific = '''{"state":{"numbers":[1e6,1E+06,1.00,-0E+00,1e-05,1e-04,1e15,1e16,
            1.2345678901234567,0.00000000000000001,-1000000000000000001]},"questions":{
            "scientific":{"type":"choice","instructions":"Choose the numeric description.",
            "criteria":{"A":{"weight":1.00},"B":{"weight":1e6}}}}}'''
        scientific = scientific.replace("\n", " ")
        requests.append(json.loads(scientific))
        result = self.invoke([*requests[:-1], scientific])
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(responses), len(requests))
        for index, (request, response) in enumerate(zip(requests, responses)):
            expected_usage = 0
            self.assertEqual(set(response["answers"]), set(request["questions"]))
            for name, question in request["questions"].items():
                with self.subTest(request=index, question=name):
                    q, ids, markers, truncation, logits, acts, probabilities, temperature = self.reference(request, question)
                    answer = response["answers"][name]
                    self.assertEqual(answer["raw"]["input_ids"], ids, "tokenized typed JSON differs from public Laya")
                    self.assertEqual(answer["raw"]["marker_positions"], markers)
                    np.testing.assert_allclose(answer["raw"]["logits"], logits, atol=1e-3, rtol=1e-3)
                    np.testing.assert_allclose(answer["raw"]["act_logits"], acts, atol=1e-3, rtol=1e-3)
                    self.assertAlmostEqual(answer["raw"]["temperature"], temperature, places=6)
                    expected_usage += len(ids)
                    best = int(probabilities.argmax())
                    self.assertAlmostEqual(answer["answer_confidence"], probabilities[best], delta=2e-4)
                    expected_act = torch.softmax(torch.from_numpy(acts), -1)[0].item()
                    self.assertAlmostEqual(answer["action"]["act_probability"], expected_act, delta=2e-4)
                    if q["t"] == "noul":
                        self.assertAlmostEqual(answer["noul"], probabilities[1], delta=2e-4)
                        confidence = probabilities[best]
                    else:
                        labels = list(q["crit"]) if q["t"] == "choice" else [str(i) for i in range(len(q["crit"]))]
                        np.testing.assert_allclose([answer["probabilities"][label] for label in labels],
                                                   probabilities, atol=2e-4, rtol=0)
                        confidence = confidence_from_probs(probabilities, len(probabilities))
                        if q["t"] == "choice":
                            self.assertEqual(answer["choice"], labels[best])
                        else:
                            self.assertEqual(answer["legend"], {str(i): render_criterion(value)
                                                                for i, value in enumerate(q["crit"])})
                            self.assertAlmostEqual(answer["score"], float(np.arange(len(probabilities)) @ probabilities),
                                                   delta=3e-4)
                    self.assertAlmostEqual(answer["confidence"], confidence, delta=2e-4)
                    if truncation["truncated"]:
                        self.assertEqual(answer["truncation"]["state_tokens_dropped"], truncation["state_tokens_dropped"])
            self.assertEqual(response["usage"]["input_tokens"], expected_usage)

    def test_trained_puffer_weights_produce_changed_finite_predictions(self):
        if not self.weights:
            self.skipTest("pass --weights with a trained native Puffer checkpoint")
        before, after = (self.invoke([simple_request()], weights) for weights in (None, self.weights))
        for result in (before, after):
            self.assertEqual(result.returncode, 0, result.stderr)
        before, after = (json.loads(result.stdout)["answers"] for result in (before, after))
        differences = []
        for name in before:
            self.assertEqual(before[name]["raw"]["input_ids"], after[name]["raw"]["input_ids"])
            for field in ("logits", "act_logits"):
                initial = np.asarray(before[name]["raw"][field])
                trained = np.asarray(after[name]["raw"][field])
                self.assertTrue(np.isfinite(trained).all(), field)
                differences.append(np.max(np.abs(initial - trained)))
        self.assertGreater(max(differences), 1e-6, "trained flat checkpoint did not change any prediction")

    def test_invalid_question_types_fail_with_an_error(self):
        for invalid in ("not-a-question-type", 42, None):
            with self.subTest(type=invalid):
                request = {"state": "text", "questions": {"bad": {"type": invalid,
                           "instructions": "Choose.", "criteria": {"A": "first", "B": "second"}}}}
                result = self.invoke([request])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pretrained_decision:", result.stderr)
                self.assertFalse(result.stdout.strip())

    def test_invalid_custom_labels_fail_with_an_error(self):
        variants = [
            {"type": "noul", "labels": {"false": " ", "true": "yes"}},
            {"type": "noul", "labels": {"false": "same", "true": " same "}},
            {"type": "noul", "labels": {"false": "no", "true": "yes", "extra": "invalid"}},
            {"type": "choice", "criteria": {"A": "first", "B": "second"},
             "labels": {"false": "no", "true": "yes"}},
        ]
        for invalid in variants:
            with self.subTest(question=invalid):
                request = {"state": "text", "questions": {"bad": dict(invalid, instructions="Choose.")}}
                result = self.invoke([request])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pretrained_decision:", result.stderr)
                self.assertFalse(result.stdout.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--bundle")
    parser.add_argument("--laya-snapshot")
    parser.add_argument("--weights")
    args, remaining = parser.parse_known_args()
    for name, value in (("PUFFER_PRETRAINED_CLI_BINARY", args.executable), ("PUFFER_LAYA_BUNDLE", args.bundle),
                        ("PUFFER_LAYA_SNAPSHOT", args.laya_snapshot), ("PUFFER_PRETRAINED_CLI_WEIGHTS", args.weights)):
        if value:
            os.environ[name] = value
    unittest.main(argv=[__file__, *remaining], verbosity=2)
