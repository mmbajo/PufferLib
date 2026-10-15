"""Full native Laya model oracle using the public laya.common.DecisionModel.

No models are downloaded. Small genuine HF encoders exercise every trainable
parameter of the preserved Laya architecture, including its detached act features.
"""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from laya.common import DecisionModel
    if __package__:
        from .test_pretrained_encoder import configuration, reference_model, write_model, write_tensor, read_u32, read_floats
    else:
        from test_pretrained_encoder import configuration, reference_model, write_model, write_tensor, read_u32, read_floats
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "full decision oracle needs laya, Transformers, Torch and NumPy")
class PretrainedDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PUFFER_PRETRAINED_DECISION_TEST_BINARY")
        if not executable:
            raise unittest.SkipTest("set PUFFER_PRETRAINED_DECISION_TEST_BINARY or pass --executable")
        cls.executable = str(Path(executable).resolve())
        if not Path(cls.executable).is_file():
            raise AssertionError(f"missing decision test harness: {cls.executable}")
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def compare(self, actual, expected, name, gradient=False):
        np.testing.assert_allclose(actual, expected.detach().numpy(),
                                   atol=2e-4 if gradient else 4e-5,
                                   rtol=1e-3 if gradient else 5e-4, err_msg=name)

    def run_case(self, config, head_layers=2, batch=3, options=4, critic=False, act_only=False):
        torch.manual_seed(38439)
        encoder, _ = reference_model(config)
        model = DecisionModel(encoder, head_layers=head_layers, n_act=3, dropout=0)
        model.train()
        if critic:
            model.value_head = torch.nn.Linear(config["width"], 1)
        tokens, n_act = 11, 3
        ids = torch.randint(3, config["vocab"], (batch, tokens))
        ids[:, 0], ids[:, -1] = 1, 2
        ids[:, 2:4] = 7  # Repeated embeddings must accumulate their gradients.
        if batch > 1:
            ids[0, 5] = 2
            ids[0, 6:] = config["pad_token_id"]
        mask = ids.ne(config["pad_token_id"]).long()
        types = torch.zeros_like(ids)  # Laya's public forward supplies no segment IDs.
        markers = torch.tensor([1, 3, 3, -1][:options]).expand(batch, -1).clone()
        marker_mask = torch.ones_like(markers, dtype=torch.bool)
        if options > 1:
            marker_mask[0, -1] = False
            if batch > 1:
                marker_mask[1, 1:] = False  # One valid answer among padded alternatives.
        qtype = torch.arange(batch) % 3
        d_logits = torch.randn(batch, options) / batch
        if act_only:
            d_logits.zero_()
        d_act = torch.randn(batch, n_act) / batch
        d_values = torch.randn(batch) / batch
        captured = []
        if critic:
            self.assertGreater(head_layers, 0)
            model.head.layers[-1].register_forward_hook(lambda _m, _a, result: captured.append(result))
        logits, act_logits = model(ids, mask, markers, marker_mask, qtype)
        values = model.value_head(captured[0][:, 0]).squeeze(-1) if critic else None
        loss = (logits * d_logits).sum() + (act_logits * d_act).sum()
        if critic:
            loss = loss + (values * d_values).sum()
        loss.backward()
        if act_only:
            for name, parameter in model.scorer.named_parameters():
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, "detached scorer " + name)

        with tempfile.TemporaryDirectory(prefix="puffer-laya-decision-") as directory:
            directory = Path(directory)
            weights, inputs, results = (directory / name for name in ("weights.bin", "inputs.bin", "results.bin"))
            write_model(weights, config, dict(model.named_parameters()))
            with inputs.open("wb") as stream:
                stream.write(b"PUFPDI1\0")
                stream.write(struct.pack("<6I", head_layers, n_act, critic, batch, tokens, options))
                for tensor in (ids, mask, types, markers, marker_mask, qtype):
                    stream.write(tensor.to(torch.int32).numpy().astype("<i4").tobytes())
                for tensor in (d_logits, d_act, d_values):
                    write_tensor(stream, tensor)
            result = subprocess.run([self.executable, str(weights), str(inputs), str(results)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with results.open("rb") as stream:
                self.assertEqual(stream.read(8), b"PUFPDO1\0")
                self.assertEqual(struct.unpack("<4I", stream.read(16)), (batch, options, n_act, critic))
                self.compare(read_floats(stream, batch * options).reshape(batch, options), logits, "option logits")
                self.compare(read_floats(stream, batch * n_act).reshape(batch, n_act), act_logits, "act logits")
                if critic:
                    self.compare(read_floats(stream, batch), values, "additional critic values")
                parameters = dict(model.named_parameters())
                self.assertEqual(read_u32(stream), len(parameters))
                seen = set()
                for _ in range(len(parameters)):
                    name = stream.read(read_u32(stream)).decode()
                    self.assertIn(name, parameters)
                    self.assertNotIn(name, seen)
                    seen.add(name)
                    parameter = parameters[name]
                    self.assertIsNotNone(parameter.grad, name)
                    count = struct.unpack("<Q", stream.read(8))[0]
                    self.assertEqual(count, parameter.numel(), name)
                    self.compare(read_floats(stream, count).reshape(parameter.shape), parameter.grad,
                                 name + " gradient", gradient=True)
                self.assertFalse(stream.read(1))

    def test_full_modernbert_laya_and_additional_critic(self):
        self.run_case(configuration("modernbert", layers=3, intermediate=24), critic=True)

    def test_full_bert_laya(self):
        self.run_case(configuration("bert", width=24, intermediate=36))

    def test_single_option_without_transformer_head(self):
        self.run_case(configuration("modernbert", layers=1), head_layers=0, batch=1, options=1)

    def test_act_features_are_detached_from_option_scorer(self):
        self.run_case(configuration("modernbert", layers=1), head_layers=1, act_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_PRETRAINED_DECISION_TEST_BINARY"] = args.executable
    unittest.main(argv=[__file__, *remaining], verbosity=2)
