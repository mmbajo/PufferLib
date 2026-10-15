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


def typed_losses(logits, mask, targets, temperatures, qtypes, order):
    losses = []
    false_slot, true_slot = order.index(0), order.index(1)
    for row, qtype in enumerate(qtypes):
        scaled = logits[row] / temperatures[row]
        if qtype == 2:
            # Use actual binary BCE as an independent oracle for the shared CE.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled[true_slot] - scaled[false_slot],
                (targets[row] == true_slot).float())
        else:
            loss = torch.nn.functional.cross_entropy(
                scaled.masked_fill(~mask[row], -torch.inf)[None], targets[row:row + 1])
        losses.append(loss)
    return torch.stack(losses)


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
        cls.gemm_precision = os.environ.get("PUFFER_PRETRAINED_GEMM_PRECISION", "fp32")
        if cls.gemm_precision not in ("fp32", "tf32"):
            raise ValueError("PUFFER_PRETRAINED_GEMM_PRECISION must be fp32 or tf32")

    def compare(self, actual, expected, name, gradient=False):
        if self.gemm_precision == "tf32":
            expected = expected.detach().numpy()
            self.assertEqual(actual.shape, expected.shape, name)
            self.assertTrue(np.isfinite(actual).all(), name)
            # Exclude the fixed masking sentinel from the norm: its magnitude
            # must not hide drift in the meaningful option logits.
            if name == "option logits":
                masked = expected == -1e4
                np.testing.assert_array_equal(actual[masked], expected[masked])
                actual, expected = actual[~masked], expected[~masked]
            error = np.linalg.norm((actual - expected).astype(np.float64))
            scale = np.linalg.norm(expected.astype(np.float64))
            absolute = (2e-4 if gradient else 4e-5) * np.sqrt(expected.size)
            self.assertLessEqual(error, absolute + (2e-2 if gradient else 5e-3) * scale, name)
            return
        np.testing.assert_allclose(actual, expected.detach().numpy(),
                                   atol=2e-4 if gradient else 4e-5,
                                   rtol=1e-3 if gradient else 5e-4, err_msg=name)

    def run_case(self, config, head_layers=2, batch=3, options=4, critic=False, act_only=False,
                 tokens=11, require_tf32_difference=False, supervised=False, permuted=False):
        torch.manual_seed(38439)
        encoder, _ = reference_model(config)
        model = DecisionModel(encoder, head_layers=head_layers, n_act=3, dropout=0)
        model.train()
        if critic:
            model.value_head = torch.nn.Linear(config["width"], 1)
        n_act = 3
        ids = torch.randint(3, config["vocab"], (batch, tokens))
        ids[:, 0], ids[:, -1] = 1, 2
        ids[:, 2:4] = 7  # Repeated embeddings must accumulate their gradients.
        if batch > 1:
            padded_row = batch - 1 if supervised else 0
            ids[padded_row, 5] = 2
            ids[padded_row, 6:] = config["pad_token_id"]
        mask = ids.ne(config["pad_token_id"]).long()
        types = torch.zeros_like(ids)  # Laya's public forward supplies no segment IDs.
        if supervised:
            markers = torch.tensor([1, 3, 4, 8, 10]).expand(batch, -1).clone()
            counts = torch.tensor([5, 3, 2, 1, 4, 2])
            marker_mask = torch.arange(options)[None] < counts[:, None]
            markers[~marker_mask] = -1
            targets = torch.tensor([4, 1, 1, 0, 3, 0])
            temperatures = torch.tensor([.5, 1, 1.8, 1.8, .5, 1])
            order = [3, 1, 4, 0, 2] if permuted else list(range(options))
            markers, marker_mask = markers[:, order], marker_mask[:, order]
            targets = torch.tensor([order.index(target) for target in targets.tolist()])
        else:
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
        if supervised:
            logits.retain_grad()
            losses = typed_losses(logits, marker_mask, targets, temperatures, qtype, order)
            loss = losses.mean()
        else:
            loss = (logits * d_logits).sum() + (act_logits * d_act).sum()
            if critic:
                loss = loss + (values * d_values).sum()
        loss.backward()
        if supervised:
            for branch in ("encoder.", "head.", "scorer.", "type_emb."):
                self.assertTrue(any(parameter.grad is not None and parameter.grad.count_nonzero()
                    for name, parameter in model.named_parameters() if name.startswith(branch)), branch)
        if act_only:
            for name, parameter in model.scorer.named_parameters():
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, "detached scorer " + name)

        with tempfile.TemporaryDirectory(prefix="puffer-laya-decision-") as directory:
            directory = Path(directory)
            weights, inputs, results = (directory / name for name in ("weights.bin", "inputs.bin", "results.bin"))
            write_model(weights, config, dict(model.named_parameters()))
            with inputs.open("wb") as stream:
                stream.write(b"PUFPDI2\0" if supervised else b"PUFPDI1\0")
                stream.write(struct.pack("<6I", head_layers, n_act, critic, batch, tokens, options))
                for tensor in (ids, mask, types, markers, marker_mask, qtype):
                    stream.write(tensor.to(torch.int32).numpy().astype("<i4").tobytes())
                if supervised:
                    stream.write(targets.numpy().astype("<i4").tobytes())
                    write_tensor(stream, temperatures)
                else:
                    for tensor in (d_logits, d_act, d_values):
                        write_tensor(stream, tensor)
            result = subprocess.run([self.executable, str(weights), str(inputs), str(results), self.gemm_precision],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            fp32_logits = None
            if self.gemm_precision == "tf32" and require_tf32_difference:
                fp32_results = directory / "fp32.bin"
                result = subprocess.run(
                    [self.executable, str(weights), str(inputs), str(fp32_results), "fp32"],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with fp32_results.open("rb") as stream:
                    stream.seek(24)  # Magic and four dimensions precede option logits.
                    fp32_logits = read_floats(stream, batch * options)
            with results.open("rb") as stream:
                self.assertEqual(stream.read(8), b"PUFPDO1\0")
                self.assertEqual(struct.unpack("<4I", stream.read(16)), (batch, options, n_act, critic))
                native_logits = read_floats(stream, batch * options)
                self.compare(native_logits.reshape(batch, options), logits, "option logits")
                if fp32_logits is not None:
                    self.assertFalse(np.array_equal(native_logits, fp32_logits),
                        "aligned TF32 decision logits equal FP32; GEMM precision may be ignored")
                self.compare(read_floats(stream, batch * n_act).reshape(batch, n_act), act_logits, "act logits")
                if critic:
                    self.compare(read_floats(stream, batch), values, "additional critic values")
                if supervised:
                    self.compare(read_floats(stream, batch), losses, "typed row losses")
                    gradient = read_floats(stream, batch * options).reshape(batch, options)
                    self.compare(gradient, logits.grad, "typed logit gradient", gradient=True)
                    np.testing.assert_array_equal(gradient[~marker_mask.numpy()], 0)
                parameters = dict(model.named_parameters())
                self.assertEqual(read_u32(stream), len(parameters))
                seen = set()
                for _ in range(len(parameters)):
                    name = stream.read(read_u32(stream)).decode()
                    self.assertIn(name, parameters)
                    self.assertNotIn(name, seen)
                    seen.add(name)
                    parameter = parameters[name]
                    if not supervised:
                        self.assertIsNotNone(parameter.grad, name)
                    count = struct.unpack("<Q", stream.read(8))[0]
                    self.assertEqual(count, parameter.numel(), name)
                    native_gradient = read_floats(stream, count).reshape(parameter.shape)
                    expected = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                    self.compare(native_gradient, expected, name + " gradient", gradient=True)
                    if supervised and name.startswith(("act_head.", "value_head.")):
                        np.testing.assert_array_equal(native_gradient, 0, err_msg=name)
                self.assertFalse(stream.read(1))

    def test_full_modernbert_laya_and_additional_critic(self):
        self.run_case(configuration("modernbert", layers=3, intermediate=24), critic=True)

    def test_full_bert_laya(self):
        self.run_case(configuration("bert", width=24, intermediate=36))

    def test_single_option_without_transformer_head(self):
        self.run_case(configuration("modernbert", layers=1), head_layers=0, batch=1, options=1)

    def test_act_features_are_detached_from_option_scorer(self):
        self.run_case(configuration("modernbert", layers=1), head_layers=1, act_only=True)

    def test_larger_full_decision_projection_shapes(self):
        for family in ("bert", "modernbert"):
            with self.subTest(family=family):
                self.run_case(configuration(family, width=128, heads=4,
                                            intermediate=256, layers=2), critic=True, batch=2,
                              tokens=32, require_tf32_difference=True)

    def test_typed_loss_all_model_gradients(self):
        for family in ("bert", "modernbert"):
            for permuted in (False, True):
                with self.subTest(family=family, permuted=permuted):
                    self.run_case(configuration(family, width=24, intermediate=36),
                        batch=6, options=5, tokens=13, critic=True,
                        supervised=True, permuted=permuted)

    def test_typed_loss_extremes_masks_temperatures_and_permutation(self):
        logits = torch.tensor([
            [10000., 1e20, -10000, 1e20, 9999],
            [-10000, -9999, -10001, 1e20, -10002],
            [-10000, 10000, 1e20, 1e20, 1e20],
            [-10000, 10000, 1e20, 1e20, 1e20],
            [1e20, 1e20, 1e20, -10000, 1e20],
        ])
        mask = torch.tensor([[1, 0, 1, 0, 1], [1, 1, 1, 0, 1],
                             [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 0]]).bool()
        targets = torch.tensor([4, 2, 1, 0, 3])
        temperatures = torch.tensor([.5, 1.8, 1.8, 1, .5])
        qtypes = [0, 1, 2, 2, 0]
        base_losses, base_gradients = [], []
        with tempfile.TemporaryDirectory(prefix="puffer-typed-loss-") as directory:
            inputs, results = (Path(directory) / name for name in ("input.bin", "output.bin"))
            for order in (list(range(5)), [3, 1, 4, 0, 2]):
                shuffled = logits[:, order].clone().requires_grad_()
                shuffled_mask = mask[:, order]
                shuffled_targets = torch.tensor([order.index(target) for target in targets.tolist()])
                with inputs.open("wb") as stream:
                    stream.write(struct.pack("<2I", *logits.shape))
                    write_tensor(stream, shuffled)
                    for tensor in (shuffled_mask, shuffled_targets):
                        stream.write(tensor.to(torch.int32).numpy().astype("<i4").tobytes())
                    write_tensor(stream, temperatures)
                result = subprocess.run([self.executable, "--loss-only", str(inputs), str(results)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with results.open("rb") as stream:
                    for i, temperature in enumerate((temperatures, torch.ones(5))):
                        shuffled.grad = None
                        losses = typed_losses(shuffled, shuffled_mask, shuffled_targets,
                            temperature, qtypes, order)
                        losses.mean().backward()
                        native_losses = read_floats(stream, 5)
                        native_gradient = read_floats(stream, 25).reshape(5, 5)
                        eval_losses = read_floats(stream, 5)
                        np.testing.assert_allclose(native_losses, losses.detach(), atol=2e-3, rtol=2e-6)
                        np.testing.assert_allclose(native_gradient, shuffled.grad, atol=2e-5, rtol=2e-4)
                        np.testing.assert_array_equal(native_gradient[~shuffled_mask.numpy()], 0)
                        np.testing.assert_array_equal(native_losses, eval_losses)
                        if order == list(range(5)):
                            base_losses.append(native_losses)
                            base_gradients.append(native_gradient)
                        else:
                            np.testing.assert_allclose(native_losses, base_losses[i], atol=2e-7, rtol=2e-7)
                            np.testing.assert_allclose(native_gradient, base_gradients[i][:, order],
                                atol=2e-7, rtol=2e-7)
                    self.assertFalse(stream.read(1))

    def test_adam_first_and_third_updates(self):
        initial = torch.tensor([0., 1., -2., 10., -1e-4, 1e4])
        gradients = torch.tensor([[0., 1e-8, -1., 1e3, .2, -1e-3],
                                  [1., -1e-7, .5, -10., 0., 1.],
                                  [-.5, 0., 1., .01, -.3, .02]])
        weights = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([weights], lr=.003, foreach=False)
        with tempfile.TemporaryDirectory(prefix="puffer-decision-adam-") as directory:
            inputs, results = (Path(directory) / name for name in ("input.bin", "output.bin"))
            with inputs.open("wb") as stream:
                stream.write(struct.pack("<2If", initial.numel(), len(gradients), .003))
                write_tensor(stream, initial)
                write_tensor(stream, gradients)
            result = subprocess.run([self.executable, "--adam-only", str(inputs), str(results)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with results.open("rb") as stream:
                for step, gradient in enumerate(gradients, 1):
                    weights.grad = gradient
                    optimizer.step()
                    state = optimizer.state[weights]
                    for name, expected in (("weights", weights), ("first", state["exp_avg"]),
                                           ("second", state["exp_avg_sq"])):
                        np.testing.assert_allclose(read_floats(stream, initial.numel()),
                            expected.detach(), atol=2e-7, rtol=2e-6, err_msg=f"Adam step {step} {name}")
                self.assertFalse(stream.read(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--gemm-precision", choices=("fp32", "tf32"), default="fp32")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_PRETRAINED_DECISION_TEST_BINARY"] = args.executable
    os.environ["PUFFER_PRETRAINED_GEMM_PRECISION"] = args.gemm_precision
    unittest.main(argv=[__file__, *remaining], verbosity=2)
