"""Independent PyTorch numerical oracle for the native CUDA Transformer.

Build the CUDA harness, then run inside an existing GPU allocation:
    python tests/test_decision_transformer.py --executable /path/to/native-test

PyTorch is a development dependency only. The oracle uses the upstream
TransformerEncoder implementation, including its automatic differentiation;
it does not reimplement the native attention or backward kernels.
"""

import argparse
import importlib.util
import itertools
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]


def load_exporter():
    spec = importlib.util.spec_from_file_location("decision_export", ROOT / "tools/export_decision.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter


if nn is not None:
    class ReferencePolicy(nn.Module):
        def __init__(self, width, layers, pooling, coordinates):
            super().__init__()
            self.pooling, self.coordinates = pooling, coordinates
            self.cells = nn.Embedding(102, width)
            self.cls = nn.Parameter(torch.empty(1, 1, width))
            self.positions = nn.Parameter(torch.empty(1, 101, width))
            layer = nn.TransformerEncoderLayer(
                width, 4, 4 * width, dropout=0, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
            )
            self.action_head = nn.Linear(width, 4)
            self.value_head = nn.Linear(width, 1)
            nn.init.normal_(self.cls, std=0.02)
            nn.init.normal_(self.positions, std=0.02)
            with torch.no_grad():
                self.cells.weight.mul_(0.02)
            if coordinates == "head-relative":
                self.relative_rows = nn.Embedding(19, width)
                self.relative_columns = nn.Embedding(19, width)
                nn.init.normal_(self.relative_rows.weight, std=0.02)
                nn.init.normal_(self.relative_columns.weight, std=0.02)

        def forward(self, boards):
            batch = len(boards)
            head = (boards == 1).long().argmax(dim=1)
            cells = self.cells(boards + 1)
            if self.coordinates == "head-relative":
                positions = torch.arange(100)
                cells = (cells + self.relative_rows(positions // 10 - head[:, None] // 10 + 9)
                         + self.relative_columns(positions % 10 - head[:, None] % 10 + 9))
            tokens = torch.cat((self.cls.expand(batch, -1, -1), cells), dim=1) + self.positions
            encoded = self.encoder(tokens)
            selected = encoded[torch.arange(batch), head + 1] if self.pooling == "head" else encoded[:, 0]
            return self.action_head(selected), self.value_head(selected).squeeze(-1)


def write_floats(stream, tensor):
    stream.write(tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes())


def read_u32(stream):
    return struct.unpack("<I", stream.read(4))[0]


def read_floats(stream, count):
    raw = stream.read(4 * count)
    if len(raw) != 4 * count:
        raise AssertionError("Truncated native test output")
    return np.frombuffer(raw, dtype="<f4").copy()


def read_results(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"PUFDTR1\0"
        batch, iterations, parameter_count = (read_u32(stream) for _ in range(3))
        records = []
        for _ in range(iterations):
            record = {
                "logits": read_floats(stream, batch * 4).reshape(batch, 4),
                "values": read_floats(stream, batch),
                "norm": float(read_floats(stream, 1)[0]),
                "parameters": {},
            }
            for _ in range(parameter_count):
                name = stream.read(read_u32(stream)).decode()
                count = struct.unpack("<Q", stream.read(8))[0]
                record["parameters"][name] = {
                    key: read_floats(stream, count)
                    for key in ("gradient", "weight")
                }
            records.append(record)
        assert not stream.read(1), "Trailing native test output"
    return records


@unittest.skipIf(torch is None, "numerical parity tests need PyTorch and NumPy")
class DecisionExporterTest(unittest.TestCase):
    def test_named_export_preserves_every_weight_and_shape(self):
        torch.manual_seed(134)
        policy = ReferencePolicy(8, 2, "head", "head-relative")
        state = policy.state_dict()
        config = dict(width=8, layers=2, pooling="head", coordinates="head-relative")
        with tempfile.TemporaryDirectory(prefix="decision-export-") as directory:
            path = Path(directory) / "model.bin"
            load_exporter().write_checkpoint(path, state, config)
            with path.open("rb") as stream:
                self.assertEqual(stream.read(8), b"PUFDT01\0")
                self.assertEqual(struct.unpack("<5I", stream.read(20)), (8, 2, 1, 1, len(state)))
                for expected_name, expected_tensor in state.items():
                    name = stream.read(read_u32(stream)).decode()
                    self.assertEqual(name, expected_name)
                    dimensions = tuple(read_u32(stream) for _ in range(read_u32(stream)))
                    self.assertEqual(dimensions, tuple(expected_tensor.shape), name)
                    count = struct.unpack("<Q", stream.read(8))[0]
                    self.assertEqual(count, expected_tensor.numel(), name)
                    np.testing.assert_array_equal(read_floats(stream, count), expected_tensor.numpy().flatten())
                self.assertFalse(stream.read(1))

    def test_rejected_exports_preserve_the_existing_checkpoint(self):
        policy = ReferencePolicy(8, 1, "cls", "absolute")
        state = policy.state_dict()
        config = dict(width=8, layers=1, pooling="cls", coordinates="absolute")
        bad_shape = dict(state, **{"action_head.weight": torch.zeros(4, 7)})
        nonfinite = dict(state, **{"value_head.bias": torch.tensor([float("nan")])})
        missing = {name: value for name, value in state.items() if name != "cls"}
        extra = dict(state, unexpected=torch.ones(1))
        variants = [
            (state, dict(config, body_features="tail-distance")),
            (bad_shape, config), (nonfinite, config), (missing, config), (extra, config),
        ]
        exporter = load_exporter()
        with tempfile.TemporaryDirectory(prefix="decision-export-") as directory:
            path = Path(directory) / "model.bin"
            exporter.write_checkpoint(path, state, config)
            original = path.read_bytes()
            for index, (weights, settings) in enumerate(variants):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        exporter.write_checkpoint(path, weights, settings)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(Path(directory).iterdir()), [path])


@unittest.skipIf(torch is None, "numerical parity tests need PyTorch and NumPy")
class DecisionTransformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PUFFER_DECISION_TEST_BINARY")
        if not executable:
            raise unittest.SkipTest("set PUFFER_DECISION_TEST_BINARY or pass --executable")
        cls.executable = str(Path(executable).resolve())
        if not Path(cls.executable).is_file():
            raise AssertionError(f"Missing native harness: {cls.executable}")
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        cls.exporter = load_exporter()

    def assert_close(self, actual, expected, label, atol=4e-5, rtol=4e-4):
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().numpy()
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape, label)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)

    def invoke(self, mode, model, inputs, outputs, check=True):
        result = subprocess.run(
            [self.executable, mode, str(model), str(inputs), str(outputs)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def model_case(self, width, layers, pooling, coordinates, batch, iterations=1,
                   external=False, accumulate=False):
        torch.manual_seed(3471)
        reference = ReferencePolicy(width, layers, pooling, coordinates)
        config = dict(width=width, layers=layers, pooling=pooling, coordinates=coordinates,
                      body_features="ordered")
        # Repeated empty cells exercise embedding gradient accumulation; heads
        # at an interior cell and opposite corners exercise relative indexing.
        boards = torch.zeros((3, 100), dtype=torch.long)
        boards[0, [55, 54, 53, 4]] = torch.tensor([1, 2, 3, -1])
        boards[1, [0, 1, 2, 3, 99]] = torch.tensor([1, 2, 3, 4, -1])
        full_path = [row * 10 + column for row in range(9, -1, -1)
                     for column in (range(9, -1, -1) if row % 2 else range(10))]
        boards[2, full_path] = torch.arange(1, 101)  # Full-board terminal observation.
        boards = boards[:batch]
        dlogits, dvalues = torch.randn(batch, 4) / batch, torch.randn(batch) / batch
        learning_rate = 3e-3
        with tempfile.TemporaryDirectory(prefix="decision-transformer-") as directory:
            directory = Path(directory)
            model, inputs, output = (directory / name for name in ("model.bin", "input.bin", "output.bin"))
            self.exporter.write_checkpoint(model, reference.state_dict(), config)
            with inputs.open("wb") as stream:
                stream.write(b"PUFDTI1\0")
                stream.write(struct.pack("<IIff", batch, iterations, learning_rate, 0.0))
                stream.write(boards.to(torch.int32).numpy().astype("<i4").tobytes())
                write_floats(stream, dlogits)
                write_floats(stream, dvalues)
            mode = "bound" if external else "overfit" if iterations > 1 else "forward"
            if accumulate:
                mode = "accumulate"
            self.invoke(mode, model, inputs, output)
            self.assertEqual(model.read_bytes(), Path(str(output) + ".checkpoint").read_bytes(),
                             "native load/save must preserve all exported tensors")
            records = read_results(output)
            self.assertEqual(len(records), iterations)
            optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
            losses = []
            for step, record in enumerate(records):
                optimizer.zero_grad(set_to_none=True)
                logits, values = reference(boards)
                self.assert_close(record["logits"], logits, f"step {step} logits")
                self.assert_close(record["values"], values, f"step {step} values")
                if mode == "overfit":
                    loss = .5 * ((logits - dlogits).square().sum() + (values - dvalues).square().sum()) / batch
                    # Judge learning from native outputs, independently of the
                    # parameter alignment used for each numerical comparison.
                    native_loss = .5 * (np.square(record["logits"] - dlogits.numpy()).sum()
                                        + np.square(record["values"] - dvalues.numpy()).sum()) / batch
                    losses.append(float(native_loss))
                else:
                    loss = (logits * dlogits).sum() + (values * dvalues).sum()
                    if accumulate:
                        loss = 2 * loss
                loss.backward()
                self.assertEqual(set(record["parameters"]), set(dict(reference.named_parameters())))
                for name, parameter in reference.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assert_close(record["parameters"][name]["gradient"], parameter.grad.flatten(),
                                      f"step {step} {name} gradient", atol=1.5e-4, rtol=8e-4)
                if mode == "overfit":
                    optimizer.step()
                for name, parameter in reference.named_parameters():
                    result = record["parameters"][name]
                    self.assert_close(result["weight"], parameter.flatten(), f"step {step} {name} weight",
                                      atol=3e-5, rtol=3e-4)
                if mode == "overfit":
                    # CUDA and CPU reductions have different FP32 rounding.
                    # Start the next oracle update from the preceding native
                    # weights, after validating this update. This isolates the
                    # error of each forward/backward/SGD step instead of letting
                    # two slightly different optimization trajectories diverge.
                    with torch.no_grad():
                        for name, parameter in reference.named_parameters():
                            parameter.copy_(torch.from_numpy(record["parameters"][name]["weight"])
                                            .reshape_as(parameter))
            if mode == "overfit":
                self.assertLess(losses[-1], 0.6 * losses[0], "SGD must reduce the fixture loss by at least 40%")

    def test_forward_and_every_parameter_gradient(self):
        # Eight cases cover both choices on every axis and both batch sizes.
        for width, layers, head in itertools.product((8, 32), (1, 2), (False, True)):
            pooling = "head" if head else "cls"
            relative = (layers == 1) == head
            coordinates = "head-relative" if relative else "absolute"
            batch = 3 if (width == 8) == relative else 1
            with self.subTest(width=width, layers=layers, pooling=pooling,
                              coordinates=coordinates, batch=batch):
                self.model_case(width, layers, pooling, coordinates, batch)

    def test_native_backward_can_overfit_a_small_board_batch(self):
        self.model_case(8, 2, "head", "head-relative", 3, iterations=25)

    def test_external_parameter_and_gradient_storage(self):
        self.model_case(8, 2, "head", "head-relative", 3, external=True)

    def test_backward_accumulates_parameter_gradients(self):
        self.model_case(8, 2, "cls", "head-relative", 3, accumulate=True)

    def test_checkpoint_orders_nonblocking_stream_and_rejects_partial_load(self):
        reference = ReferencePolicy(8, 1, "head", "head-relative")
        config = dict(width=8, layers=1, pooling="head", coordinates="head-relative")
        with tempfile.TemporaryDirectory(prefix="decision-checkpoint-") as directory:
            directory = Path(directory)
            initial, corrupt, output = (directory / name for name in
                                        ("initial.bin", "corrupt.bin", "result.txt"))
            self.exporter.write_checkpoint(initial, reference.state_dict(), config)
            damaged = bytearray(initial.read_bytes())
            damaged[-4:] = struct.pack("<f", float("nan"))
            corrupt.write_bytes(damaged)
            self.invoke("checkpoint", initial, corrupt, output)
            self.assertEqual(output.read_text(), "checkpoint stream and corruption checks passed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_DECISION_TEST_BINARY"] = args.executable
    unittest.main(argv=[__file__, *remaining], verbosity=2)
