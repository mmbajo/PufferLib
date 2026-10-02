"""Native safetensors conversion/rejection checks and optional full cached Laya oracle.

The default tests need no GPU. The optional --laya-snapshot/--laya-bundle test
executes the actual imported checkpoint and needs a GPU, without downloading it.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    if __package__:
        from .test_import_pretrained import importer, make_source, package
        from .test_pretrained_encoder import read_u32, read_floats
    else:
        from test_import_pretrained import importer, make_source, package
        from test_pretrained_encoder import read_u32, read_floats
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "bundle fixtures need Torch, Transformers, NumPy and safetensors")
class PretrainedBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PUFFER_PRETRAINED_BUNDLE_TEST_BINARY")
        if not executable:
            raise unittest.SkipTest("set PUFFER_PRETRAINED_BUNDLE_TEST_BINARY or pass --executable")
        cls.executable = str(Path(executable).resolve())
        if not Path(cls.executable).is_file():
            raise AssertionError("missing bundle harness: " + cls.executable)
        torch.set_num_threads(1)

    def run_native(self, mode, bundle, output, inputs=None):
        command = [self.executable, mode, str(bundle), str(output)]
        if inputs is not None:
            command.append(str(inputs))
        return subprocess.run(command, capture_output=True, text=True)

    def test_native_tensor_conversion_and_metadata(self):
        for family, dtype, shards in (("bert", torch.float32, False),
                                      ("modernbert", torch.bfloat16, False),
                                      ("bert", torch.float16, True)):
            with self.subTest(family=family, dtype=dtype, shards=shards):
                with tempfile.TemporaryDirectory(prefix="puffer-native-bundle-") as directory:
                    directory = Path(directory)
                    source, bundle, output = (directory / name for name in ("source", "bundle", "output.bin"))
                    state = make_source(source, family, dtype, shards)
                    package(importer(), source, bundle)
                    result = self.run_native("dump", bundle, output)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    with output.open("rb") as stream:
                        self.assertEqual(stream.read(8), b"PUFPBD1\0")
                        metadata = struct.unpack("<11I", stream.read(44))
                        self.assertEqual(metadata, (int(family == "bert"), 8, 1, 2, 8, 2, 2, 2, 3, 4, 1))
                        self.assertEqual(read_u32(stream), len(state))
                        seen = set()
                        for _ in state:
                            name = stream.read(read_u32(stream)).decode()
                            self.assertTrue(name.startswith("encoder."))
                            source_name = name[len("encoder."):]
                            self.assertNotIn(source_name, seen)
                            seen.add(source_name)
                            count = struct.unpack("<Q", stream.read(8))[0]
                            expected = state[source_name].float().numpy().reshape(-1)
                            np.testing.assert_array_equal(read_floats(stream, count), expected, err_msg=name)
                        self.assertFalse(stream.read(1))

    def test_native_rejects_corrupt_or_incompatible_weights_before_output(self):
        with tempfile.TemporaryDirectory(prefix="puffer-native-corrupt-") as directory:
            directory = Path(directory)
            source, bundle, output = (directory / name for name in ("source", "bundle", "output.bin"))
            make_source(source)
            package(importer(), source, bundle)
            weights = bundle / "model.safetensors"
            manifest_path = bundle / "manifest.json"
            original_weights = weights.read_bytes()
            original_manifest = manifest_path.read_bytes()
            header_size = struct.unpack("<Q", original_weights[:8])[0]
            original_header = json.loads(original_weights[8:8 + header_size])
            original_payload = original_weights[8 + header_size:]
            first = next(name for name in original_header if name != "__metadata__")
            matrix = next(name for name in original_header if name != "__metadata__"
                          and len(original_header[name]["shape"]) == 2)
            for mutation in ("nan", "shape", "rename", "dtype", "overlap", "trailing", "manifest"):
                with self.subTest(mutation=mutation):
                    header = copy.deepcopy(original_header)
                    manifest = json.loads(original_manifest)
                    payload = bytearray(original_payload)
                    if mutation == "nan":
                        start = header[first]["data_offsets"][0]
                        payload[start:start + 4] = struct.pack("<f", float("nan"))
                    elif mutation == "shape":
                        header[matrix]["shape"] = list(reversed(header[matrix]["shape"]))
                        manifest["tensors"][matrix]["shape"] = header[matrix]["shape"]
                    elif mutation == "rename":
                        header[first + ".renamed"] = header.pop(first)
                        manifest["tensors"][first + ".renamed"] = manifest["tensors"].pop(first)
                    elif mutation == "dtype":
                        header[first]["dtype"] = manifest["tensors"][first]["dtype"] = "I32"
                    elif mutation == "overlap":
                        other = next(name for name in header if name not in (first, "__metadata__"))
                        header[other]["data_offsets"] = header[first]["data_offsets"]
                    elif mutation == "trailing":
                        payload.append(0)
                    else:
                        manifest["tensors"][first]["file"] = "missing.safetensors"
                    encoded = json.dumps(header).encode()
                    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
                    manifest_path.write_text(json.dumps(manifest))
                    result = self.run_native("dump", bundle, output)
                    self.assertNotEqual(result.returncode, 0, mutation)
                    self.assertIn("pretrained bundle parity:", result.stderr)
                    self.assertFalse(output.exists(), "validation must finish before producing output")
            weights.write_bytes(original_weights)
            manifest_path.write_bytes(original_manifest)

    def test_native_rejects_incompatible_head_dimension(self):
        for family in ("bert", "modernbert"):
            with self.subTest(family=family), tempfile.TemporaryDirectory(prefix="puffer-head-dimension-") as directory:
                directory = Path(directory)
                source, bundle, output = (directory / name for name in ("source", "bundle", "output.bin"))
                make_source(source, family)
                package(importer(), source, bundle)
                path = bundle / "encoder_config.json"
                config = json.loads(path.read_text())
                dimension = config["hidden_size"] // config["num_attention_heads"]
                for supported in (None, dimension):
                    config["head_dim"] = supported
                    path.write_text(json.dumps(config))
                    result = self.run_native("dump", bundle, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    output.unlink()
                config["head_dim"] = dimension + 2
                path.write_text(json.dumps(config))
                result = self.run_native("dump", bundle, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("head_dim must match", result.stderr)
                self.assertFalse(output.exists(), "configuration validation must precede output")

    def test_cached_full_laya_forward_on_real_texts(self):
        snapshot = os.environ.get("PUFFER_LAYA_SNAPSHOT")
        bundle = os.environ.get("PUFFER_LAYA_BUNDLE")
        if not snapshot or not bundle:
            self.skipTest("pass --laya-snapshot and --laya-bundle to compare the full cached model on GPU")
        from laya.common import build_model, build_sequence
        from transformers import AutoTokenizer
        snapshot, bundle = Path(snapshot), Path(bundle)
        config = json.loads((snapshot / "rl_agent_config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
        model = build_model(config, encoder_dir=str(snapshot / "encoder"), pretrained=False)
        model.load_state_dict(load_file(str(snapshot / "model.safetensors")), strict=True)
        model.encoder.set_attn_implementation("eager")
        model.eval()
        torch.set_num_threads(4)
        questions = [
            {"t": "choice", "ins": "Select the next move.", "crit": {"turn left": "", "go straight": "", "turn right": ""}},
            {"t": "score", "ins": "Rate how clearly the reply answers the question.", "crit": ["unclear", "partial", "clear"]},
            {"t": "noul", "ins": "The statement is supported by the text."},
        ]
        states = ["A snake is heading east. Food is one cell north and the eastern cell contains a wall.",
                  "Question: What is 2 + 2? Reply: Four. Café, 東京, and a literal [MASK] token.",
                  "The report says that rain started at noon and continued into the evening."]
        sequences = [build_sequence(tokenizer, state, question, max_len=128, head_max_len=72)
                     for state, question in zip(states, questions)]
        B, T, K = len(sequences), max(len(ids) for ids, _ in sequences), max(len(m) for _, m in sequences)
        ids = torch.full((B, T), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        markers = torch.full((B, K), -1, dtype=torch.long)
        marker_mask = torch.zeros((B, K), dtype=torch.bool)
        for i, (sequence, positions) in enumerate(sequences):
            ids[i, :len(sequence)], mask[i, :len(sequence)] = torch.tensor(sequence), 1
            markers[i, :len(positions)], marker_mask[i, :len(positions)] = torch.tensor(positions), True
        qtype = torch.arange(B)
        with torch.no_grad():
            logits, acts = model(ids, mask, markers, marker_mask, qtype)
        with tempfile.TemporaryDirectory(prefix="puffer-actual-laya-") as directory:
            inputs, output = (Path(directory) / name for name in ("input.bin", "output.bin"))
            with inputs.open("wb") as stream:
                stream.write(b"PUFPDI1\0")
                stream.write(struct.pack("<6I", config["head_layers"], acts.shape[-1], 0, B, T, K))
                for tensor in (ids, mask, torch.zeros_like(ids), markers, marker_mask, qtype):
                    stream.write(tensor.to(torch.int32).numpy().astype("<i4").tobytes())
            result = self.run_native("forward", bundle, output, inputs)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with output.open("rb") as stream:
                self.assertEqual(stream.read(8), b"PUFPDO1\0")
                self.assertEqual(struct.unpack("<4I", stream.read(16)), (B, K, acts.shape[-1], 0))
                np.testing.assert_allclose(read_floats(stream, B * K).reshape(B, K), logits.numpy(),
                                           atol=1e-3, rtol=1e-3, err_msg="full pretrained Laya option logits")
                np.testing.assert_allclose(read_floats(stream, acts.numel()).reshape(acts.shape), acts.numpy(),
                                           atol=1e-3, rtol=1e-3, err_msg="full pretrained Laya act logits")
                self.assertFalse(stream.read(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--laya-snapshot")
    parser.add_argument("--laya-bundle")
    args, remaining = parser.parse_known_args()
    for name, value in (("PUFFER_PRETRAINED_BUNDLE_TEST_BINARY", args.executable),
                        ("PUFFER_LAYA_SNAPSHOT", args.laya_snapshot), ("PUFFER_LAYA_BUNDLE", args.laya_bundle)):
        if value:
            os.environ[name] = value
    unittest.main(argv=[__file__, *remaining], verbosity=2)
