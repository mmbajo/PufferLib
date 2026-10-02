"""Hugging Face/PyTorch oracle for native BERT, ModernBERT and Laya head layers.

Uses small randomly initialized public reference architectures and no downloads.
Run with --executable pointing to the compiled CUDA harness, inside an existing
GPU allocation. All named parameter gradients and intermediate layers are checked.
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
    from torch import nn
    from transformers import BertConfig, BertModel, ModernBertConfig, ModernBertModel
except ImportError:
    torch = None


FAMILIES = {"modernbert": 0, "bert": 1, "laya_head": 2}


def write_tensor(stream, tensor):
    stream.write(tensor.detach().contiguous().numpy().astype("<f4").tobytes())


def read_u32(stream):
    return struct.unpack("<I", stream.read(4))[0]


def read_floats(stream, count):
    raw = stream.read(count * 4)
    if len(raw) != count * 4:
        raise AssertionError("truncated encoder test output")
    return np.frombuffer(raw, dtype="<f4").copy()


def write_model(path, config, state):
    with path.open("wb") as stream:
        stream.write(b"PUFPRE1\0")
        stream.write(struct.pack("<9i", FAMILIES[config["family"]], *[
            config[name] for name in ("width", "layers", "heads", "intermediate", "vocab",
                                      "max_positions", "type_vocab", "pad_token_id")]))
        stream.write(struct.pack("<f6i2f", config["epsilon"], *[
            int(config[name]) for name in ("attention_bias", "mlp_bias", "norm_bias", "local_window", "global_every")],
            len(config["layer_types"]), config["global_rope_theta"], config["local_rope_theta"]))
        for layer_type in config["layer_types"]:
            stream.write(struct.pack("<i", layer_type))
        stream.write(struct.pack("<I", len(state)))
        for name, tensor in state.items():
            encoded = name.encode()
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<I", tensor.ndim))
            stream.write(struct.pack("<" + "I" * tensor.ndim, *tensor.shape))
            stream.write(struct.pack("<Q", tensor.numel()))
            write_tensor(stream, tensor)


def reference_model(config):
    common = dict(hidden_size=config["width"], num_hidden_layers=config["layers"],
                  num_attention_heads=config["heads"], intermediate_size=config["intermediate"],
                  vocab_size=config["vocab"], max_position_embeddings=config["max_positions"],
                  pad_token_id=config["pad_token_id"])
    if config["family"] == "bert":
        hf = BertConfig(**common, type_vocab_size=config["type_vocab"], hidden_act="gelu",
                        layer_norm_eps=config["epsilon"], hidden_dropout_prob=0,
                        attention_probs_dropout_prob=0)
        hf._attn_implementation = "eager"
        model = BertModel(hf, add_pooling_layer=False)
        layers = model.encoder.layer
    elif config["family"] == "modernbert":
        layer_types = config["layer_types"] or [int(i % config["global_every"] != 0)
                                               for i in range(config["layers"])]
        hf = ModernBertConfig(**common, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
                              norm_eps=config["epsilon"], norm_bias=config["norm_bias"],
                              attention_bias=config["attention_bias"], mlp_bias=config["mlp_bias"],
                              attention_dropout=0, embedding_dropout=0, mlp_dropout=0,
                              local_attention=config["local_window"],
                              layer_types=["sliding_attention" if value else "full_attention" for value in layer_types],
                              rope_parameters={
                                  "full_attention": {"rope_type": "default", "rope_theta": config["global_rope_theta"]},
                                  "sliding_attention": {"rope_type": "default", "rope_theta": config["local_rope_theta"]},
                              })
        hf._attn_implementation = "eager"
        model = ModernBertModel(hf)
        layers = model.layers
    else:
        layer = nn.TransformerEncoderLayer(config["width"], config["heads"], config["intermediate"],
                                           dropout=0, activation="relu", batch_first=True, norm_first=True,
                                           layer_norm_eps=config["epsilon"])
        model = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
        layers = model.layers
    return model, layers


def configuration(family, **changes):
    config = dict(family=family, width=16, layers=2, heads=4, intermediate=40,
                  vocab=64, max_positions=64, type_vocab=2, pad_token_id=0,
                  epsilon=1e-12 if family == "bert" else 1e-5,
                  attention_bias=False, mlp_bias=False, norm_bias=False,
                  local_window=8, global_every=3, global_rope_theta=160000., local_rope_theta=10000.,
                  layer_types=[])
    config.update(changes)
    return config


@unittest.skipIf(torch is None, "encoder parity needs PyTorch, NumPy and Transformers")
class PretrainedEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PUFFER_PRETRAINED_TEST_BINARY")
        if not executable:
            raise unittest.SkipTest("set PUFFER_PRETRAINED_TEST_BINARY or pass --executable")
        cls.executable = str(Path(executable).resolve())
        if not Path(cls.executable).is_file():
            raise AssertionError(f"missing encoder test harness: {cls.executable}")
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def compare(self, actual, expected, description, gradient=False):
        expected = expected.detach().numpy()
        self.assertEqual(actual.shape, expected.shape, description)
        np.testing.assert_allclose(actual, expected, atol=2e-4 if gradient else 4e-5,
                                   rtol=1e-3 if gradient else 5e-4, err_msg=description)

    def run_case(self, config, batch=2, tokens=13):
        torch.manual_seed(9824)
        model, layers = reference_model(config)
        model.train()  # Dropout is explicitly zero; disable inference-only shortcuts.
        ids = torch.randint(3, config["vocab"], (batch, tokens))
        ids[:, 0], ids[:, -1] = 1, 2
        if batch > 1:
            ids[0, 4] = 2
            ids[0, 5:] = config["pad_token_id"]
        mask = ids.ne(config["pad_token_id"]).long()
        types = torch.arange(tokens).expand(batch, -1) % config["type_vocab"]
        hidden = torch.randn(batch, tokens, config["width"], requires_grad=True)
        upstream = torch.randn(batch, tokens, config["width"]) / (batch * tokens)
        intermediate = []

        def capture(_module, _arguments, result):
            if isinstance(result, tuple):
                result = result[0]
            intermediate.append(result.detach().clone())

        for layer in layers:
            layer.register_forward_hook(capture)
        if config["family"] == "laya_head":
            output = model(hidden, src_key_padding_mask=~mask.bool())
        else:
            kwargs = dict(input_ids=ids, attention_mask=mask)
            if config["family"] == "bert":
                kwargs["token_type_ids"] = types
            output = model(**kwargs).last_hidden_state
        (output * upstream).sum().backward()

        with tempfile.TemporaryDirectory(prefix="puffer-pretrained-") as directory:
            directory = Path(directory)
            weights, inputs, results = (directory / name for name in ("weights.bin", "inputs.bin", "results.bin"))
            write_model(weights, config, model.state_dict())
            with inputs.open("wb") as stream:
                stream.write(b"PUFPRI1\0")
                stream.write(struct.pack("<II", batch, tokens))
                for tensor in (ids, mask, types):
                    stream.write(tensor.to(torch.int32).numpy().astype("<i4").tobytes())
                write_tensor(stream, hidden)
                write_tensor(stream, upstream)
            result = subprocess.run([self.executable, str(weights), str(inputs), str(results)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with results.open("rb") as stream:
                self.assertEqual(stream.read(8), b"PUFPRO1\0")
                self.assertEqual(struct.unpack("<4I", stream.read(16)),
                                 (batch, tokens, config["width"], config["layers"]))
                shape = (batch, tokens, config["width"])
                elements = batch * tokens * config["width"]
                self.compare(read_floats(stream, elements).reshape(shape), output, "encoder output")
                self.assertEqual(len(intermediate), config["layers"])
                for index, expected in enumerate(intermediate):
                    self.compare(read_floats(stream, elements).reshape(shape), expected, f"layer {index} output")
                input_gradient = read_floats(stream, elements).reshape(shape)
                if config["family"] == "laya_head":
                    self.compare(input_gradient, hidden.grad, "head input gradient", gradient=True)
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

    def test_modernbert_rope_geglu_and_padding(self):
        self.run_case(configuration("modernbert", layers=4, intermediate=24))

    def test_modernbert_explicit_layer_types_and_biases(self):
        self.run_case(configuration("modernbert", width=32, layers=3, intermediate=48,
                                    layer_types=[1, 0, 1], local_window=4,
                                    attention_bias=True, mlp_bias=True, norm_bias=True,
                                    global_rope_theta=1000., local_rope_theta=20000.))

    def test_bert_postnorm_positions_and_token_types(self):
        self.run_case(configuration("bert"))

    def test_bert_alternate_head_width(self):
        self.run_case(configuration("bert", width=24, intermediate=36, layers=1, type_vocab=3),
                      batch=1, tokens=7)

    def test_laya_head_prenorm_relu(self):
        self.run_case(configuration("laya_head", heads=1, intermediate=64))

    def test_laya_head_multiple_attention_heads(self):
        self.run_case(configuration("laya_head", width=128, heads=2, intermediate=512, layers=1),
                      batch=1, tokens=7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_PRETRAINED_TEST_BINARY"] = args.executable
    unittest.main(argv=[__file__, *remaining], verbosity=2)
