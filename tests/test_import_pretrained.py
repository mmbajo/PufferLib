"""Offline pretrained packaging checks; no Hub access or CUDA is required."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, processors
    from transformers import BertConfig, BertModel, ModernBertConfig, ModernBertModel
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def importer():
    spec = importlib.util.spec_from_file_location("import_pretrained", ROOT / "tools/import_pretrained.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(path, family="bert", dtype=None, shards=False):
    path.mkdir()
    kwargs = dict(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                  intermediate_size=12, max_position_embeddings=64, pad_token_id=1)
    if family == "bert":
        model = BertModel(BertConfig(**kwargs, type_vocab_size=2), add_pooling_layer=False)
    else:
        model = ModernBertModel(ModernBertConfig(**kwargs, cls_token_id=2, sep_token_id=3,
                                               bos_token_id=2, eos_token_id=3, local_attention=8))
    model.config.save_pretrained(path)
    state = {name: tensor.to(dtype or torch.float32).contiguous() for name, tensor in model.state_dict().items()}
    if shards:
        names = list(state)
        split = len(names) // 2
        mapping = {}
        for i, selection in enumerate((names[:split], names[split:]), start=1):
            filename = f"model-{i:05d}-of-00002.safetensors"
            save_file({name: state[name] for name in selection}, path / filename)
            mapping.update({name: filename for name in selection})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_file(state, path / "model.safetensors")
    vocabulary = dict(zip(("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##s"), range(8)))
    tokenizer = Tokenizer(models.WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]", "pad_token": "[PAD]",
        "unk_token": "[UNK]", "tokenizer_class": "PreTrainedTokenizerFast",
    }))
    return state


def package(module, source, destination):
    return module.import_bundle(source, destination, max_length=32, head_max_length=16)


class SafetensorsHeaderTest(unittest.TestCase):
    def test_remote_sources_pin_moving_revisions_once_without_network(self):
        module = importer()
        info = mock.Mock(side_effect=[types.SimpleNamespace(sha="a" * 40), types.SimpleNamespace(sha="b" * 40)])
        download = mock.Mock(return_value="/unused/mock-cache/file")
        api = mock.Mock(return_value=types.SimpleNamespace(model_info=info))
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi, hub.hf_hub_download = api, download
        errors = types.ModuleType("huggingface_hub.errors")
        errors.EntryNotFoundError = type("EntryNotFoundError", (Exception,), {})
        with mock.patch.dict("sys.modules", {"huggingface_hub": hub, "huggingface_hub.errors": errors}):
            source = module.Source("mock-organization/encoder", "main")
            source.get("config.json")
            source.get("model.safetensors")
            tokenizer = module.Source("mock-organization/tokenizer", "release")
            tokenizer.get("tokenizer.json")
            pinned = module.Source("mock-organization/pinned", "c" * 40)
            pinned.get("config.json")
        self.assertEqual(source.revision, "a" * 40)
        self.assertEqual(tokenizer.revision, "b" * 40)
        self.assertEqual(info.call_args_list, [mock.call("mock-organization/encoder", revision="main"),
                                               mock.call("mock-organization/tokenizer", revision="release")])
        self.assertEqual(download.call_args_list, [
            mock.call("mock-organization/encoder", "config.json", revision="a" * 40),
            mock.call("mock-organization/encoder", "model.safetensors", revision="a" * 40),
            mock.call("mock-organization/tokenizer", "tokenizer.json", revision="b" * 40),
            mock.call("mock-organization/pinned", "config.json", revision="c" * 40),
        ])

    def test_rejects_invalid_extents_dtypes_and_unindexed_bytes(self):
        module = importer()
        malformed = [
            ({"a": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}, b"\0" * 8),
            ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\0" * 4),
            ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
              "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4),
            ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, b"\0" * 8),
            ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 8),
        ]
        with tempfile.TemporaryDirectory(prefix="puffer-import-header-") as directory:
            path = Path(directory) / "model.safetensors"
            for header, payload in malformed:
                with self.subTest(header=header):
                    raw = json.dumps(header).encode()
                    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
                    with self.assertRaises(ValueError):
                        module.tensor_header(path)


@unittest.skipIf(torch is None, "packaging fixtures need Torch, Transformers, tokenizers and safetensors")
class PretrainedImporterTest(unittest.TestCase):
    def test_imports_both_families_and_preserves_dtype_shards_and_tokenizer(self):
        module = importer()
        cases = [("bert", torch.float32, False), ("modernbert", torch.bfloat16, False),
                 ("bert", torch.float16, True)]
        for family, dtype, shards in cases:
            with self.subTest(family=family, dtype=dtype, shards=shards):
                with tempfile.TemporaryDirectory(prefix="puffer-import-") as directory:
                    source, destination = (Path(directory) / name for name in ("source", "bundle"))
                    state = make_source(source, family, dtype, shards)
                    manifest = package(module, source, destination)
                    self.assertEqual(manifest["format"], "puffer-pretrained-v1")
                    self.assertEqual(manifest["family"], family)
                    self.assertEqual(manifest["kind"], "encoder")
                    self.assertEqual(manifest["source_encoder_prefix"], "")
                    self.assertEqual(set(manifest["tensors"]), set(state))
                    for name, tensor in state.items():
                        self.assertEqual(manifest["tensors"][name]["shape"], list(tensor.shape))
                        self.assertEqual(manifest["tensors"][name]["dtype"],
                                         {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}[dtype])
                    self.assertEqual((destination / "tokenizer.json").read_bytes(),
                                     (source / "tokenizer.json").read_bytes())
                    for weights in manifest["weights"]:
                        copied = (destination / weights["file"]).read_bytes()
                        self.assertEqual(copied, (source / weights["file"]).read_bytes())
                        self.assertEqual(hashlib.sha256(copied).hexdigest(), weights["sha256"])

    def test_rejects_unsupported_config_and_tokenizer_without_publishing(self):
        module = importer()
        variants = [
            ("config.json", "model_type", "roberta"),
            ("config.json", "hidden_size", 7),
            ("config.json", "is_decoder", True),
            ("config.json", "position_embedding_type", "relative_key"),
            ("config.json", "hidden_act", "gelu_new"),
            ("config.json", "head_dim", 8),
            ("config.json", "intermediate_size", 65537),
            ("config.json", "max_position_embeddings", 2147483648),
            ("tokenizer_config.json", "mask_token", None),
        ]
        with tempfile.TemporaryDirectory(prefix="puffer-import-reject-") as directory:
            source, destination = (Path(directory) / name for name in ("source", "bundle"))
            make_source(source)
            for filename, key, value in variants:
                with self.subTest(filename=filename, key=key):
                    path = source / filename
                    original = path.read_bytes()
                    contents = json.loads(original)
                    contents[key] = value
                    path.write_text(json.dumps(contents))
                    with self.assertRaises(ValueError):
                        package(module, source, destination)
                    self.assertFalse(destination.exists())
                    self.assertEqual(list(Path(directory).iterdir()), [source])
                    path.write_bytes(original)
            path = source / "tokenizer.json"
            contents = json.loads(path.read_text())
            contents["added_tokens"].append({"id": 999, "content": "bad"})
            path.write_text(json.dumps(contents))
            with self.assertRaises(ValueError):
                package(module, source, destination)
            self.assertFalse(destination.exists())

    def test_existing_destination_is_preserved(self):
        module = importer()
        with tempfile.TemporaryDirectory(prefix="puffer-import-existing-") as directory:
            source, destination = (Path(directory) / name for name in ("source", "bundle"))
            make_source(source)
            package(module, source, destination)
            original = {path.name: path.read_bytes() for path in destination.iterdir()}
            with self.assertRaises(ValueError):
                package(module, source, destination)
            self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()}, original)
            self.assertEqual(set(Path(directory).iterdir()), {source, destination})

    def test_rejects_missing_or_wrong_shape_required_weights(self):
        module = importer()
        for family in ("bert", "modernbert"):
            with tempfile.TemporaryDirectory(prefix="puffer-import-architecture-") as directory:
                source, destination = (Path(directory) / name for name in ("source", "bundle"))
                state = make_source(source, family)
                # All tensors remain valid safetensors; only the architecture is invalid.
                names = list(state)
                cases = [("missing", names[-1]), ("wrong_shape", names[0]),
                         ("renamed", names[len(names) // 2])]
                for mutation, name in cases:
                    with self.subTest(family=family, mutation=mutation, name=name):
                        changed = dict(state)
                        if mutation == "missing":
                            del changed[name]
                        elif mutation == "renamed":
                            changed[name + ".unexpected"] = changed.pop(name)
                        else:
                            changed[name] = changed[name].reshape(-1)[:1].clone()
                        save_file(changed, source / "model.safetensors")
                        with self.assertRaises(ValueError):
                            package(module, source, destination)
                        self.assertFalse(destination.exists())
                        self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_full_laya_preserves_heads_and_distinct_temperature_metadata(self):
        try:
            from laya.common import DecisionModel
        except ImportError:
            self.skipTest("full Laya packaging fixture needs the public laya package")
        module = importer()
        with tempfile.TemporaryDirectory(prefix="puffer-import-laya-") as directory:
            directory = Path(directory)
            source, destination = directory / "source", directory / "bundle"
            make_source(source, "modernbert")
            config = ModernBertConfig.from_pretrained(source, local_files_only=True)
            model = DecisionModel(ModernBertModel(config), head_layers=1, n_act=2, dropout=0)
            state = model.state_dict()
            save_file(state, source / "model.safetensors")
            (source / "encoder").mkdir()
            (source / "config.json").rename(source / "encoder/config.json")
            (source / "tokenizer").mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (source / name).rename(source / "tokenizer" / name)
            decision = {"head_layers": 1, "act_costs": {"escalate": 0.5},
                        "temperature": [1.5, 2.0, 3.0], "temperature_by_options": {"choice:11+": 0.1},
                        "max_len": 32, "head_max_len": 16}
            (source / "rl_agent_config.json").write_text(json.dumps(decision))
            manifest = package(module, source, destination)
            self.assertEqual(manifest["kind"], "laya")
            self.assertEqual(manifest["source_encoder_prefix"], "encoder.")
            self.assertEqual(set(manifest["tensors"]), set(state))
            self.assertEqual(json.loads((destination / "decision_config.json").read_text()), decision)
            # Calibration metadata must not overwrite the checkpoint's original buffer.
            self.assertEqual(state["temperature"].tolist(), [1., 1., 1.])
            self.assertEqual((destination / "model.safetensors").read_bytes(),
                             (source / "model.safetensors").read_bytes())
            for missing in ("temperature", "act_head.2.weight", "scorer.3.bias"):
                with self.subTest(missing=missing):
                    changed = dict(state)
                    del changed[missing]
                    save_file(changed, source / "model.safetensors")
                    failed = directory / "rejected"
                    with self.assertRaises(ValueError):
                        package(module, source, failed)
                    self.assertFalse(failed.exists())
            save_file(state, source / "model.safetensors")
            variants = [("temperature", [1, 2]), ("temperature", [True, 1, 1]),
                        ("temperature", [0, 1, 1]), ("temperature", [float("inf"), 1, 1]),
                        ("temperature_by_options", {"choice:2": -1}),
                        ("temperature_by_options", {"choice:2": float("nan")})]
            for key, value in variants:
                with self.subTest(key=key, value=value):
                    (source / "rl_agent_config.json").write_text(json.dumps(dict(decision, **{key: value})))
                    with self.assertRaises(ValueError):
                        package(module, source, directory / "rejected")
                    self.assertFalse((directory / "rejected").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
