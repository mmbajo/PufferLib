#!/usr/bin/env python3
"""Import a Laya or Hugging Face BERT/ModernBERT checkpoint and its tokenizer.

Weights retain the safetensors representation; native loading converts FP16,
BF16 or FP32 to FP32. Python is an offline packaging dependency only. Remote
sources require huggingface_hub; local safetensors directories need stdlib only.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import re
import struct
import tempfile


FORMAT = "puffer-pretrained-v1"
DTYPES = {"F32": 4, "F16": 2, "BF16": 2}


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_header(path):
    size = path.stat().st_size
    with open(path, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= min(32 << 20, size - 8):
            raise ValueError(f"{path}: invalid safetensors header length")
        header = json.loads(stream.read(length))
    tensors, intervals = {}, []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if dtype not in DTYPES or not isinstance(shape, list) or not 1 <= len(shape) <= 4:
            raise ValueError(f"{name}: unsupported tensor type/rank")
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 1:
                raise ValueError(f"{name}: invalid dimension")
            count *= dim
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)):
            raise ValueError(f"{name}: invalid offsets")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length or end - start != count * DTYPES[dtype]:
            raise ValueError(f"{name}: tensor extent does not match shape")
        intervals.append((start, end))
        tensors[name] = {"dtype": dtype, "shape": shape}
    intervals.sort()
    previous = 0
    for start, end in intervals:
        if start != previous:
            raise ValueError(f"{path}: overlapping or noncontiguous tensor data")
        previous = end
    if previous != size - 8 - length:
        raise ValueError(f"{path}: unindexed tensor data")
    return tensors


class Source:
    def __init__(self, location, revision=None):
        self.location, self.revision = str(location), revision
        self.local = Path(location) if Path(location).is_dir() else None
        if self.local is None and not (revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise ValueError("remote imports require huggingface_hub; use a local directory otherwise") from error
            # Resolve a moving branch/tag once, so config, weights and tokenizer
            # cannot come from different commits during one import.
            self.revision = HfApi().model_info(self.location, revision=revision).sha

    def get(self, name, optional=False):
        if self.local is not None:
            path = self.local / name
            if path.is_file():
                return path
            if optional:
                return None
            raise ValueError(f"missing checkpoint file: {path}")
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError
        except ImportError as error:
            raise ValueError("remote imports require huggingface_hub; use a local directory otherwise") from error
        try:
            return Path(hf_hub_download(self.location, name, revision=self.revision))
        except EntryNotFoundError:
            if optional:
                return None
            raise


def validate_config(config):
    family = config.get("model_type")
    if family not in ("bert", "modernbert"):
        raise ValueError(f"unsupported model_type {family!r}; native families are bert and modernbert")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                "vocab_size", "max_position_embeddings"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    width, heads = config["hidden_size"], config["num_attention_heads"]
    if width % heads or width > 16384 or config["num_hidden_layers"] > 128:
        raise ValueError("unsupported encoder dimensions")
    if config.get("head_dim") not in (None, width // heads):
        raise ValueError("head_dim overrides are not supported")
    if config["intermediate_size"] > 65536 or any(config[key] > 2147483647
            for key in ("vocab_size", "max_position_embeddings")):
        raise ValueError("encoder dimensions exceed native limits")
    if config.get("is_decoder", False) or config.get("add_cross_attention", False):
        raise ValueError("decoder and cross-attention checkpoints are not supported")
    if family == "bert":
        if config.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError("BERT relative-position attention is not supported")
        if config.get("hidden_act", "gelu") != "gelu":
            raise ValueError("BERT currently requires exact GELU")
        if type(config.get("type_vocab_size", 2)) is not int or config.get("type_vocab_size", 2) < 1:
            raise ValueError("type_vocab_size must be a positive integer")
    else:
        if width // heads % 2 or config.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("ModernBERT requires an even head dimension and GELU gating")
        if config.get("rope_scaling"):
            raise ValueError("scaled rotary positions are not supported")
        for value in config.get("rope_parameters", {}).values():
            if isinstance(value, dict) and value.get("rope_type", "default") != "default":
                raise ValueError("only default rotary positions are supported")
        kinds = config.get("layer_types")
        if kinds is not None and (len(kinds) != config["num_hidden_layers"]
                or any(k not in ("full_attention", "sliding_attention") for k in kinds)):
            raise ValueError("invalid ModernBERT layer_types")
    return family


def tokenizer_ids(tokenizer):
    model = tokenizer.get("model", {})
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        ids = set(vocab.values())
    elif isinstance(vocab, list):
        ids = set(range(len(vocab)))
    else:
        raise ValueError("tokenizer.json has no supported vocabulary representation")
    ids.update(token["id"] for token in tokenizer.get("added_tokens", []))
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("invalid tokenizer vocabulary IDs")
    return ids


def expected_parameters(config, decision=None):
    """Exact supported architecture shapes, independent of the CUDA registry."""
    d, f, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    result = {}
    def add(name, *shape):
        result["encoder." + name] = list(shape)
    def norm(name, bias=True):
        add(name + ".weight", d)
        if bias:
            add(name + ".bias", d)
    def linear(name, rows, columns, bias=True):
        add(name + ".weight", rows, columns)
        if bias:
            add(name + ".bias", rows)
    if config["model_type"] == "bert":
        add("embeddings.word_embeddings.weight", config["vocab_size"], d)
        add("embeddings.position_embeddings.weight", config["max_position_embeddings"], d)
        add("embeddings.token_type_embeddings.weight", config.get("type_vocab_size", 2), d)
        norm("embeddings.LayerNorm")
        for layer in range(layers):
            p = f"encoder.layer.{layer}."
            for qkv in ("query", "key", "value"):
                linear(p + "attention.self." + qkv, d, d)
            linear(p + "attention.output.dense", d, d)
            norm(p + "attention.output.LayerNorm")
            linear(p + "intermediate.dense", f, d)
            linear(p + "output.dense", d, f)
            norm(p + "output.LayerNorm")
    else:
        add("embeddings.tok_embeddings.weight", config["vocab_size"], d)
        nb, ab, mb = (config.get(key, False) for key in ("norm_bias", "attention_bias", "mlp_bias"))
        norm("embeddings.norm", nb)
        for layer in range(layers):
            p = f"layers.{layer}."
            if layer:
                norm(p + "attn_norm", nb)
            linear(p + "attn.Wqkv", 3 * d, d, ab)
            linear(p + "attn.Wo", d, d, ab)
            norm(p + "mlp_norm", nb)
            linear(p + "mlp.Wi", 2 * f, d, mb)
            linear(p + "mlp.Wo", d, f, mb)
        norm("final_norm", nb)
    if decision is not None:
        for layer in range(decision.get("head_layers", 2)):
            p = f"head.layers.{layer}."
            result.update({p + name: shape for name, shape in {
                "self_attn.in_proj_weight": [3 * d, d], "self_attn.in_proj_bias": [3 * d],
                "self_attn.out_proj.weight": [d, d], "self_attn.out_proj.bias": [d],
                "linear1.weight": [4 * d, d], "linear1.bias": [4 * d],
                "linear2.weight": [d, 4 * d], "linear2.bias": [d],
                "norm1.weight": [d], "norm1.bias": [d],
                "norm2.weight": [d], "norm2.bias": [d],
            }.items()})
        result.update({"type_emb.weight": [3, d], "scorer.0.weight": [d], "scorer.0.bias": [d],
            "scorer.1.weight": [d, d], "scorer.1.bias": [d], "scorer.3.weight": [1, d],
            "scorer.3.bias": [1], "act_head.0.weight": [256, d + 4], "act_head.0.bias": [256],
            "act_head.2.weight": [len(decision.get("act_costs", {})) + 1, 256],
            "act_head.2.bias": [len(decision.get("act_costs", {})) + 1], "temperature": [3]})
    return result


def import_bundle(source, destination, *, revision=None, tokenizer=None,
                  tokenizer_revision=None, head_layers=2, max_length=512, head_max_length=192):
    source = Source(source, revision)
    decision_path = source.get("rl_agent_config.json", optional=True)
    is_laya = decision_path is not None
    config_path = source.get("encoder/config.json" if is_laya else "config.json")
    config = read_json(config_path)
    family = validate_config(config)
    decision = read_json(decision_path) if is_laya else {
        "head_layers": head_layers, "max_len": max_length, "head_max_len": head_max_length,
        "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0],
    }
    if type(decision.get("head_layers", 2)) is not int or not 0 <= decision.get("head_layers", 2) <= 32:
        raise ValueError("head_layers must be in [0, 32]")
    if any(type(decision.get(key, default)) is not int
           for key, default in (("head_max_len", 192), ("max_len", 512))):
        raise ValueError("decision token budgets must be integers")
    if not 1 <= decision.get("head_max_len", 192) <= decision.get("max_len", 512) <= config["max_position_embeddings"]:
        raise ValueError("invalid decision token budgets")
    temperatures = decision.get("temperature", [1.0, 1.0, 1.0])
    overrides = decision.get("temperature_by_options", {})
    if not isinstance(temperatures, list) or len(temperatures) != 3 or not isinstance(overrides, dict):
        raise ValueError("invalid temperature configuration")
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0
           for t in [*temperatures, *overrides.values()]):
        raise ValueError("temperatures must be finite and positive")
    tok_source = Source(tokenizer, tokenizer_revision) if tokenizer else source
    tok_prefix = "" if tokenizer or not is_laya else "tokenizer/"
    tok_path = tok_source.get(tok_prefix + "tokenizer.json")
    tok_config_path = tok_source.get(tok_prefix + "tokenizer_config.json")
    tok_json = read_json(tok_path)
    if max(tokenizer_ids(tok_json)) >= config["vocab_size"]:
        raise ValueError("tokenizer produces IDs outside the model's embedding vocabulary")
    tok_config = read_json(tok_config_path)
    for key in ("cls_token", "sep_token", "mask_token", "pad_token"):
        if not tok_config.get(key):
            raise ValueError(f"typed decisions require tokenizer_config.json {key}")

    index_path = source.get("model.safetensors.index.json", optional=True)
    if index_path:
        weight_files = sorted(set(read_json(index_path)["weight_map"].values()))
    else:
        weight_files = ["model.safetensors"]
    for name in weight_files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("safetensors index must reference sibling .safetensors files")
    weights = [(name, source.get(name)) for name in weight_files]
    tensors = {}
    for name, path in weights:
        for tensor, info in tensor_header(path).items():
            if tensor in tensors:
                raise ValueError(f"duplicate tensor across shards: {tensor}")
            tensors[tensor] = {**info, "file": name}
    if is_laya:
        prefix = "encoder."
    else:
        anchor = "embeddings.word_embeddings.weight" if family == "bert" else "embeddings.tok_embeddings.weight"
        matches = [p for p in ("", "bert.", "model.") if p + anchor in tensors]
        if len(matches) != 1:
            raise ValueError("cannot identify an unambiguous encoder tensor namespace")
        prefix = matches[0]
    used = set()
    for name, shape in expected_parameters(config, decision if is_laya else None).items():
        source_name = prefix + name[len("encoder."):] if name.startswith("encoder.") else name
        if source_name not in tensors or tensors[source_name]["shape"] != shape:
            raise ValueError(f"missing or incompatible tensor {source_name}: expected shape {shape}")
        used.add(source_name)
    unused = sorted(set(tensors) - used)
    if is_laya and unused:
        raise ValueError(f"unexpected Laya tensors: {unused}")
    manifest = {
        "format": FORMAT, "kind": "laya" if is_laya else "encoder",
        "family": family, "source": source.location, "revision": source.revision,
        "tokenizer_source": tok_source.location, "tokenizer_revision": tok_source.revision,
        "source_encoder_prefix": prefix,
        "unused_source_tensors": unused,
        "encoder_config": "encoder_config.json", "decision_config": "decision_config.json",
        "tokenizer": "tokenizer.json", "tokenizer_config": "tokenizer_config.json",
        "weights": [{"file": name, "sha256": checksum(path)} for name, path in weights],
        "tensors": tensors,
    }
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copyfile(config_path, temporary / "encoder_config.json")
        shutil.copyfile(tok_path, temporary / "tokenizer.json")
        shutil.copyfile(tok_config_path, temporary / "tokenizer_config.json")
        for name, path in weights:
            shutil.copyfile(path, temporary / name)
        for name, value in (("decision_config.json", decision), ("manifest.json", manifest)):
            with open(temporary / name, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local checkpoint directory or Hugging Face model ID")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", help="pin a Hugging Face model revision")
    parser.add_argument("--tokenizer", help="local tokenizer directory or Hugging Face tokenizer ID")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--head-layers", type=int, default=2, help="new decision head depth for encoder-only imports")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--head-max-length", type=int, default=192)
    args = parser.parse_args()
    try:
        manifest = import_bundle(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"import_pretrained: {error}\n")
    print(f"Imported {manifest['kind']} / {manifest['family']} and its tokenizer into {args.destination}")


if __name__ == "__main__":
    main()
