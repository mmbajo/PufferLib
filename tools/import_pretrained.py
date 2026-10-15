#!/usr/bin/env python3
"""Import a Laya or Hugging Face BERT/ModernBERT checkpoint and its tokenizer.

Weights retain the safetensors representation; native loading converts FP16,
BF16 or FP32 to FP32. Python is an offline packaging dependency only. Remote
sources require huggingface_hub; local safetensors directories need stdlib only.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import re
import struct
import tempfile


FORMAT = "puffer-pretrained-v1"


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def tensor_names(path):
    """Read names only; Bundle validates tensor metadata and model compatibility."""
    with open(path, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= min(32 << 20, path.stat().st_size - 8):
            raise ValueError(f"{path}: invalid safetensors header length")
        return set(json.loads(stream.read(length))) - {"__metadata__"}


class Source:
    def __init__(self, location, revision=None):
        self.location, self.revision = str(location), revision
        self.local = Path(location) if Path(location).is_dir() else None
        if self.local is None and not (revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise ValueError("remote imports require huggingface_hub") from error
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
            raise ValueError("remote imports require huggingface_hub") from error
        try:
            return Path(hf_hub_download(self.location, name, revision=self.revision))
        except EntryNotFoundError:
            if optional:
                return None
            raise


def import_bundle(source, destination, *, revision=None, tokenizer=None,
                  tokenizer_revision=None, head_layers=2, max_length=512, head_max_length=192):
    source = Source(source, revision)
    decision_path = source.get("rl_agent_config.json", optional=True)
    is_laya = decision_path is not None
    config_path = source.get("encoder/config.json" if is_laya else "config.json")
    config = read_json(config_path)
    family = config["model_type"]
    decision = read_json(decision_path) if is_laya else {
        "head_layers": head_layers, "max_len": max_length, "head_max_len": head_max_length,
        "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0],
    }
    tok_source = Source(tokenizer, tokenizer_revision) if tokenizer else source
    tok_prefix = "" if tokenizer or not is_laya else "tokenizer/"
    tok_path = tok_source.get(tok_prefix + "tokenizer.json")
    tok_config_path = tok_source.get(tok_prefix + "tokenizer_config.json")

    index_path = source.get("model.safetensors.index.json", optional=True)
    if index_path:
        weight_files = sorted(set(read_json(index_path)["weight_map"].values()))
    else:
        weight_files = ["model.safetensors"]
    for name in weight_files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("safetensors index must reference sibling .safetensors files")
    weights = [(name, source.get(name)) for name in weight_files]
    if is_laya:
        prefix = "encoder."
    else:
        names = set().union(*(tensor_names(path) for _, path in weights))
        anchors = ("embeddings.word_embeddings.weight", "embeddings.tok_embeddings.weight")
        matches = [p for p in ("", "bert.", "model.")
                   if any(p + anchor in names for anchor in anchors)]
        if len(matches) != 1:
            raise ValueError("cannot identify an unambiguous encoder tensor namespace")
        prefix = matches[0]
    manifest = {
        "format": FORMAT, "kind": "laya" if is_laya else "encoder",
        "family": family, "source": source.location, "revision": source.revision,
        "tokenizer_source": tok_source.location, "tokenizer_revision": tok_source.revision,
        "source_encoder_prefix": prefix,
        "encoder_config": "encoder_config.json", "decision_config": "decision_config.json",
        "tokenizer": "tokenizer.json", "tokenizer_config": "tokenizer_config.json",
        "weights": [{"file": name} for name, path in weights],
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
    parser.add_argument("--tokenizer", help="local tokenizer directory or Hub tokenizer ID")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--head-layers", type=int, default=2,
                        help="new decision head depth for encoder-only imports")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--head-max-length", type=int, default=192)
    args = parser.parse_args()
    try:
        manifest = import_bundle(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"import_pretrained: {error}\n")
    print(f"Imported {manifest['kind']} / {manifest['family']} "
          f"and its tokenizer into {args.destination}")


if __name__ == "__main__":
    main()
