#!/usr/bin/env python3
"""Convert a decisions SnakePolicy .pt file to a native named FP32 checkpoint.

This optional offline tool requires PyTorch. The CUDA training executable does
not load Python, PyTorch, or a sibling repository at runtime.
"""

import argparse
from array import array
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile

import torch


MAGIC = b"PUFDT01\0"


def normalize_config(config):
    if not isinstance(config, dict):
        raise ValueError("policy_config must be a dictionary")
    unknown = set(config) - {"width", "layers", "pooling", "coordinates", "body_features"}
    if unknown:
        raise ValueError(f"unsupported policy_config keys: {sorted(unknown)}")
    width, layers = config.get("width"), config.get("layers")
    if type(width) is not int or not 4 <= width <= 4096 or width % 4:
        raise ValueError("width must be a multiple of four in [4, 4096]")
    if type(layers) is not int or not 1 <= layers <= 128:
        raise ValueError("layers must be an integer in [1, 128]")
    pooling = config.get("pooling", "cls")
    coordinates = config.get("coordinates", "absolute")
    if pooling not in ("cls", "head"):
        raise ValueError("pooling must be cls or head")
    if coordinates not in ("absolute", "head-relative"):
        raise ValueError("coordinates must be absolute or head-relative")
    if config.get("body_features", "ordered") != "ordered":
        raise ValueError("native checkpoints currently support only ordered body features")
    return {"width": width, "layers": layers, "pooling": pooling,
            "coordinates": coordinates, "body_features": "ordered"}


def parameter_specs(config):
    """Ordered names/shapes in decision::Model's native parameter registry."""
    width = config["width"]
    specs = [("cls", (1, 1, width)), ("positions", (1, 101, width)),
             ("cells.weight", (102, width))]
    for layer in range(config["layers"]):
        prefix = f"encoder.layers.{layer}."
        specs.extend((prefix + name, shape) for name, shape in (
            ("self_attn.in_proj_weight", (3 * width, width)),
            ("self_attn.in_proj_bias", (3 * width,)),
            ("self_attn.out_proj.weight", (width, width)),
            ("self_attn.out_proj.bias", (width,)),
            ("linear1.weight", (4 * width, width)), ("linear1.bias", (4 * width,)),
            ("linear2.weight", (width, 4 * width)), ("linear2.bias", (width,)),
            ("norm1.weight", (width,)), ("norm1.bias", (width,)),
            ("norm2.weight", (width,)), ("norm2.bias", (width,)),
        ))
    specs.extend((name, shape) for name, shape in (
        ("encoder.norm.weight", (width,)), ("encoder.norm.bias", (width,)),
        ("action_head.weight", (4, width)), ("action_head.bias", (4,)),
        ("value_head.weight", (1, width)), ("value_head.bias", (1,)),
    ))
    if config["coordinates"] == "head-relative":
        specs.extend([(name + ".weight", (19, width))
                      for name in ("relative_rows", "relative_columns")])
    return specs


def write_checkpoint(path, state_dict, config):
    """Validate and atomically write named tensors; linear weights are [out, in]."""
    config = normalize_config(config)
    specs = parameter_specs(config)
    if not isinstance(state_dict, dict):
        raise ValueError("policy_state must be a state_dict")
    expected = {name for name, _ in specs}
    missing, extra = expected - set(state_dict), set(state_dict) - expected
    if missing or extra:
        raise ValueError(f"parameter mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    tensors = []
    for name, shape in specs:
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name}: expected a dense tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, found {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"{name}: expected floating-point weights")
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}: nonfinite FP32 weight")
        tensors.append(tensor)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(MAGIC)
            output.write(struct.pack("<5I", config["width"], config["layers"],
                                     config["pooling"] == "head",
                                     config["coordinates"] == "head-relative", len(specs)))
            for (name, shape), tensor in zip(specs, tensors):
                encoded = name.encode("utf-8")
                output.write(struct.pack("<I", len(encoded)))
                output.write(encoded)
                output.write(struct.pack("<I", len(shape)))
                output.write(struct.pack("<" + "I" * len(shape), *shape))
                output.write(struct.pack("<Q", tensor.numel()))
                flat = tensor.view(-1)
                # array avoids a NumPy dependency and keeps conversion memory bounded.
                for start in range(0, flat.numel(), 65536):
                    values = array("f", flat[start:start + 65536].tolist())
                    if values.itemsize != 4:
                        raise RuntimeError("this platform's float array is not IEEE float32")
                    if sys.byteorder != "little":
                        values.byteswap()
                    output.write(values.tobytes())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="decisions SnakePolicy .pt checkpoint")
    parser.add_argument("destination", type=Path, help="native model checkpoint (e.g. policy.pufdt)")
    args = parser.parse_args()
    try:
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"policy_config", "policy_state"} <= set(checkpoint):
            raise ValueError("expected checkpoint fields policy_config and policy_state")
        config = write_checkpoint(args.destination, checkpoint["policy_state"], checkpoint["policy_config"])
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
        parser.exit(1, f"export_decision: {error}\n")
    print(f"Exported {args.destination}: width={config['width']} layers={config['layers']} "
          f"pooling={config['pooling']} coordinates={config['coordinates']} (model weights only)")


if __name__ == "__main__":
    main()
