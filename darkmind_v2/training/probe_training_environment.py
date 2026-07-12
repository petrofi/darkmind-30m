"""Report local training capabilities without starting a training run."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch


RELEVANT_DEPENDENCIES = ("torch", "transformers", "sentencepiece", "safetensors", "numpy", "pytest")


def dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment(path: Path = Path(".")) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if cuda else None
    adamw_parameters = inspect.signature(torch.optim.AdamW).parameters
    fused_adamw = "fused" in adamw_parameters and cuda
    disk = shutil.disk_usage(path)
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_memory_bytes": int(properties.total_memory) if properties else 0,
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "fp16_supported": bool(cuda),
        "fused_optimizer_options": {
            "adamw_fused_argument_available": "fused" in adamw_parameters,
            "adamw_fused_currently_usable": bool(fused_adamw),
        },
        "cpu_threads": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "disk_path": str(path.resolve()),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "dependencies": {name: dependency_version(name) for name in RELEVANT_DEPENDENCIES},
        "transformers_importable": importlib.util.find_spec("transformers") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe_environment(args.path)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
