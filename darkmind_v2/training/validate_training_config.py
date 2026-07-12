"""Fail-closed validation for tiny-smoke training configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "darkmind-v2-training-config-v1"


def load_and_validate_training_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training config schema")
    if payload.get("tied_embeddings_required") is not True:
        raise ValueError("training config must require tied embeddings")
    if payload.get("mixed_precision") not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("unsupported mixed_precision setting")
    optimizer = payload.get("optimizer", {})
    if optimizer.get("name") != "AdamW":
        raise ValueError("only AdamW is supported")
    if (optimizer.get("beta1"), optimizer.get("beta2")) != (0.9, 0.95):
        raise ValueError("AdamW betas must be 0.9 and 0.95")
    if optimizer.get("weight_decay") != 0.1:
        raise ValueError("weight decay must be 0.1")
    if payload.get("gradient_clipping") != 1.0:
        raise ValueError("gradient clipping must be 1.0")
    fixture = payload.get("fixture_smoke", {})
    for key in ("max_steps", "micro_batch_size", "gradient_accumulation_steps", "sequence_length"):
        if not isinstance(fixture.get(key), int) or fixture[key] <= 0:
            raise ValueError(f"fixture_smoke.{key} must be a positive integer")
    if fixture["sequence_length"] > payload["data"]["sequence_length"]:
        raise ValueError("fixture sequence length exceeds the model training sequence length")
    if payload.get("planned_full_run", {}).get("requires_benchmark_approval") is not True:
        raise ValueError("full-run parameters must remain approval-gated")
    if payload["planned_full_run"].get("learning_rate") is not None:
        raise ValueError("full-run learning rate must not be guessed during Phase 2A")
    if payload["planned_full_run"].get("micro_batch_size") is not None:
        raise ValueError("full-run batch size must not be guessed during Phase 2A")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_and_validate_training_config(args.config)
    print(json.dumps({"result": "PASS", "schema_version": config["schema_version"]}, indent=2))


if __name__ == "__main__":
    main()
