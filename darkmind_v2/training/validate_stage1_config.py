"""Validate the bounded Phase 2B.1 Stage-1 training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "darkmind-v2-stage1-training-config-v1"


def load_and_validate_stage1_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Stage-1 config schema")
    if config.get("seed") != 20260712:
        raise ValueError("Stage-1 seed must be 20260712")
    if config.get("mixed_precision") != "bf16" or config.get("device") != "cuda":
        raise ValueError("Stage-1 requires CUDA BF16")
    if config.get("tied_embeddings_required") is not True:
        raise ValueError("Stage-1 requires tied embeddings")
    if config.get("maximum_total_training_tokens") != 1_048_576:
        raise ValueError("Stage-1 token ceiling changed")
    if config.get("segment_a_target_tokens") != 524_288:
        raise ValueError("Segment A target changed")
    if config.get("segment_b_target_tokens") != 1_048_576:
        raise ValueError("Segment B target changed")
    data = config.get("data", {})
    if data.get("sequence_length") != 256:
        raise ValueError("Stage-1 sequence length must remain 256")
    if data.get("effective_tokens_per_optimizer_step") != 4096:
        raise ValueError("Stage-1 effective token batch must remain 4096")
    optimizer = config.get("optimizer", {})
    expected_optimizer = {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "peak_learning_rate": 0.0003,
    }
    if any(optimizer.get(key) != value for key, value in expected_optimizer.items()):
        raise ValueError("Stage-1 optimizer contract changed")
    schedule = config.get("schedule", {})
    if schedule.get("name") != "warmup_cosine":
        raise ValueError("Stage-1 schedule must be warmup_cosine")
    if schedule.get("warmup_optimizer_steps") != 20:
        raise ValueError("Stage-1 warmup must be 20 optimizer steps")
    if schedule.get("minimum_learning_rate") != 0.00003:
        raise ValueError("Stage-1 minimum learning rate changed")
    profiles = config.get("profiles", [])
    expected_profiles = [(4, 4), (2, 8), (1, 16)]
    actual_profiles = [
        (item.get("micro_batch_size"), item.get("gradient_accumulation_steps"))
        for item in profiles
    ]
    if actual_profiles != expected_profiles:
        raise ValueError("Stage-1 calibration profiles changed")
    for micro_batch, accumulation in actual_profiles:
        if micro_batch * accumulation * data["sequence_length"] != 4096:
            raise ValueError("profile does not preserve the 4,096-token optimizer batch")
    if config.get("compile", {}).get("enabled") is not False:
        raise ValueError("compile mode must remain disabled")
    if 1_048_576 // 4096 != 256 or 524_288 // 4096 != 128:
        raise AssertionError("Stage-1 optimizer-step arithmetic changed")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_and_validate_stage1_config(args.config)
    print(
        json.dumps(
            {
                "result": "PASS",
                "total_optimizer_steps": config["maximum_total_training_tokens"]
                // config["data"]["effective_tokens_per_optimizer_step"],
                "segment_a_optimizer_steps": config["segment_a_target_tokens"]
                // config["data"]["effective_tokens_per_optimizer_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
