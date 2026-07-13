"""Fail-closed validation for the Phase 2C full-epoch contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "darkmind-v2-full-epoch-training-config-v1"
EXPECTED_MILESTONES = [0, 256, 717, 1434, 2150, 2867]


def learning_rate_for_step(step: int, config: dict[str, Any]) -> float:
    """Return the LR used by a one-indexed optimizer step."""
    if not 1 <= step <= config["maximum_optimizer_steps"]:
        raise ValueError("optimizer step is outside the planned run")
    peak = config["optimizer"]["peak_learning_rate"]
    minimum = config["schedule"]["minimum_learning_rate"]
    warmup = config["schedule"]["warmup_optimizer_steps"]
    total = config["maximum_optimizer_steps"]
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_and_validate_full_epoch_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported full-epoch config schema")
    if config.get("initialization_seed") != 20260712:
        raise ValueError("initialization seed changed")
    if config.get("device") != "cuda" or config.get("mixed_precision") != "bf16":
        raise ValueError("Phase 2C requires CUDA BF16")
    if config.get("vocabulary_size") != 24000 or config.get("tied_embeddings_required") is not True:
        raise ValueError("model vocabulary or tied-embedding contract changed")
    if config.get("maximum_optimizer_steps") != 2867:
        raise ValueError("full-epoch optimizer-step target changed")
    if config.get("maximum_total_training_tokens") != 11_743_232:
        raise ValueError("full-epoch consumed-token target changed")
    if config.get("segment_a_target_step") != 1434 or config.get("segment_b_target_step") != 2867:
        raise ValueError("forced-resume boundaries changed")

    data = config.get("data", {})
    expected_data = {
        "data_order_seed": 20260712,
        "sequence_length": 256,
        "effective_tokens_per_optimizer_step": 4096,
        "train_corpus_tokens": 11_744_226,
        "excluded_tail_tokens": 994,
        "deterministic_traversal": "contiguous_no_shuffle",
        "no_data_repetition": True,
        "no_sample_replacement": True,
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("full-epoch data contract changed")
    if config["maximum_optimizer_steps"] * data["effective_tokens_per_optimizer_step"] != config["maximum_total_training_tokens"]:
        raise ValueError("optimizer-step/token arithmetic mismatch")
    if data["train_corpus_tokens"] - config["maximum_total_training_tokens"] != data["excluded_tail_tokens"]:
        raise ValueError("excluded-tail arithmetic mismatch")

    optimizer = config.get("optimizer", {})
    expected_optimizer = {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "peak_learning_rate": 0.0003,
    }
    if any(optimizer.get(key) != value for key, value in expected_optimizer.items()):
        raise ValueError("optimizer contract changed")
    schedule = config.get("schedule", {})
    if schedule != {
        "minimum_learning_rate": 0.00003,
        "name": "warmup_cosine",
        "planned_optimizer_steps": 2867,
        "warmup_optimizer_steps": 100,
    }:
        raise ValueError("scheduler contract changed")
    if learning_rate_for_step(100, config) != optimizer["peak_learning_rate"]:
        raise ValueError("warmup does not reach its peak at step 100")
    if not math.isclose(learning_rate_for_step(2867, config), schedule["minimum_learning_rate"], abs_tol=1e-15):
        raise ValueError("scheduler does not reach its minimum at the final step")

    profiles = config.get("profiles", [])
    actual_profiles = [
        (item.get("micro_batch_size"), item.get("gradient_accumulation_steps"))
        for item in profiles
    ]
    if actual_profiles != [(4, 4), (2, 8), (1, 16)]:
        raise ValueError("approved calibration profiles changed")
    for micro_batch, accumulation in actual_profiles:
        if micro_batch * accumulation * data["sequence_length"] != 4096:
            raise ValueError("profile does not preserve the 4,096-token optimizer batch")
    if config.get("gradient_clipping") != 1.0:
        raise ValueError("gradient clipping changed")
    if config.get("compile", {}).get("enabled") is not False:
        raise ValueError("compile mode must remain disabled")
    if config.get("checkpointing", {}).get("milestone_steps") != EXPECTED_MILESTONES:
        raise ValueError("checkpoint milestones changed")
    if config.get("evaluation", {}).get("milestone_steps") != EXPECTED_MILESTONES:
        raise ValueError("evaluation milestones changed")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_and_validate_full_epoch_config(args.config)
    print(json.dumps({
        "result": "PASS",
        "optimizer_steps": config["maximum_optimizer_steps"],
        "consumed_tokens": config["maximum_total_training_tokens"],
        "coverage_percent": 100 * config["maximum_total_training_tokens"] / config["data"]["train_corpus_tokens"],
        "warmup_peak_learning_rate": learning_rate_for_step(100, config),
        "final_learning_rate": learning_rate_for_step(2867, config),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
