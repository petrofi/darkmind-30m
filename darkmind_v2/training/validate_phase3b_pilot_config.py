"""Fail-closed validation for the equal-token Phase 3B finalist contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "darkmind-v2-phase3b-finalist-pilot-v1"
EXPECTED_TOKENIZED_MANIFEST_HASH = "23e92169ae6ef3b0b0f11c4d0ca327ef60d59b0d8b697a1c4261218a233cce28"
PILOT_STEPS = 610
PILOT_TOKENS = 4_997_120
CHECKPOINT_STEPS = [0, 152, 305, 458, 610]


def load_and_validate_phase3b_pilot_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Phase 3B pilot schema")
    if config.get("initialization_seed") != 20260712:
        raise ValueError("Phase 3B initialization seed changed")
    if config.get("precision") != "bf16":
        raise ValueError("Phase 3B requires BF16")
    if config.get("attention_implementation") != "sdpa" or config.get("gradient_checkpointing") is not False:
        raise ValueError("Phase 3B safe attention/checkpointing profile changed")
    if config.get("vocabulary_size") != 24000 or config.get("tied_embeddings_required") is not True:
        raise ValueError("tokenizer/model compatibility contract changed")
    if config.get("candidates") != {
        "C": "darkmind_v2/config/model_base_candidate_c_100m.json",
        "D": "darkmind_v2/config/model_base_candidate_d_120m.json",
    }:
        raise ValueError("Phase 3B finalists changed")

    data = config.get("data", {})
    expected_data = {
        "data_order_seed": 20260712,
        "deterministic_traversal": "contiguous_no_shuffle",
        "effective_tokens_per_optimizer_step": 8192,
        "gradient_accumulation_steps": 8,
        "micro_batch_size": 2,
        "no_data_repetition": True,
        "no_sample_replacement": True,
        "sequence_length": 512,
        "tokenized_dir": "darkmind_v2/data/phase2a/tokenized/full_v1",
        "train_corpus_tokens": 11_744_226,
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("Phase 3B equal-data contract changed")
    if data["micro_batch_size"] * data["gradient_accumulation_steps"] * data["sequence_length"] != 8192:
        raise ValueError("effective optimizer batch arithmetic mismatch")

    calibration = config.get("calibration", {})
    if calibration.get("learning_rates") != [0.0001, 0.0002, 0.0003]:
        raise ValueError("learning-rate candidate set changed")
    if calibration.get("optimizer_steps") != 64 or calibration.get("tokens") != 524_288:
        raise ValueError("calibration token budget changed")
    if calibration["optimizer_steps"] * data["effective_tokens_per_optimizer_step"] != calibration["tokens"]:
        raise ValueError("calibration token arithmetic mismatch")

    pilot = config.get("pilot", {})
    if pilot.get("maximum_optimizer_steps") != PILOT_STEPS:
        raise ValueError("pilot optimizer-step target changed")
    if pilot.get("maximum_total_training_tokens") != PILOT_TOKENS:
        raise ValueError("pilot token target changed")
    if pilot.get("checkpoint_steps") != CHECKPOINT_STEPS:
        raise ValueError("pilot checkpoint milestones changed")
    if pilot.get("segment_a_target_step") != 305 or pilot.get("segment_b_target_step") != 610:
        raise ValueError("forced process-restart boundary changed")
    if PILOT_STEPS * data["effective_tokens_per_optimizer_step"] != PILOT_TOKENS:
        raise ValueError("pilot step/token arithmetic mismatch")
    if not PILOT_TOKENS <= pilot.get("requested_token_ceiling", 0) < PILOT_TOKENS + 8192:
        raise ValueError("pilot target is not the largest full effective batch at or below the ceiling")
    if PILOT_TOKENS > data["train_corpus_tokens"]:
        raise ValueError("pilot would wrap or repeat training data")

    optimizer = config.get("optimizer", {})
    if optimizer != {
        "beta1": 0.9,
        "beta2": 0.95,
        "gradient_clipping": 1.0,
        "name": "AdamW",
        "weight_decay": 0.1,
    }:
        raise ValueError("Phase 3B optimizer contract changed")
    if config.get("schedule") != {
        "minimum_learning_rate_ratio": 0.1,
        "name": "warmup_cosine",
        "warmup_optimizer_steps": 20,
    }:
        raise ValueError("Phase 3B pilot schedule changed")
    if config.get("materiality") != {
        "final_validation_relative_advantage_percent": 2.0,
        "score_tie_percent": 2.0,
    }:
        raise ValueError("predeclared Phase 3B decision materiality changed")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("darkmind_v2/config/phase3b_finalist_pilot.json"),
    )
    args = parser.parse_args()
    config = load_and_validate_phase3b_pilot_config(args.config)
    print(json.dumps({
        "result": "PASS",
        "pilot_optimizer_steps": config["pilot"]["maximum_optimizer_steps"],
        "pilot_tokens_per_candidate": config["pilot"]["maximum_total_training_tokens"],
        "calibration_tokens_per_learning_rate": config["calibration"]["tokens"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
