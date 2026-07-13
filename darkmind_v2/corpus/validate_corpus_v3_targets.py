"""Fail-closed validation for the Phase 3 corpus target contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "darkmind-v2-corpus-v3-targets-v1"
REQUIRED_STAGES = [5_000_000, 25_000_000, 100_000_000, 250_000_000, 500_000_000]
REQUIRED_CATEGORIES = {
    "turkish_high_quality_prose": (55, 65),
    "english_high_quality_prose": (20, 30),
    "technical_and_educational": (8, 12),
    "code_and_structured_text": (3, 7),
    "controlled_bilingual_or_dialogue": (0, 5),
}


def validate_corpus_v3_targets(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus v3 target schema")
    if payload.get("default_target_tokens") != 500_000_000:
        raise ValueError("default Phase 3 corpus target must be 500M tokens")
    if payload.get("training_stages_tokens") != REQUIRED_STAGES:
        raise ValueError("training stages must be exactly 5M, 25M, 100M, 250M, and 500M")
    composition = payload.get("exclusive_composition")
    if not isinstance(composition, dict) or set(composition) != set(REQUIRED_CATEGORIES):
        raise ValueError("exclusive composition categories are incomplete")
    total = 0
    for name, expected_range in REQUIRED_CATEGORIES.items():
        category = composition[name]
        target = category.get("target_percent")
        allowed = category.get("allowed_percent")
        if allowed != list(expected_range) or not isinstance(target, int):
            raise ValueError(f"invalid composition contract for {name}")
        if not allowed[0] <= target <= allowed[1]:
            raise ValueError(f"target outside allowed range for {name}")
        total += target
    if total != 100:
        raise ValueError("exclusive composition targets must sum to 100 percent")
    if payload.get("tokenizer") != "darkmind_v2_sp_bpe24k_v1":
        raise ValueError("corpus v3 must use the frozen tokenizer")
    if payload.get("token_storage_dtype") != "uint16":
        raise ValueError("24k token IDs must be stored as uint16")
    if payload.get("maximum_single_source_percent", 101) > 35:
        raise ValueError("single-source cap may not exceed 35 percent")
    split = payload.get("split_policy", {})
    if sum(split.get(key, 0) for key in ("train_percent", "validation_percent", "eval_percent")) != 100:
        raise ValueError("corpus splits must sum to 100 percent")
    return {"result": "PASS", "target_tokens": 500_000_000, "composition_percent": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_corpus_v3_targets(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
