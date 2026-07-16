"""Validate the frozen Base V1 production scheduler and Stage-1 authorization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_SHARD_CHECKSUMS_HASH,
    EXPECTED_TOKENIZED_HASH,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "darkmind_v2" / "config" / "train_base_v1_production_100m.json"
TOKENS_PER_STEP = 8_192
STAGE1_STEPS = 610
STAGE1_TOKENS = 4_997_120
SCHEDULER_STEPS = 12_207
SCHEDULER_TOKENS = 99_999_744


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def learning_rate_for_step(step: int, config: dict[str, Any]) -> float:
    schedule = config["schedule"]
    total_steps = int(schedule["scheduler_horizon_optimizer_steps"])
    warmup_steps = int(schedule["warmup_optimizer_steps"])
    peak = float(schedule["peak_learning_rate"])
    minimum = float(schedule["minimum_learning_rate"])
    if not 1 <= step <= total_steps:
        raise ValueError("learning-rate step is outside the production horizon")
    if step <= warmup_steps:
        return peak * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def classify_stage1(initial_validation: float, final_validation: float, initial_eval: float, final_eval: float) -> dict[str, Any]:
    validation_improvement = (initial_validation - final_validation) / initial_validation * 100.0
    eval_improvement = (initial_eval - final_eval) / initial_eval * 100.0
    minimum = min(validation_improvement, eval_improvement)
    if minimum >= 15.0:
        classification = "Strong PASS"
    elif minimum >= 5.0:
        classification = "Conditional PASS"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "validation_improvement_percent": validation_improvement,
        "eval_improvement_percent": eval_improvement,
    }


def load_and_validate_phase4a_config(path: Path = DEFAULT_CONFIG, *, check_runtime_assets: bool = False) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    _require(config["schema_version"] == "darkmind-v2-base-v1-production-100m-v1", "unsupported Phase 4A config")
    _require(config["initialization_seed"] == config["data"]["data_order_seed"] == 20260712, "seed mismatch")
    data = config["data"]
    _require(data["sequence_length"] == 512, "sequence length changed")
    _require(data["micro_batch_size"] == 2 and data["gradient_accumulation_steps"] == 8, "batch profile changed")
    _require(data["effective_tokens_per_optimizer_step"] == TOKENS_PER_STEP, "tokens per step changed")
    _require(TOKENS_PER_STEP == 512 * 2 * 8, "effective-token arithmetic mismatch")
    _require(all(data[key] is True for key in ("no_data_wrap", "no_replacement_sampling", "no_sequence_repetition")), "data traversal safety disabled")
    optimizer = config["optimizer"]
    _require(optimizer == {"name": "AdamW", "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-08, "weight_decay": 0.1, "gradient_clipping": 1.0}, "optimizer contract changed")
    schedule = config["schedule"]
    _require(schedule["scheduler_horizon_optimizer_steps"] == SCHEDULER_STEPS, "scheduler horizon changed")
    _require(schedule["scheduler_horizon_tokens"] == SCHEDULER_TOKENS == SCHEDULER_STEPS * TOKENS_PER_STEP, "scheduler token horizon changed")
    _require(schedule["warmup_optimizer_steps"] == 100, "warmup changed")
    _require(schedule["peak_learning_rate"] == 0.0003 and schedule["minimum_learning_rate"] == 0.00003, "learning-rate bounds changed")
    _require(config["stage_gates"] == {
        "5m": {"authorized": True, "optimizer_steps": 610, "tokens": 4_997_120},
        "25m": {"authorized": False, "optimizer_steps": 3_051, "tokens": 24_993_792},
        "100m": {"authorized": False, "optimizer_steps": 12_207, "tokens": 99_999_744},
    }, "stage gates changed")
    _require(config["authorization"]["maximum_optimizer_steps"] == STAGE1_STEPS, "Stage-1 step authorization changed")
    _require(config["authorization"]["maximum_training_tokens"] == STAGE1_TOKENS == STAGE1_STEPS * TOKENS_PER_STEP, "Stage-1 token authorization changed")
    _require(config["evaluation"]["milestone_steps"] == [0, 128, 305, 458, 610], "milestones changed")
    _require(config["precision"] == "bf16" and config["attention_implementation"] == "sdpa", "numerical profile changed")
    _require(config["gradient_checkpointing"] is False, "gradient checkpointing must be disabled")
    _require(config["training_compile"] is False and config["fused_optimizer"] is False, "unproven optimization enabled")
    _require(config["vocabulary_size"] == 24_000 and config["safetensors_weights"] is True, "vocabulary or weight format changed")
    _require(config["corpus"]["corpus_hash"] == EXPECTED_CORPUS_HASH, "Corpus V3 hash changed")
    _require(config["corpus"]["tokenized_manifest_hash"] == EXPECTED_TOKENIZED_HASH, "tokenized manifest hash changed")
    _require(config["corpus"]["shard_checksums_hash"] == EXPECTED_SHARD_CHECKSUMS_HASH, "shard checksum hash changed")
    _require(config["model_config_sha256"] == EXPECTED_CONFIG_SHA256, "Base V1 config hash reference changed")
    model_path = ROOT / config["model_config"]
    _require(sha256_file(model_path) == EXPECTED_CONFIG_SHA256, "Base V1 config file changed")
    model_config = DarkMindV2Config.from_json_file(model_path)
    _require(model_config_hash(model_config) == EXPECTED_ARCHITECTURE_HASH, "Base V1 architecture changed")
    _require(learning_rate_for_step(STAGE1_STEPS, config) > schedule["minimum_learning_rate"], "scheduler reaches minimum at 5M")
    _require(config["corpus"]["train_complete_sequence_tokens"] >= config["stage_gates"]["25m"]["tokens"], "25M exceeds complete train sequences")
    _require(config["corpus"]["train_complete_sequence_tokens"] < SCHEDULER_TOKENS, "100M data limitation must remain explicit")
    if check_runtime_assets:
        tokenized_dir = ROOT / config["corpus"]["tokenized_dir"]
        manifest = json.loads((tokenized_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
        _require(manifest["deterministic_content_hash"] == EXPECTED_TOKENIZED_HASH, "runtime Corpus V3 manifest changed")
        _require(manifest["statistics"]["split_tokens"]["train"] == config["corpus"]["train_tokens"], "runtime train tokens changed")
    return config
