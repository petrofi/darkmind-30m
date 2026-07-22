"""Pure validation and decision helpers for the Phase 5A planning artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


AUTOMATIC_METRIC_FIELDS = {
    "generation_count",
    "repetition_rate",
    "exact_loop_rate",
    "eos_completion_rate",
    "empty_output_rate",
    "output_tokens",
    "mean_unique_token_ratio",
    "language_id_consistency_rate",
    "language_switch_error_rate",
    "unicode_health",
    "special_token_leakage_count",
    "mean_prompt_token_overlap_ratio",
    "code_structure_valid_rate",
    "punctuation_completion_rate",
    "human_semantic_quality_claimed",
}


def validate_automatic_metric_schema(metrics: dict[str, Any]) -> None:
    missing = AUTOMATIC_METRIC_FIELDS - set(metrics)
    if missing:
        raise ValueError(f"automatic metrics missing fields: {sorted(missing)}")
    if not isinstance(metrics["generation_count"], int) or metrics["generation_count"] <= 0:
        raise ValueError("automatic metrics must cover at least one record")
    if metrics["human_semantic_quality_claimed"] is not False:
        raise ValueError("automatic proxies must not claim human semantic quality")


def validate_corpus_v4_targets(payload: dict[str, Any]) -> None:
    if payload.get("planning_only") is not True or payload.get("acquisition_authorized") is not False:
        raise ValueError("Corpus V4 targets must remain planning-only")
    tranche = payload["tranche_2"]
    categories = tranche["categories"]
    total = sum(item["tokens"] for item in categories.values())
    if total != tranche["target_unique_tokens"]:
        raise ValueError("Corpus V4 category totals do not match the tranche target")
    shares = sum(item["share"] for item in categories.values())
    if abs(shares - 1.0) > 1e-9:
        raise ValueError("Corpus V4 category shares must total one")
    for name, item in categories.items():
        low, high = item["allowed_share_range"]
        if not low <= item["share"] <= high:
            raise ValueError(f"Corpus V4 category is outside its range: {name}")
    projected = payload["current_corpus_v3"]["total_unique_tokens"] + tranche["target_unique_tokens"]
    if projected != tranche["projected_cumulative_unique_tokens"]:
        raise ValueError("Corpus V4 projected cumulative tokens are inconsistent")
    if payload["hard_rules"].get("category_double_counting") is not False:
        raise ValueError("category double counting must be forbidden")


def validate_continuation_policies(payload: dict[str, Any]) -> None:
    if payload.get("planning_only") is not True or payload.get("execution_authorized") is not False:
        raise ValueError("continuation policies must remain planning-only")
    comparison = payload["controlled_comparison"]
    required_controls = {
        "same_new_data_slice",
        "same_model_checkpoint",
        "same_sequence_order",
        "same_token_budget",
        "same_evaluation_probes",
        "one_factor_at_a_time",
    }
    if any(comparison.get(field) is not True for field in required_controls):
        raise ValueError("continuation comparison controls are incomplete")
    if comparison.get("training_authorized") is not False:
        raise ValueError("Phase 5A must not authorize training")
    candidates = {item["id"]: item for item in payload["candidates"]}
    if set(candidates) != {"P1", "P2", "P3"}:
        raise ValueError("continuation policies must define P1, P2, and P3")
    if candidates["P1"]["rewarm_steps"] != 0 or candidates["P1"]["optimizer_moments"] != "preserve":
        raise ValueError("P1 must preserve moments and avoid rewarm")
    if candidates["P2"]["maximum_learning_rate"] > 5e-5:
        raise ValueError("P2 maximum learning rate exceeds the approved ceiling")
    if candidates["P3"]["optimizer_reset"] != "not_approved_by_default":
        raise ValueError("P3 must not approve an optimizer reset by default")
    if any(item["abrupt_learning_rate_jump"] is not False for item in candidates.values()):
        raise ValueError("abrupt learning-rate jumps are forbidden")


def classify_public_release(
    greedy_repetition_rate: float,
    greedy_exact_loop_rate: float,
    eos_completion_rate: float,
    model_weight_license_finalized: bool,
) -> dict[str, Any]:
    blockers = []
    if greedy_repetition_rate > 0.20:
        blockers.append("high_greedy_repetition")
    if greedy_exact_loop_rate > 0.10:
        blockers.append("high_exact_loop_rate")
    if eos_completion_rate < 0.50:
        blockers.append("low_eos_completion")
    if not model_weight_license_finalized:
        blockers.append("model_weight_license_unresolved")
    return {
        "technically_uploadable": True,
        "publicly_advisable": not blockers,
        "blockers": blockers,
    }


def validate_archival_inventory(entries: Iterable[dict[str, Any]]) -> None:
    required = {
        "absolute_path",
        "size_bytes",
        "purpose",
        "required_for_reproducibility",
        "required_for_future_training",
        "safe_to_archive",
        "safe_to_delete_only_after_verified_archive",
        "must_remain_local",
        "depends_on",
    }
    materialized = list(entries)
    if not materialized:
        raise ValueError("archival inventory cannot be empty")
    for entry in materialized:
        if set(entry) != required:
            raise ValueError("archival inventory schema mismatch")
        if not str(entry["absolute_path"]).startswith(("C:\\", "D:\\")):
            raise ValueError("archival inventory paths must be absolute")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
            raise ValueError("archival inventory sizes must be non-negative integers")


def assert_nondestructive_archival_plan(text: str) -> None:
    normalized = text.lower()
    forbidden = (
        "remove-item ",
        "git clean",
        "git reset",
        "rmdir ",
        "del /",
        "move-item ",
    )
    found = [item.strip() for item in forbidden if item in normalized]
    if found:
        raise ValueError(f"destructive archival instruction found: {found}")


def choose_next_phase(
    learning_remained_positive: bool,
    base_quality_stable: bool,
    capacity_saturation_evidence: bool,
    legally_usable_unique_data_available: bool,
) -> str:
    if capacity_saturation_evidence:
        return "architecture_capacity_review"
    if learning_remained_positive and legally_usable_unique_data_available:
        return "expand_unique_corpus"
    if base_quality_stable:
        return "controlled_sft_preparation"
    return "architecture_capacity_review"
