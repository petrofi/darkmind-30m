"""Evidence scoring and schema checks for the Phase 3B architecture freeze."""

from __future__ import annotations

from typing import Any


WEIGHTS = {
    "validation_improvement": 0.25,
    "eval_improvement": 0.15,
    "loss_improvement_per_hour": 0.10,
    "transformer_body_capacity": 0.10,
    "generation_health_trend": 0.10,
    "long_run_stability": 0.10,
    "vram_headroom": 0.10,
    "checkpoint_resume_reliability": 0.05,
    "implementation_backend_reliability": 0.05,
}
MATERIAL_VALIDATION_ADVANTAGE = 0.02
TIE_BAND = 0.02


def generation_health(record: dict[str, Any]) -> float:
    """Return a bounded greedy/sampling health composite in [0, 1]."""
    components: list[float] = []
    for mode in ("greedy", "sampling"):
        metrics = record[mode]
        generations = int(metrics["generations"])
        meaningful = int(record[f"{mode}_subsets"]["meaningful_proxy_total"])
        components.extend(
            [
                1.0 - int(metrics["quality_warning_counts"].get("repetition", 0)) / generations,
                1.0 - int(metrics["exact_repeated_ngram_loop_outputs"]) / generations,
                float(metrics["mean_unique_token_ratio"]),
                float(metrics["eos_completion_rate"]),
                meaningful / generations,
            ]
        )
    return sum(components) / len(components)


def _ratio_scores(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    if maximum <= 0:
        return {candidate: 100.0 for candidate in values}
    return {candidate: value / maximum * 100.0 for candidate, value in values.items()}


def score_finalists(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score finalists after applying hard eligibility gates."""
    required = {"C", "D"}
    if set(evidence) != required:
        raise ValueError("finalist evidence must contain exactly Candidates C and D")

    raw: dict[str, dict[str, float]] = {}
    for candidate, item in evidence.items():
        validation_improvement = float(item["initial_validation_loss"]) - float(
            item["final_validation_loss"]
        )
        eval_improvement = float(item["initial_eval_loss"]) - float(item["final_eval_loss"])
        average_improvement = (validation_improvement + eval_improvement) / 2.0
        wall_hours = float(item["pilot_wall_seconds"]) / 3600.0
        raw[candidate] = {
            "validation_improvement": validation_improvement,
            "eval_improvement": eval_improvement,
            "loss_improvement_per_hour": average_improvement / wall_hours,
            "transformer_body_capacity": float(item["transformer_body_params"]),
            "generation_health_trend": max(
                0.0,
                generation_health(item["final_generation"])
                - generation_health(item["initial_generation"]),
            ),
            "long_run_stability": 100.0 if item["soak_passed"] else 0.0,
            "vram_headroom": float(item["vram_headroom_percent"]),
            "checkpoint_resume_reliability": float(item["checkpoint_resume_reliability"]),
            "implementation_backend_reliability": float(
                item["implementation_backend_reliability"]
            ),
        }

    ratio_dimensions = {
        key: _ratio_scores({candidate: values[key] for candidate, values in raw.items()})
        for key in (
            "validation_improvement",
            "eval_improvement",
            "loss_improvement_per_hour",
            "transformer_body_capacity",
            "generation_health_trend",
            "vram_headroom",
        )
    }
    rows: list[dict[str, Any]] = []
    for candidate in ("C", "D"):
        component_scores = {
            **{key: values[candidate] for key, values in ratio_dimensions.items()},
            "long_run_stability": raw[candidate]["long_run_stability"],
            "checkpoint_resume_reliability": raw[candidate]["checkpoint_resume_reliability"],
            "implementation_backend_reliability": raw[candidate][
                "implementation_backend_reliability"
            ],
        }
        weighted_score = sum(component_scores[key] * WEIGHTS[key] for key in WEIGHTS)
        hard_failures = list(evidence[candidate].get("hard_failures", []))
        rows.append(
            {
                "candidate": candidate,
                "weighted_score": weighted_score,
                "eligible": not hard_failures,
                "hard_failures": hard_failures,
                "component_scores": component_scores,
                "raw_metrics": raw[candidate],
            }
        )

    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: row["weighted_score"],
        reverse=True,
    )
    if not eligible:
        selected = None
        tie_break_applied = False
    elif len(eligible) == 1:
        selected = eligible[0]["candidate"]
        tie_break_applied = False
    else:
        top, second = eligible[:2]
        relative_gap = (top["weighted_score"] - second["weighted_score"]) / top[
            "weighted_score"
        ]
        tie_break_applied = relative_gap <= TIE_BAND
        selected = "C" if tie_break_applied else top["candidate"]

    c_final = float(evidence["C"]["final_validation_loss"])
    d_final = float(evidence["D"]["final_validation_loss"])
    d_materially_better = (c_final - d_final) / c_final >= MATERIAL_VALIDATION_ADVANTAGE
    return {
        "rows": rows,
        "selected": selected,
        "tie_break_applied": tie_break_applied,
        "d_materially_better_validation": d_materially_better,
        "materiality_definition": (
            "Candidate D must have at least 2% lower final validation loss than C, or a "
            "clearly better loss slope without a stability penalty."
        ),
    }


def validate_memory_audit(payload: dict[str, Any]) -> None:
    if payload.get("result") != "PASS" or set(payload.get("results", {})) != {"C", "D"}:
        raise ValueError("memory audit must pass for both finalists")
    for item in payload["results"].values():
        if int(item.get("optimizer_steps", 0)) < 10:
            raise ValueError("memory audit requires at least 10 optimizer steps")
        if int(item.get("optimizer_state_tensors", 0)) <= 0:
            raise ValueError("optimizer state was not materialized")
        if float(item.get("reserved_headroom_percent", 0.0)) < 15.0:
            raise ValueError("unsafe post-warmup VRAM headroom")


def validate_resume_continuity(payload: dict[str, Any]) -> None:
    required = (
        "rng_continuity",
        "scheduler_continuity",
        "data_position_continuity",
        "no_repeated_or_skipped_sequence",
    )
    if payload.get("result") != "PASS" or not all(payload.get(key) is True for key in required):
        raise ValueError("pilot resume continuity failed")


def validate_learning_rate_result(payload: dict[str, Any]) -> None:
    required = {
        "candidate",
        "learning_rate",
        "tokens",
        "initial_validation_loss",
        "final_validation_loss",
        "gradient_norm_max",
        "stable",
        "non_finite_events",
        "result",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"learning-rate result is missing {sorted(missing)}")
    if int(payload["tokens"]) != 524_288:
        raise ValueError("learning-rate calibration token budget mismatch")
    if payload["result"] != "PASS" or not payload["stable"] or int(payload["non_finite_events"]) != 0:
        raise ValueError("unstable learning-rate calibration")


def validate_corpus_approval(payload: dict[str, Any]) -> None:
    allowed = {"approved", "conditional", "deferred", "rejected"}
    sources = payload.get("sources", [])
    if len(sources) != 20:
        raise ValueError("Corpus V3 approval must classify all 20 candidates")
    required = {
        "official_url",
        "edition",
        "license",
        "redistribution_terms",
        "attribution_requirements",
        "checksum_availability",
        "expected_download_bytes",
        "expected_usable_tokens",
        "language",
        "content_category",
        "maximum_source_cap_tokens",
        "quality_tier",
        "duplication_risks",
        "contamination_risks",
        "pii_privacy_risks",
        "extraction_implementation_readiness",
        "approval_status",
        "approval_reason",
    }
    for source in sources:
        missing = required - set(source)
        if missing:
            raise ValueError(f"{source.get('id')} is missing {sorted(missing)}")
        if source["approval_status"] not in allowed:
            raise ValueError(f"invalid approval status for {source['id']}")
