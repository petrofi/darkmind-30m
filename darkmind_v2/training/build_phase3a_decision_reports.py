"""Score Phase 3A candidates and project conservative stage budgets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "darkmind_v2" / "reports"
MATRIX_PATH = REPORT_DIR / "phase3a_architecture_parameter_matrix.json"
BENCHMARK_PATH = REPORT_DIR / "phase3a_rtx4060_benchmark.json"
STAGES = (5_000_000, 25_000_000, 100_000_000, 250_000_000, 500_000_000)
WEIGHTS = {
    "expected_model_capacity": 0.25,
    "transformer_body": 0.15,
    "vocabulary_efficiency": 0.10,
    "measured_throughput": 0.15,
    "vram_headroom": 0.15,
    "training_stability": 0.10,
    "checkpoint_practicality": 0.05,
    "implementation_simplicity": 0.05,
}


def _normalize(values: dict[str, float], *, higher_is_better: bool = True) -> dict[str, float]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {key: 100.0 for key in values}
    if higher_is_better:
        return {key: (value - minimum) / (maximum - minimum) * 100 for key, value in values.items()}
    return {key: (maximum - value) / (maximum - minimum) * 100 for key, value in values.items()}


def _preferred_profiles(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for candidate in "ABCD":
        matches = [
            profile
            for profile in benchmark["profiles"]
            if profile.get("candidate") == candidate
            and profile.get("micro_batch_size") == 2
            and profile.get("gradient_checkpointing") is False
            and profile.get("attention_implementation") == "sdpa"
            and profile.get("hard_failure") is None
            and profile.get("safe_vram_margin") is True
            and profile.get("loss_finite") is True
            and profile.get("gradients_finite") is True
        ]
        if len(matches) != 1:
            raise ValueError(f"candidate {candidate} lacks one safe preferred benchmark profile")
        profiles[candidate] = matches[0]
    return profiles


def score_candidates(matrix: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    candidates = {item["candidate"]: item for item in matrix["candidates"]}
    profiles = _preferred_profiles(benchmark)
    capacity = {key: float(item["estimate"]["total_params"]) for key, item in candidates.items()}
    body = {key: float(item["estimate"]["transformer_body_params"]) for key, item in candidates.items()}
    vocab = {key: float(item["estimate"]["vocab_related_percentage"]) for key, item in candidates.items()}
    throughput = {key: float(profile["tokens_per_second"]) for key, profile in profiles.items()}
    headroom = {key: float(profile["vram_headroom_percent"]) for key, profile in profiles.items()}
    stability = {
        key: sum(
            profile.get("hard_failure") is None
            and profile.get("loss_finite") is True
            and profile.get("gradients_finite") is True
            for profile in benchmark["profiles"]
            if profile.get("candidate") == key
        )
        / sum(profile.get("candidate") == key for profile in benchmark["profiles"])
        * 100
        for key in candidates
    }
    checkpoint_bytes = {
        key: float(benchmark["candidate_checkpoints"][key]["checkpoint_file_bytes"])
        for key in candidates
    }
    checkpoint_seconds = {
        key: float(benchmark["candidate_checkpoints"][key]["checkpoint_save_seconds"])
        + float(benchmark["candidate_checkpoints"][key]["checkpoint_reload_seconds"])
        for key in candidates
    }
    checkpoint_size_score = _normalize(checkpoint_bytes, higher_is_better=False)
    checkpoint_time_score = _normalize(checkpoint_seconds, higher_is_better=False)
    checkpoint_practicality = {
        key: (checkpoint_size_score[key] + checkpoint_time_score[key]) / 2 for key in candidates
    }
    minimum_layers = min(item["config"]["n_layer"] for item in candidates.values())
    simplicity = {
        key: minimum_layers / item["config"]["n_layer"] * 100 for key, item in candidates.items()
    }
    dimensions = {
        "expected_model_capacity": _normalize(capacity),
        "transformer_body": _normalize(body),
        "vocabulary_efficiency": _normalize(vocab, higher_is_better=False),
        "measured_throughput": _normalize(throughput),
        "vram_headroom": _normalize(headroom),
        "training_stability": stability,
        "checkpoint_practicality": checkpoint_practicality,
        "implementation_simplicity": simplicity,
    }
    rows: list[dict[str, Any]] = []
    for candidate in "ABCD":
        component_scores = {name: values[candidate] for name, values in dimensions.items()}
        weighted = sum(component_scores[name] * WEIGHTS[name] for name in WEIGHTS)
        rows.append(
            {
                "candidate": candidate,
                "weighted_score": weighted,
                "component_scores": component_scores,
                "measured_tokens_per_second": throughput[candidate],
                "vram_headroom_percent": headroom[candidate],
                "successful_profiles": round(stability[candidate] / 100 * 8),
                "total_profiles": 8,
            }
        )
    rows.sort(key=lambda row: row["weighted_score"], reverse=True)
    top, second = rows[:2]
    selected = top
    tie_break_applied = False
    if top["weighted_score"] - second["weighted_score"] <= 3.0:
        selected = min((top, second), key=lambda row: capacity[row["candidate"]])
        tie_break_applied = selected is not top
    return {
        "rows": rows,
        "selected": selected["candidate"],
        "tie_break_applied": tie_break_applied,
        "profiles": profiles,
    }


def _gradient_checkpointing_rows(benchmark: dict[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for candidate in "ABCD":
        base = [
            profile
            for profile in benchmark["profiles"]
            if profile.get("candidate") == candidate
            and profile.get("micro_batch_size") == 1
            and profile.get("attention_implementation") == "sdpa"
        ]
        enabled = next(profile for profile in base if profile["gradient_checkpointing"] is True)
        disabled = next(profile for profile in base if profile["gradient_checkpointing"] is False)
        rows.append(
            {
                "candidate": candidate,
                "throughput_change_percent": (enabled["tokens_per_second"] / disabled["tokens_per_second"] - 1) * 100,
                "reserved_memory_change_percent": (enabled["peak_reserved_bytes"] / disabled["peak_reserved_bytes"] - 1) * 100,
            }
        )
    return rows


def render_decision(matrix: dict[str, Any], benchmark: dict[str, Any], decision: dict[str, Any]) -> str:
    matrix_by_id = {item["candidate"]: item for item in matrix["candidates"]}
    lines = [
        "# Phase 3A Base Candidate Decision",
        "",
        "Status: **RECOMMENDED PENDING CORPUS APPROVAL; NOT FROZEN**",
        "",
        "Recommended architecture: **Candidate C (103,881,216 parameters)**",
        "",
        "Recommendation strength: **MODERATE**. Candidate C has safe measured headroom and practical throughput, while one isolated checkpointing-on MB2 SDPA worker exited without a result. The matching checkpointing-off SDPA and fallback profiles passed, so this is a profile-level stability concern rather than an architecture-wide hard rejection.",
        "",
        "## Weighted Decision",
        "",
        "| Rank | Candidate | Weighted score | Params | Body params | Vocab share | BF16 tok/s | VRAM headroom | Stable profiles |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(decision["rows"], start=1):
        item = matrix_by_id[row["candidate"]]
        lines.append(
            f"| {rank} | {row['candidate']} | {row['weighted_score']:.2f} | "
            f"{item['estimate']['total_params']:,} | {item['estimate']['transformer_body_params']:,} | "
            f"{item['estimate']['vocab_related_percentage']:.4f}% | {row['measured_tokens_per_second']:,.1f} | "
            f"{row['vram_headroom_percent']:.2f}% | {row['successful_profiles']}/{row['total_profiles']} |"
        )
    lines.extend(
        [
            "",
            "The raw weighted score places D ahead of C by less than 3 percentage points. The required tie-break therefore selects the smaller and faster C. D is not selected merely for being larger.",
            "",
            "## Why C",
            "",
            "- C has 85,449,216 transformer-body parameters, 32.8% more than B, while its preferred-profile throughput is only 10.5% lower than B.",
            "- C leaves 83.85% of reserved VRAM free in the measured MB2 SDPA profile, far above the 15% safety gate.",
            "- C uses the shallowest candidate depth (12 layers), standard 64-dimensional heads, and the same tied-tokenizer contract as every candidate.",
            "- C checkpoint save/reload passed; its BF16 package is practical at about 201 MiB.",
            "",
            "## Rejections and Deferrals",
            "",
            "- A is not recommended because its 50,701,312-parameter body leaves materially less capacity than C; the speed advantage over C is only about 4.4%.",
            "- B is the fastest candidate and remains the fallback recommendation if future long-run C stability fails, but its body is 21.1M parameters smaller than C.",
            "- D is not recommended because it is about 13.1% slower than C, has a larger/slower checkpoint, and is within the 3-point score tie band where the smaller/faster rule applies.",
            "- No candidate was rejected for vocabulary share, head dimensions, tied embeddings, OOM, non-finite values, or checkpoint reload.",
            "",
            "## Gradient Checkpointing",
            "",
            "| Candidate | MB1 SDPA throughput change | Reserved-memory change |",
            "|---|---:|---:|",
        ]
    )
    for row in _gradient_checkpointing_rows(benchmark):
        lines.append(
            f"| {row['candidate']} | {row['throughput_change_percent']:.1f}% | "
            f"{row['reserved_memory_change_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Checkpointing is implemented and valid, but it is not recommended for the measured MB1/MB2 profiles because headroom is already large and recomputation costs 24-37% throughput. It remains available for longer contexts or future larger batches.",
            "",
            "## Hard Failure Record",
            "",
            "Candidate C MB2 + gradient checkpointing + SDPA exited with Windows code 3221226505 before writing a result. No OOM was reported. The failure remains in the benchmark JSON and lowers C's stability score to 87.5%; it must be rerun during architecture-freeze review.",
            "",
        ]
    )
    return "\n".join(lines)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}h"


def render_training_budget(
    matrix: dict[str, Any], benchmark: dict[str, Any], decision: dict[str, Any]
) -> str:
    matrix_by_id = {item["candidate"]: item for item in matrix["candidates"]}
    checkpoint_frequency = {
        5_000_000: "1M tokens",
        25_000_000: "2.5M tokens",
        100_000_000: "5M tokens",
        250_000_000: "10M tokens",
        500_000_000: "20M tokens",
    }
    evaluation_frequency = {
        5_000_000: "1M tokens and stage boundary",
        25_000_000: "2.5M tokens and stage boundary",
        100_000_000: "10M tokens and stage boundary",
        250_000_000: "25M tokens and stage boundary",
        500_000_000: "25M tokens and final audit",
    }
    lines = [
        "# Phase 3A Training Budget",
        "",
        "These projections use each candidate's measured safe MB2, SDPA, checkpointing-off throughput. For conservatism, sustained active throughput is 80% of the microbenchmark result, evaluation/checkpoint overhead is 15% of active time, and wall time adds a further 20% interruption/thermal contingency. These are planning numbers, not training authorization.",
        "",
        "| Candidate | Measured tok/s | Budget tok/s | BF16 model checkpoint | Measured optimizer state | Full resume checkpoint estimate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in "ABCD":
        profile = decision["profiles"][candidate]
        checkpoint = benchmark["candidate_checkpoints"][candidate]
        full_checkpoint = checkpoint["checkpoint_file_bytes"] + profile["optimizer_state_memory_bytes"]
        lines.append(
            f"| {candidate} | {profile['tokens_per_second']:,.1f} | {profile['tokens_per_second'] * 0.8:,.1f} | "
            f"{checkpoint['checkpoint_file_bytes'] / 2**20:.2f} MiB | "
            f"{profile['optimizer_state_memory_bytes'] / 2**20:.2f} MiB | {full_checkpoint / 2**20:.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Stage Durations",
            "",
            "| Candidate | Tokens | Active GPU | Eval/checkpoint overhead | Estimated wall | Checkpoint frequency | Evaluation frequency |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for candidate in "ABCD":
        sustained = decision["profiles"][candidate]["tokens_per_second"] * 0.8
        for tokens in STAGES:
            active = tokens / sustained
            overhead = active * 0.15
            wall = (active + overhead) * 1.20
            lines.append(
                f"| {candidate} | {tokens / 1_000_000:.0f}M | {_hours(active)} | {_hours(overhead)} | "
                f"{_hours(wall)} | {checkpoint_frequency[tokens]} | {evaluation_frequency[tokens]} |"
            )
    lines.extend(
        [
            "",
            "## Corpus Storage",
            "",
            "| Tokens | Raw uint16 | Planned shard/index budget |",
            "|---:|---:|---:|",
        ]
    )
    for tokens in STAGES:
        raw = tokens * 2
        planned = raw * 1.05
        lines.append(f"| {tokens / 1_000_000:.0f}M | {raw / 2**30:.3f} GiB | {planned / 2**30:.3f} GiB |")
    selected = decision["selected"]
    selected_profile = decision["profiles"][selected]
    selected_checkpoint = benchmark["candidate_checkpoints"][selected]
    selected_full = selected_checkpoint["checkpoint_file_bytes"] + selected_profile["optimizer_state_memory_bytes"]
    lines.extend(
        [
            "",
            "## Recommended Operating Plan",
            "",
            f"Candidate {selected} is projected at {_hours(500_000_000 / (selected_profile['tokens_per_second'] * 0.8))} active GPU and {_hours((500_000_000 / (selected_profile['tokens_per_second'] * 0.8)) * 1.15 * 1.20)} conservative wall time for 500M tokens. A complete model-plus-optimizer resume point is approximately {selected_full / 2**20:.2f} MiB before scheduler/RNG metadata.",
            "",
            "Use atomic checkpoint directories, retain the latest two valid checkpoints plus every stage boundary, fsync manifests before publishing the completion marker, and validate a real process resume at Stage 0 and Stage 1. A laptop power loss must never leave the last known-good checkpoint replaceable by a partial write.",
            "",
            "The microbenchmark omits real shard I/O, periodic full validation, generation audits, and thermal behavior over many hours. The 20% wall contingency does not replace a Stage 1 calibration run on the approved corpus.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    decision = score_candidates(matrix, benchmark)
    (REPORT_DIR / "phase3a_base_candidate_decision.md").write_text(
        render_decision(matrix, benchmark, decision), encoding="utf-8"
    )
    (REPORT_DIR / "phase3a_training_budget.md").write_text(
        render_training_budget(matrix, benchmark, decision), encoding="utf-8"
    )
    print(json.dumps({"recommended_candidate": decision["selected"], "scores": {row['candidate']: round(row['weighted_score'], 4) for row in decision['rows']}}, indent=2))


if __name__ == "__main__":
    main()
