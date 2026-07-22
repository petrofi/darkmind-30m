"""Classify the completed first Corpus V3 pass and render the Phase 4F quality review."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from darkmind_v2.evaluation.phase4f_generation import compact_metrics
from darkmind_v2.training.phase4c_diagnostics import learning_rate_for_policy
from darkmind_v2.training.phase4f_completion import (
    BASELINE_EVAL,
    BASELINE_VALIDATION,
    FINAL_STEP,
    FINAL_TOKENS,
    MILESTONES,
    ROOT,
    RUN_DIR,
    RUNTIME_ROOT,
    SOURCE_STEP,
    USABLE_SEQUENCE_CAPACITY,
    V2_CONFIG,
    atomic_write_json,
    directory_size,
    load_json,
)


REPORT_PATH = ROOT / "darkmind_v2" / "reports" / "phase4f_base_quality_review.md"


def improvement_percent(start: float, end: float) -> float:
    return (start - end) * 100.0 / start


def classify_first_pass(
    *,
    validation_improvement: float,
    eval_improvement: float,
    validation_rebound: float,
    eval_rebound: float,
    integrity_pass: bool,
    memorization_pass: bool,
    catastrophic_probe_regression: bool,
    sustained_late_worsening: bool,
) -> str:
    hard_failure = (
        validation_improvement < -1.0
        or eval_improvement < -1.0
        or validation_rebound > 1.0
        or eval_rebound > 1.0
        or not integrity_pass
        or not memorization_pass
        or catastrophic_probe_regression
        or sustained_late_worsening
    )
    if hard_failure:
        return "FAIL"
    if validation_improvement >= 1.0 and eval_improvement >= 1.0:
        return "STRONG PASS"
    if max(validation_improvement, eval_improvement) > 0.25:
        return "PASS"
    return "PLATEAU"


def _rebound(losses: list[float]) -> float:
    return (losses[-1] - min(losses)) * 100.0 / min(losses)


def _probe_summary(evaluations: dict[str, Any]) -> dict[str, Any]:
    start = evaluations[str(SOURCE_STEP)]["probes"]
    final = evaluations[str(FINAL_STEP)]["probes"]
    return {
        name: {
            "baseline_loss": float(item["loss"]),
            "final_loss": float(final[name]["loss"]),
            "improvement_percent": improvement_percent(float(item["loss"]), float(final[name]["loss"])),
            "catastrophic_regression": float(final[name]["loss"]) > float(item["loss"]) * 1.20,
        }
        for name, item in start.items()
        if name in final
    }


def _meaningful_proxy() -> dict[str, Any]:
    manifest = load_json(RUN_DIR / "audits" / f"step_{FINAL_STEP:06d}" / "authoritative" / "greedy_manifest.json")
    counts = {"turkish": 0, "english": 0, "technical": 0}
    reviewed = {"turkish": 0, "english": 0, "technical": 0}
    for record in manifest["results"]:
        key = "technical" if record["category"] == "technical" else "turkish" if record["language"] == "tr" else "english"
        reviewed[key] += 1
        warnings = set(record["policy"]["warnings"])
        healthy = record["generated_token_count"] >= 8 and record["unique_token_ratio"] >= 0.35 and "repetition" not in warnings
        counts[key] += int(healthy)
    return {
        "method": "automatic non-empty/non-repetitive continuation-health proxy; not a semantic-quality judgment",
        "healthy_proxy_counts": counts,
        "reviewed_counts": reviewed,
        "manual_semantic_review_required": True,
    }


def _thermal(manifest: dict[str, Any]) -> dict[str, Any]:
    samples = [sample for segment in manifest["segments"] for sample in segment["gpu_samples"] if sample.get("available")]
    temperatures = [float(item["temperature_c"]) for item in samples if item.get("temperature_c") is not None]
    powers = [float(item["power_w"]) for item in samples if item.get("power_w") is not None]
    return {
        "samples": len(samples),
        "temperature_min_c": min(temperatures) if temperatures else None,
        "temperature_mean_c": statistics.fmean(temperatures) if temperatures else None,
        "temperature_max_c": max(temperatures) if temperatures else None,
        "power_min_w": min(powers) if powers else None,
        "power_mean_w": statistics.fmean(powers) if powers else None,
        "power_max_w": max(powers) if powers else None,
    }


def select_best_validation_step(evaluations: dict[str, Any], steps: list[int]) -> int:
    if not steps:
        raise ValueError("at least one validation milestone is required")
    return min(steps, key=lambda step: (float(evaluations[str(step)]["validation"]["loss"]), step))


def build_assessment() -> dict[str, Any]:
    manifest = load_json(RUN_DIR / "run_manifest.json")
    progress = load_json(RUN_DIR / "progress.json")
    resume = load_json(RUN_DIR / "final_resume_validation.json")
    evaluations = load_json(RUN_DIR / "evaluations.json")
    generation = load_json(RUN_DIR / "generation_analysis.json")
    memorization = load_json(RUN_DIR / "memorization_audit.json")
    if int(progress["optimizer_step"]) != FINAL_STEP or resume.get("result") != "PASS":
        raise RuntimeError("Phase 4F final training/integrity evidence is incomplete")
    steps = [step for step in MILESTONES if str(step) in evaluations]
    validation_losses = [float(evaluations[str(step)]["validation"]["loss"]) for step in steps]
    eval_losses = [float(evaluations[str(step)]["eval"]["loss"]) for step in steps]
    validation_improvement = improvement_percent(BASELINE_VALIDATION, validation_losses[-1])
    eval_improvement = improvement_percent(BASELINE_EVAL, eval_losses[-1])
    validation_rebound = _rebound(validation_losses)
    eval_rebound = _rebound(eval_losses)
    probes = _probe_summary(evaluations)
    catastrophic = any(item["catastrophic_regression"] for item in probes.values())
    sustained_worsening = len(validation_losses) >= 3 and all(
        validation_losses[index] > validation_losses[index - 1] and eval_losses[index] > eval_losses[index - 1]
        for index in range(len(validation_losses) - 2, len(validation_losses))
    )
    integrity = (
        len(manifest["segments"]) == 3
        and all(segment["result"] == "PASS" and segment["no_data_wrap"] for segment in manifest["segments"])
        and all(load_json(RUN_DIR / "gates" / f"step_{step:06d}.json")["result"] == "PASS" for step in MILESTONES[1:-1])
        and generation["hard_failure_total"] == 0
        and resume["checks"]["step_11973_not_run"]
    )
    classification = classify_first_pass(
        validation_improvement=validation_improvement,
        eval_improvement=eval_improvement,
        validation_rebound=validation_rebound,
        eval_rebound=eval_rebound,
        integrity_pass=integrity,
        memorization_pass=memorization["result"] == "PASS" and not memorization["hard_release_blockers"],
        catastrophic_probe_regression=catastrophic,
        sustained_late_worsening=sustained_worsening,
    )
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gradients = [float(item["pre_clip_gradient_norm"]) for item in metrics]
    coefficients = [float(item["clipping_coefficient"]) for item in metrics]
    updates = [float(item["update_to_weight"]["maximum"]) for item in metrics]
    config = load_json(V2_CONFIG)
    best_step = select_best_validation_step(evaluations, steps)
    assessment = {
        "schema_version": "darkmind-v2-phase4f-first-pass-assessment-v1",
        "result": "PASS" if classification != "FAIL" else "FAIL",
        "classification": classification,
        "source_checkpoint": manifest["source_checkpoint"],
        "starting_optimizer_step": SOURCE_STEP,
        "starting_tokens": SOURCE_STEP * 8192,
        "final_optimizer_step": FINAL_STEP,
        "final_tokens": FINAL_TOKENS,
        "consumed_sequences": USABLE_SEQUENCE_CAPACITY,
        "excluded_tail_sequences": 14,
        "validation_progression": {str(step): evaluations[str(step)]["validation"] for step in steps},
        "eval_progression": {str(step): evaluations[str(step)]["eval"] for step in steps},
        "train_progression": {str(step): evaluations[str(step)].get("train", {}) for step in steps},
        "learning_rate_progression": {str(step): learning_rate_for_policy(step, config["schedule"]) for step in steps},
        "validation_improvement_percent": validation_improvement,
        "eval_improvement_percent": eval_improvement,
        "validation_rebound_percent": validation_rebound,
        "eval_rebound_percent": eval_rebound,
        "sustained_late_worsening": sustained_worsening,
        "probe_trends": probes,
        "catastrophic_probe_regression": catastrophic,
        "integrity_pass": integrity,
        "memorization_pass": memorization["result"] == "PASS" and not memorization["hard_release_blockers"],
        "best_validation_step": best_step,
        "best_checkpoint": manifest["checkpoints"][str(best_step)],
        "best_validation_loss": evaluations[str(best_step)]["validation"]["loss"],
        "final_eval_perplexity": evaluations[str(FINAL_STEP)]["eval"]["perplexity"],
        "gradient_norm_p50": statistics.median(gradients),
        "gradient_norm_p95": sorted(gradients)[int(0.95 * (len(gradients) - 1))],
        "gradient_norm_max": max(gradients),
        "clipping_coefficient_p50": statistics.median(coefficients),
        "clipping_coefficient_p95": sorted(coefficients)[int(0.95 * (len(coefficients) - 1))],
        "clipping_coefficient_min": min(coefficients),
        "clipped_step_fraction": sum(int(item["clipped"]) for item in metrics) / len(metrics),
        "update_to_weight_p50": statistics.median(updates),
        "update_to_weight_p95": sorted(updates)[int(0.95 * (len(updates) - 1))],
        "update_to_weight_max": max(updates),
        "non_finite_events": sum(int(item["non_finite_event_count"]) for item in metrics),
        "generation": generation,
        "meaningful_continuation_review": _meaningful_proxy(),
        "memorization": memorization,
        "thermal": _thermal(manifest),
        "second_epoch_justified": False,
        "new_unique_corpus_expansion_recommended": classification in {"STRONG PASS", "PASS", "PLATEAU"},
        "instruction_tuning_technically_justified": False,
        "upload_performed": False,
        "second_epoch_started": False,
    }
    atomic_write_json(RUN_DIR / "final_assessment.json", assessment)
    return assessment


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_report(assessment: dict[str, Any]) -> None:
    generation = assessment["generation"]
    final_generation = generation["final"]
    memorization = assessment["memorization"]
    meaningful = assessment["meaningful_continuation_review"]
    evaluations = load_json(RUN_DIR / "evaluations.json")
    manifest = load_json(RUN_DIR / "run_manifest.json")
    lines = [
        "# DarkMind v2 Phase 4F Base Quality Review",
        "",
        f"First-pass classification: **{assessment['classification']}**",
        "",
        "This review covers the first deterministic Corpus V3 pass only. The model is a from-scratch Base V1 model, not instruction-tuned, not a chat model, not production-ready, and not publicly uploaded.",
        "",
        "## Exact completion identity",
        "",
        f"- Start: step {SOURCE_STEP:,}, {SOURCE_STEP * 8192:,} tokens.",
        f"- Final no-wrap stop: step {FINAL_STEP:,}, {FINAL_TOKENS:,} tokens, {USABLE_SEQUENCE_CAPACITY:,} sequences.",
        "- The deterministic 14-sequence incomplete tail was excluded; step 11,973 and a second epoch were not run.",
        f"- Best validation checkpoint: step {assessment['best_validation_step']:,} at `{assessment['best_checkpoint']}`.",
        "",
        "## Loss progression",
        "",
        "| Step | Tokens | Applied LR | Train | Validation | Eval | Eval perplexity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in MILESTONES:
        item = evaluations[str(step)]
        train = item.get("train", {})
        lines.append(
            f"| {step:,} | {step * 8192:,} | {assessment['learning_rate_progression'][str(step)]:.12f} | "
            f"{train.get('loss', float('nan')):.6f} | {item['validation']['loss']:.6f} | {item['eval']['loss']:.6f} | {item['eval']['perplexity']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"75M to final validation/eval improvement: {assessment['validation_improvement_percent']:.3f}% / {assessment['eval_improvement_percent']:.3f}%. Rebound: {assessment['validation_rebound_percent']:.3f}% / {assessment['eval_rebound_percent']:.3f}%.",
            "",
            "## Optimization and activations",
            "",
            f"Gradient p50/p95/max: {assessment['gradient_norm_p50']:.4f} / {assessment['gradient_norm_p95']:.4f} / {assessment['gradient_norm_max']:.4f}. Clipping rate: {assessment['clipped_step_fraction']:.1%}.",
            f"Clipping coefficient p50/p95/min: {assessment['clipping_coefficient_p50']:.6f} / {assessment['clipping_coefficient_p95']:.6f} / {assessment['clipping_coefficient_min']:.6f}.",
            f"Update-to-weight p50/p95/max: {assessment['update_to_weight_p50']:.8f} / {assessment['update_to_weight_p95']:.8f} / {assessment['update_to_weight_max']:.8f}. Non-finite events: {assessment['non_finite_events']}.",
            "Persistent clipping remains a production-policy warning. It was not treated as an automatic failure because losses, updates, logits, and activations remained controlled under the frozen policy.",
            "",
            "| Step | Logit std | Prediction entropy | Embedding norm | Final/early residual RMS |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for step in MILESTONES[1:]:
        diagnostic = load_json(Path(manifest["diagnostic_snapshots"][str(step)]))
        activation = diagnostic["activation"]
        lines.append(
            f"| {step:,} | {activation['logits']['std']:.6f} | {activation['logits']['softmax_entropy_mean']:.6f} | "
            f"{diagnostic['embedding_norm']['mean']:.6f} | {activation['final_to_early_residual_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Language and category probes",
            "",
            "| Probe | 75M loss | Final loss | Improvement | Catastrophic regression |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, item in assessment["probe_trends"].items():
        lines.append(
            f"| {name} | {item['baseline_loss']:.6f} | {item['final_loss']:.6f} | {item['improvement_percent']:.3f}% | {'YES' if item['catastrophic_regression'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "Turkish and English grammatical/topical continuation, technical continuation, factual reliability, and code/structured behavior remain base-model diagnostics. Automatic probe improvements do not establish factual correctness or user-facing quality.",
            "",
            "## Final generation audit",
            "",
            "| Mode | Generations | Repetition | Exact loops | EOS | Mean unique-token ratio | Empty | Short |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in ("greedy", "sampling"):
        item = final_generation[mode]
        lines.append(
            f"| {mode} | {item['generations']} | {item['repetition_warning_rate']:.1%} | {item['exact_loop_rate']:.1%} | "
            f"{item['eos_completion_rate']:.1%} | {item['mean_unique_token_ratio']:.3f} | {item['empty_output_count']} | {item['short_output_count']} |"
        )
    lines.extend(
        [
            "",
            "The audit retained unsanitized raw outputs outside Git. Invalid-byte, replacement-character, mojibake, script-consistency, and special-token leakage counters are recorded in runtime evidence. No chatbot-quality claim is made.",
            "",
            f"Automatic non-empty/non-repetitive continuation-health proxy counts were Turkish {meaningful['healthy_proxy_counts']['turkish']}/{meaningful['reviewed_counts']['turkish']}, English {meaningful['healthy_proxy_counts']['english']}/{meaningful['reviewed_counts']['english']}, and technical {meaningful['healthy_proxy_counts']['technical']}/{meaningful['reviewed_counts']['technical']}. These are not semantic-quality judgments; manual review remains required.",
            "",
            "## Memorization and PII",
            "",
            f"Controlled prefixes: {memorization['train_prefix_count']} train and {memorization['heldout_prefix_count']} validation/eval. Exact continuation rates: {memorization['exact_continuation_match']['train_rate']:.1%} train and {memorization['exact_continuation_match']['heldout_rate']:.1%} held-out.",
            f"Longest exact target span: {memorization['near_exact_similarity']['longest_exact_span_tokens']} tokens. Longest generated n-gram found in the full training shards: {memorization['training_corpus_ngram']['longest_exact_generated_ngram_in_training_tokens']} tokens.",
            f"The broad regex initially flagged {memorization['pii_like_generation_counts']['phone']} numeric phone-like candidates and produced an initial {memorization['initial_audit_result']} record. That immutable record is preserved at hash `{memorization['initial_audit_sha256']}`.",
            f"Deterministic adjudication found plausible identities {memorization['plausible_identity_generation_counts']}; all five phone-like candidates were repetitive year/ISBN-like numeric false positives. Material personal-data reproductions: {memorization['material_personal_data_reproduction_count']}. Final hard blockers: {memorization['hard_release_blockers'] or 'none'}.",
            "Extraction risk is not claimed to be zero. The audit is controlled and bounded; it does not prove absence of memorization outside its probes.",
            "",
            "## Separate base-quality review",
            "",
            "| Dimension | Assessment |",
            "|---|---|",
            f"| Turkish grammatical continuation | Emerging structure, but still weak and repetition-prone; health proxy {meaningful['healthy_proxy_counts']['turkish']}/{meaningful['reviewed_counts']['turkish']}. |",
            "| Turkish topical consistency | Some probe improvement, but sustained topical coherence is not established. |",
            f"| English grammatical continuation | Emerging structure remains weaker than Turkish; health proxy {meaningful['healthy_proxy_counts']['english']}/{meaningful['reviewed_counts']['english']}. |",
            "| English topical consistency | Not established; source probes are mostly flat-to-positive. |",
            f"| Technical-text continuation | Weak; health proxy {meaningful['healthy_proxy_counts']['technical']}/{meaningful['reviewed_counts']['technical']} and English technical loss improved only marginally. |",
            "| Factual reliability | Not reliable; factual outputs retain an explicit unreliability warning. |",
            "| Code/structured output | Weak and not suitable for use; code-generation warnings remain. |",
            f"| Repetition | High: greedy {final_generation['greedy']['repetition_warning_rate']:.1%}, sampling {final_generation['sampling']['repetition_warning_rate']:.1%}. |",
            f"| EOS behavior | Weak: greedy {final_generation['greedy']['eos_completion_rate']:.1%}, sampling {final_generation['sampling']['eos_completion_rate']:.1%}. |",
            "| Memorization risk | Controlled audit passed after preserving and adjudicating the initial broad-regex failure; risk is not claimed to be zero. |",
            "| PII risk | No plausible identity or material personal-data reproduction was observed in the bounded audit. |",
            "",
            "## Base-quality decision",
            "",
            f"The model has learned measurable Turkish/English and technical language structure, with final classification **{assessment['classification']}**. It remains too weak and insufficiently reviewed for instruction tuning or user-facing use.",
            "New unique Corpus V3 expansion is the preferred next investment. A second identical epoch is not justified automatically and would require a separate approval with renewed overexposure and extraction-risk controls.",
            "A stronger open-base Instruct track should remain separate from this Base V1 evidence line. No SFT, Qwen teacher generation, second epoch, or upload occurred.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def storage_report() -> dict[str, Any]:
    phase_sizes = {name: directory_size(Path(fr"C:\DarkMindRuntime\{name}")) for name in ("phase4b", "phase4c", "phase4d", "phase4e", "phase4f")}
    payload = {
        "schema_version": "darkmind-v2-phase4f-storage-report-v1",
        "phase_sizes_bytes": phase_sizes,
        "immutable_input_bytes": directory_size(Path(r"C:\DarkMindRuntime\phase4b\inputs")),
        "required_final_checkpoint_files": ["model/model.safetensors", "resume_state.pt", "checkpoint_metadata.json", "phase4f_checkpoint_metadata.json"],
        "future_continuation_files": ["final full-resume checkpoint", "authorization", "sequence-order manifest", "frozen tokenizer/model/corpus inputs", "run manifest", "gate evidence"],
        "safe_to_move_after_hash_verified_archival": ["model-only 90M checkpoint", "raw generation collections", "disposable exploratory evidence"],
        "must_remain_until_hash_verified_archival": ["85M, 95M, and final resume checkpoints", "audit summaries", "manifests", "hash records"],
        "files_deleted_or_moved": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "storage_report.json", payload)
    return payload


def finalize() -> dict[str, Any]:
    assessment = build_assessment()
    render_report(assessment)
    assessment["storage"] = storage_report()
    atomic_write_json(RUN_DIR / "final_assessment.json", assessment)
    return assessment


def main() -> None:
    payload = finalize()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
