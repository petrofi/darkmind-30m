"""Render the honest Phase 4E conditional-stop evidence after the 75M gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from darkmind_v2.evaluation.phase4e_memorization import write_not_run_record
from darkmind_v2.training.phase4b_factorial import percentile
from darkmind_v2.training.phase4c_diagnostics import learning_rate_for_policy
from darkmind_v2.training.phase4e_stage3 import (
    FINAL_STEP,
    GATE_50_STEP,
    GATE_75_STEP,
    MILESTONES,
    PHASE4D_RUN,
    ROOT,
    RUN_DIR,
    RUNTIME_ROOT,
    SOURCE_STEP,
    UNIQUE_SEQUENCE_CAPACITY,
    UNUSABLE_TAIL_SEQUENCES,
    USABLE_SEQUENCE_CAPACITY,
    V2_CONFIG,
    atomic_write_json,
    directory_size,
    load_json,
)


STAGE_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4e_base_v1_100m.md"
SAMPLES_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4e_base_v1_100m_samples.md"
HEALTH_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4e_training_health.md"
MEMORIZATION_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4e_memorization_audit.md"
QUALITY_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4e_base_quality_review.md"
PROGRESSION_STEPS = tuple(step for step in MILESTONES if step <= GATE_75_STEP)


def _improvement(start: float, end: float) -> float:
    return (start - end) * 100.0 / start


def _rate(summary: dict[str, Any], warning: str | None = None) -> float:
    count = summary["exact_repeated_ngram_loop_outputs"] if warning is None else summary["quality_warning_counts"].get(warning, 0)
    return count / max(summary["generations"], 1)


def _probe_comparison(start: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in final.items():
        if name not in start:
            continue
        baseline = float(start[name]["loss"])
        end = float(value["loss"])
        result[name] = {
            "baseline_loss": baseline,
            "final_loss": end,
            "improvement_percent": _improvement(baseline, end),
            "catastrophic_regression": end > baseline * 1.20,
        }
    return result


def build_assessment() -> dict[str, Any]:
    manifest = load_json(RUN_DIR / "run_manifest.json")
    progress = load_json(RUN_DIR / "progress.json")
    gate50 = load_json(RUN_DIR / "gates" / f"step_{GATE_50_STEP:06d}.json")
    gate75 = load_json(RUN_DIR / "gates" / f"step_{GATE_75_STEP:06d}.json")
    if gate50["result"] != "PASS" or gate75["result"] != "CONDITIONAL" or gate75["continuation_authorized"]:
        raise RuntimeError("Phase 4E conditional-stop evidence changed")
    if int(progress["optimizer_step"]) != GATE_75_STEP or len(manifest["segments"]) != 2:
        raise RuntimeError("Phase 4E did not stop exactly at the 75M gate")
    evaluations = load_json(RUN_DIR / "evaluations.json")
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(metrics) != GATE_75_STEP - SOURCE_STEP:
        raise ValueError("Phase 4E telemetry count changed at conditional stop")
    gradients = [float(item["pre_clip_gradient_norm"]) for item in metrics]
    updates = [float(item["update_to_weight"]["maximum"]) for item in metrics]
    source = evaluations[str(SOURCE_STEP)]
    final = evaluations[str(GATE_75_STEP)]
    generation50 = load_json(RUN_DIR / "audits" / f"step_{GATE_50_STEP:06d}" / "subset" / "audit_summary.json")
    generation75 = load_json(RUN_DIR / "audits" / f"step_{GATE_75_STEP:06d}" / "subset" / "audit_summary.json")
    baseline_generation = load_json(PHASE4D_RUN / "generation_analysis.json")["authoritative_final"]
    config = load_json(V2_CONFIG)
    probes = _probe_comparison(source["probes"], final["probes"])
    memorization = write_not_run_record()
    payload = {
        "schema_version": "darkmind-v2-phase4e-conditional-stop-assessment-v1",
        "result": "PASS",
        "classification": "CONDITIONAL PASS",
        "stop_reason": "75M gate improved 1%-3% from 50M and did not authorize the final segment",
        "stopped_optimizer_step": GATE_75_STEP,
        "stopped_tokens": GATE_75_STEP * 8192,
        "stopped_sequence_index": GATE_75_STEP * 16,
        "final_no_wrap_step": FINAL_STEP,
        "final_no_wrap_reached": False,
        "remaining_full_batch_sequences": USABLE_SEQUENCE_CAPACITY - GATE_75_STEP * 16,
        "remaining_full_batch_tokens": (FINAL_STEP - GATE_75_STEP) * 8192,
        "unusable_tail_sequences": UNUSABLE_TAIL_SEQUENCES,
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "first_corpus_pass_complete": False,
        "validation_improvement_from_25m_percent": _improvement(float(source["validation"]["loss"]), float(final["validation"]["loss"])),
        "eval_improvement_from_25m_percent": _improvement(float(source["eval"]["loss"]), float(final["eval"]["loss"])),
        "gate_50m": gate50,
        "gate_75m": gate75,
        "progression": {
            str(step): {
                "tokens": step * 8192,
                "learning_rate": learning_rate_for_policy(step, config["schedule"]),
                "train": evaluations[str(step)].get("train", {}),
                "validation": evaluations[str(step)]["validation"],
                "eval": evaluations[str(step)]["eval"],
            }
            for step in PROGRESSION_STEPS
        },
        "training_health": {
            "steps_recorded": len(metrics),
            "gradient_p50": percentile(gradients, 0.50),
            "gradient_p95": percentile(gradients, 0.95),
            "gradient_max": max(gradients),
            "clipped_step_fraction": sum(int(item["clipped"]) for item in metrics) / len(metrics),
            "update_to_weight_p50": percentile(updates, 0.50),
            "update_to_weight_p95": percentile(updates, 0.95),
            "update_to_weight_max": max(updates),
            "non_finite_events": sum(int(item["non_finite_event_count"]) for item in metrics),
        },
        "probe_comparison": probes,
        "generation": {
            "baseline_25m": baseline_generation,
            "subset_50m": generation50,
            "subset_75m": generation75,
            "authoritative_final_run": False,
        },
        "process_restart": {
            "completed_fresh_processes": len(set(manifest["process_ids"])),
            "process_ids": manifest["process_ids"],
            "expected_completed_processes_at_stop": 2,
            "result": "PASS" if len(set(manifest["process_ids"])) == 2 else "FAIL",
            "optimizer_continuity": all(item["optimizer_continuity"] for item in manifest["segments"]),
            "scheduler_continuity": all(item["scheduler_continuity"] for item in manifest["segments"]),
            "rng_continuity": all(item["rng_continuity"] for item in manifest["segments"]),
            "data_position_continuity": all(item["data_position_continuity"] for item in manifest["segments"]),
            "no_repeated_or_skipped_sequence": all(item["no_repeated_or_skipped_sequence"] for item in manifest["segments"]),
        },
        "best_validation_step": manifest["best_validation_step"],
        "best_checkpoint": manifest["best_validation_checkpoint"],
        "final_checkpoint": manifest["latest_resume_checkpoint"],
        "final_checkpoint_hashes": manifest["checkpoint_hashes"][str(GATE_75_STEP)],
        "memorization_audit": memorization,
        "local_export_created": False,
        "local_export_reason": "final no-wrap checkpoint and final audit gates were not reached",
        "runtime_bytes": directory_size(RUNTIME_ROOT),
        "storage_preflight": manifest["storage_preflight"],
        "second_epoch_justified": False,
        "second_epoch_authorized": False,
        "corpus_v3_expansion_recommended": True,
        "sft_started": False,
        "qwen_generation_started": False,
        "upload_performed": False,
    }
    atomic_write_json(RUN_DIR / "conditional_stop_assessment.json", payload)
    return payload


def render_reports(payload: dict[str, Any]) -> None:
    progression = payload["progression"]
    stage_lines = [
        "# DarkMind v2 Phase 4E Base V1 100M-horizon attempt",
        "",
        "Classification: **CONDITIONAL PASS**",
        "",
        "Training stopped exactly at the 75M continuation gate. The final no-wrap stop was not authorized or reached, so this is not a completed Corpus V3 pass and not a 100M result.",
        "",
        "## Gate decision",
        "",
        f"- 50M gate: PASS; validation/eval improvement from 25M `{payload['gate_50m']['validation_improvement_percent']:.3f}%` / `{payload['gate_50m']['eval_improvement_percent']:.3f}%`.",
        f"- 75M gate: CONDITIONAL; validation/eval improvement from 50M `{payload['gate_75m']['validation_improvement_percent']:.3f}%` / `{payload['gate_75m']['eval_improvement_percent']:.3f}%`.",
        f"- Total 25M-to-75M improvement: validation `{payload['validation_improvement_from_25m_percent']:.3f}%`; eval `{payload['eval_improvement_from_25m_percent']:.3f}%`.",
        "- Rebound was 0%, no catastrophic probe regression occurred, integrity passed, and generation hard failures were zero.",
        "- The 75M gate required at least 3% validation and eval improvement from 50M. Both landed in the 1%-3% conditional-stop band.",
        "",
        "## Progression",
        "",
        "| Step | Tokens | LR | Train | Validation | Eval | Perplexity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in PROGRESSION_STEPS:
        item = progression[str(step)]
        stage_lines.append(
            f"| {step:,} | {item['tokens']:,} | {item['learning_rate']:.12f} | {item['train'].get('loss', float('nan')):.6f} | "
            f"{item['validation']['loss']:.6f} | {item['eval']['loss']:.6f} | {item['eval']['perplexity']:.3f} |"
        )
    stage_lines.extend(
        [
            "",
            "## Exact stop",
            "",
            f"- Final completed optimizer step: `{payload['stopped_optimizer_step']:,}`.",
            f"- Consumed tokens: `{payload['stopped_tokens']:,}`; sequence index: `{payload['stopped_sequence_index']:,}`.",
            f"- Unconsumed full-batch capacity: `{payload['remaining_full_batch_tokens']:,}` tokens / `{payload['remaining_full_batch_sequences']:,}` sequences.",
            f"- Immutable tail outside a full 16-sequence optimizer batch: `{payload['unusable_tail_sequences']}` sequences.",
            "- No step 9156, final segment, second epoch, SFT, Qwen generation, or upload occurred.",
            "",
            "## Checkpoint",
            "",
            f"- Full-resume checkpoint: `{payload['final_checkpoint']}`.",
            f"- Model SHA-256: `{payload['final_checkpoint_hashes']['model_sha256']}`.",
            f"- Resume-state SHA-256: `{payload['final_checkpoint_hashes']['resume_state_sha256']}`.",
            "",
            "The model remains a base model: not instruction-tuned, not a chatbot, not production-ready, and not approved for public upload.",
            "",
        ]
    )
    STAGE_REPORT.write_text("\n".join(stage_lines), encoding="utf-8")

    health = payload["training_health"]
    activation_lines = []
    for step in PROGRESSION_STEPS:
        diagnostic = load_json(Path(load_json(RUN_DIR / "run_manifest.json")["diagnostic_snapshots"][str(step)]))
        activation = diagnostic["activation"]
        activation_lines.append(
            f"| {step:,} | {activation['logits']['std']:.6f} | {activation['logits']['softmax_entropy_mean']:.6f} | "
            f"{diagnostic['embedding_norm']['mean']:.6f} | {activation['final_to_early_residual_ratio']:.6f} |"
        )
    HEALTH_REPORT.write_text(
        "\n".join(
            [
                "# DarkMind v2 Phase 4E Training Health",
                "",
                f"Recorded Stage-3 optimizer steps: `{health['steps_recorded']:,}`.",
                f"Pre-clipping gradient p50/p95/max: `{health['gradient_p50']:.6f}` / `{health['gradient_p95']:.6f}` / `{health['gradient_max']:.6f}`.",
                f"Clipped-step fraction: `{health['clipped_step_fraction']:.3%}`.",
                f"Update-to-weight p50/p95/max: `{health['update_to_weight_p50']:.9f}` / `{health['update_to_weight_p95']:.9f}` / `{health['update_to_weight_max']:.9f}`.",
                f"Non-finite events: `{health['non_finite_events']}`.",
                "",
                "Clipping remained effectively 100%. Losses improved and gradient p95 stayed below twice the Phase 4D baseline, so the hard health stop did not fire; the frequency remains a material concern in the Conditional PASS decision.",
                "",
                "| Step | Logit std | Prediction entropy | Embedding norm | Final/early residual RMS |",
                "|---:|---:|---:|---:|---:|",
                *activation_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )

    generation = payload["generation"]
    sample_lines = [
        "# DarkMind v2 Phase 4E Controlled Samples",
        "",
        "Raw outputs are retained only in the external Phase 4E runtime. This source-controlled report contains aggregate diagnostics and no copyrighted passages.",
        "",
        "| Checkpoint | Mode | Count | Repetition | Exact loops | EOS | Empty | Unique ratio | Hard failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("25M authoritative", generation["baseline_25m"]), ("50M subset", generation["subset_50m"]), ("75M subset", generation["subset_75m"])):
        for mode in ("greedy", "sampling"):
            summary = item[mode]
            sample_lines.append(
                f"| {label} | {mode} | {summary['generations']} | {_rate(summary, 'repetition'):.1%} | {_rate(summary):.1%} | "
                f"{summary['eos_completion_rate']:.1%} | {summary['empty_output_count']} | {summary['mean_unique_token_ratio']:.3f} | {summary['hard_failure_total']} |"
            )
    sample_lines.extend(["", "The 50M and 75M rows use controlled 16-prompt subsets and are directional, not authoritative quality estimates. The final 200/500 audit was not run because the final segment was not authorized.", ""])
    SAMPLES_REPORT.write_text("\n".join(sample_lines), encoding="utf-8")

    MEMORIZATION_REPORT.write_text(
        "\n".join(
            [
                "# DarkMind v2 Phase 4E Memorization Audit",
                "",
                "Status: **NOT RUN**.",
                "",
                "The controlled memorization and extraction-risk audit is defined for the final no-wrap checkpoint. Phase 4E stopped at the 75M conditional gate, so no final checkpoint existed and the audit was not executed.",
                "",
                "PII extraction findings: **NOT ASSESSED**. Memorized long-span findings: **NOT ASSESSED**. This report does not claim extraction risk is zero and does not clear any public-release blocker.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    primary = {name: payload["probe_comparison"][name] for name in ("turkish_prose", "english_prose", "turkish_technical", "english_technical")}
    quality_lines = [
        "# DarkMind v2 Phase 4E Base Quality Review",
        "",
        "Decision: **CONDITIONAL PASS - STOP FOR REVIEW**.",
        "",
        "| Probe | 25M loss | 75M loss | Improvement | Catastrophic regression |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in primary.items():
        quality_lines.append(
            f"| {name} | {item['baseline_loss']:.6f} | {item['final_loss']:.6f} | {item['improvement_percent']:.3f}% | {'YES' if item['catastrophic_regression'] else 'NO'} |"
        )
    quality_lines.extend(
        [
            "",
            "A second epoch is not justified or authorized. The current Corpus V3 first pass is incomplete because the marginal 50M-to-75M gain missed the 3% gate.",
            "",
            "Recommended next step: review the late-stage learning-rate/clipping interaction and plan a licensed Corpus V3 expansion before deciding whether to authorize any further base pretraining. Do not begin SFT automatically.",
            "",
            "No local export was created: the final checkpoint, final 200/500 generation audit, memorization audit, and Strong PASS release gates were not reached.",
            "",
            "DARKMIND V2 BASE V1 FIRST CORPUS PASS REQUIRES REVIEW BEFORE FURTHER PRETRAINING",
            "",
        ]
    )
    QUALITY_REPORT.write_text("\n".join(quality_lines), encoding="utf-8")


def finalize_conditional() -> dict[str, Any]:
    payload = build_assessment()
    render_reports(payload)
    payload["source_reports"] = [str(path) for path in (STAGE_REPORT, SAMPLES_REPORT, HEALTH_REPORT, MEMORIZATION_REPORT, QUALITY_REPORT)]
    payload["runtime_bytes_after_reports"] = directory_size(RUNTIME_ROOT)
    atomic_write_json(RUN_DIR / "conditional_stop_assessment.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize-conditional",))
    args = parser.parse_args()
    payload = finalize_conditional()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
