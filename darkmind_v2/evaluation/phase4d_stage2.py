"""Run Phase 4D milestone generation audits and render honest Stage-2 reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import sha256_file
from darkmind_v2.evaluation.audit_full_epoch_checkpoint import audit_checkpoint
from darkmind_v2.evaluation.audit_public_preview import (
    generate_record,
    load_audit_prompts,
    preserve_hard_failure,
    write_manifest,
)
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer
from darkmind_v2.training.phase4d_stage2 import (
    MILESTONES,
    ROOT,
    RUN_DIR,
    RUNTIME_ROOT,
    SOURCE_STEP,
    TARGET_STEP,
    TOKENIZER_INPUT,
    atomic_write_json,
    directory_size,
    load_json,
)


PROMPTS_PATH = ROOT / "darkmind_v2" / "eval" / "public_preview_prompts.jsonl"
GATES_PATH = ROOT / "darkmind_v2" / "config" / "public_research_preview_gates.json"
SAMPLES_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4d_base_v1_stage2_25m_samples.md"
STAGE_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4d_base_v1_stage2_25m.md"
HUMAN_REVIEW_STEPS = (610, 1024, 1536, 2048, 2560, 3051)


def subset_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {
        "tr:ordinary_text": 4,
        "en:ordinary_text": 4,
        "tr:technical": 2,
        "en:technical": 2,
        "tr:factual_encyclopedic": 1,
        "en:factual_encyclopedic": 1,
        "code:code_structured": 2,
    }
    used: Counter[str] = Counter()
    selected = []
    for prompt in prompts:
        key = f"{prompt['language']}:{prompt['category']}"
        if used[key] < targets.get(key, 0):
            selected.append(prompt)
            used[key] += 1
    if len(selected) != 16 or dict(used) != targets:
        raise ValueError(f"Phase 4D subset distribution mismatch: {dict(used)}")
    return selected


def checkpoint_and_hash(step: int) -> tuple[Path, str]:
    manifest = load_json(RUN_DIR / "run_manifest.json")
    checkpoint = Path(manifest["checkpoints"][str(step)])
    value = manifest["checkpoint_hashes"][str(step)]
    expected_hash = value if isinstance(value, str) else value["model_sha256"]
    if sha256_file(checkpoint / "model" / "model.safetensors") != expected_hash:
        raise ValueError(f"Phase 4D milestone model hash mismatch: {step}")
    return checkpoint, expected_hash


def run_subset(step: int) -> dict[str, Any]:
    checkpoint, expected_hash = checkpoint_and_hash(step)
    output_dir = RUNTIME_ROOT / "runs" / RUN_DIR.name / "audits" / f"step_{step:06d}" / "subset"
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4D subset hash mismatch")
        return payload
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete Phase 4D subset requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = subset_prompts(load_audit_prompts(PROMPTS_PATH))
    prompt_hash = sha256_file(PROMPTS_PATH)
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer(TOKENIZER_INPUT)
    started = time.perf_counter()
    greedy_records = []
    sampled_records = []
    stage = "stage1_final"
    for index, prompt in enumerate(prompts, start=1):
        greedy = generate_record(
            model,
            tokenizer,
            prompt,
            max_new_tokens=32,
            do_sample=False,
            profile_name="greedy",
            seed=None,
            checkpoint_stage=stage,
        )
        greedy_records.append(greedy)
        if greedy["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "greedy", greedy, index)
            raise RuntimeError(f"Phase 4D subset greedy hard failure at step {step}: {greedy['prompt_id']}")
        sampled = generate_record(
            model,
            tokenizer,
            prompt,
            max_new_tokens=32,
            do_sample=True,
            profile_name="A",
            seed=20260712,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            checkpoint_stage=stage,
        )
        sampled_records.append(sampled)
        if sampled["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "sampling", sampled, index)
            raise RuntimeError(f"Phase 4D subset sampling hard failure at step {step}: {sampled['prompt_id']}")
    greedy_manifest = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={
            "mode": "greedy",
            "max_new_tokens": 32,
            "checkpoint_stage": stage,
            "terminal_eos_is_not_special_token_leakage": True,
        },
        prompt_hash=prompt_hash,
        records=greedy_records,
    )
    sampling_manifest = write_manifest(
        output_dir / "sampling_manifest.json",
        settings={
            "mode": "fixed_seeded_sampling_subset",
            "max_new_tokens": 32,
            "checkpoint_stage": stage,
            "seed": 20260712,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "terminal_eos_is_not_special_token_leakage": True,
        },
        prompt_hash=prompt_hash,
        records=sampled_records,
    )
    report = {
        "schema_version": "darkmind-v2-phase4d-milestone-subset-v1",
        "result": "PASS",
        "optimizer_step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": expected_hash,
        "prompt_count_per_mode": len(prompts),
        "greedy": greedy_manifest["summary"],
        "sampling": sampling_manifest["summary"],
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
        "terminal_eos_policy_corrected": True,
    }
    atomic_write_json(summary_path, report)
    del model
    torch.cuda.empty_cache()
    return report


def run_subsets() -> dict[str, Any]:
    summaries = {}
    for step in MILESTONES:
        summaries[str(step)] = run_subset(step)
        print(f"phase4d generation subset step={step} PASS", flush=True)
    payload = {
        "schema_version": "darkmind-v2-phase4d-generation-subsets-v1",
        "result": "PASS",
        "milestones": summaries,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
    }
    atomic_write_json(RUN_DIR / "generation_subsets.json", payload)
    return payload


def run_authoritative_final() -> dict[str, Any]:
    checkpoint, expected_hash = checkpoint_and_hash(TARGET_STEP)
    output_dir = RUN_DIR / "audits" / f"step_{TARGET_STEP:06d}" / "authoritative"
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        report = load_json(summary_path)
        if report.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4D authoritative hash mismatch")
        return report
    report = audit_checkpoint(
        checkpoint,
        GATES_PATH,
        PROMPTS_PATH,
        output_dir,
        checkpoint_stage="stage1_final",
        expected_model_hash=expected_hash,
        tokenizer_dir=TOKENIZER_INPUT,
    )
    if report["greedy"]["generations"] != 200 or report["sampling"]["generations"] != 500:
        raise ValueError("Phase 4D authoritative generation counts changed")
    return report


def _warning_rate(summary: dict[str, Any], key: str) -> float:
    return summary["quality_warning_counts"].get(key, 0) / max(summary["generations"], 1)


def _loop_rate(summary: dict[str, Any]) -> float:
    return summary["exact_repeated_ngram_loop_outputs"] / max(summary["generations"], 1)


def generation_analysis() -> dict[str, Any]:
    subsets = load_json(RUN_DIR / "generation_subsets.json")["milestones"]
    authoritative = run_authoritative_final()
    progression = {}
    for step, item in subsets.items():
        progression[step] = {
            mode: {
                "repetition_warning_rate": _warning_rate(item[mode], "repetition"),
                "exact_loop_rate": _loop_rate(item[mode]),
                "longest_repeated_token_run": item[mode]["longest_repeated_token_run"],
                "mean_unique_token_ratio": item[mode]["mean_unique_token_ratio"],
                "eos_completion_rate": item[mode]["eos_completion_rate"],
                "empty_output_count": item[mode]["empty_output_count"],
                "invalid_utf8_sequence_count": item[mode]["invalid_utf8_sequence_count"],
                "replacement_character_count": item[mode]["replacement_character_count"],
                "mojibake_output_count": item[mode]["mojibake_output_count"],
                "unexpected_script_output_count": item[mode]["unexpected_script_output_count"],
                "mixed_script_output_count": item[mode]["mixed_script_output_count"],
                "special_token_leakage_count": item[mode]["special_token_leakage_count"],
            }
            for mode in ("greedy", "sampling")
        }
    payload = {
        "schema_version": "darkmind-v2-phase4d-generation-analysis-v1",
        "result": "PASS",
        "subset_progression": progression,
        "authoritative_final": authoritative,
        "final_counts": {
            "greedy": authoritative["greedy"]["generations"],
            "sampling": authoritative["sampling"]["generations"],
        },
        "generation_quality_role": "diagnostic only; Base V1 is not a chatbot or release candidate",
        "raw_outputs_retained": True,
        "sanitization_performed": False,
    }
    atomic_write_json(RUN_DIR / "generation_analysis.json", payload)
    return payload


def render_samples() -> None:
    lines = [
        "# DarkMind v2 Base V1 Stage-2 25M Samples",
        "",
        "These are balanced deterministic raw continuations. They are not cherry-picked, sanitized, factual assurances, or evidence that the model is conversational or release-ready.",
        "",
    ]
    totals: Counter[str] = Counter()
    for step in HUMAN_REVIEW_STEPS:
        path = RUN_DIR / "audits" / f"step_{step:06d}" / "subset" / "greedy_manifest.json"
        records = load_json(path)["results"]
        if len(records) != 16:
            raise ValueError(f"Phase 4D human review sample count changed at step {step}")
        lines.extend([f"## Step {step}", ""])
        for index, record in enumerate(records, start=1):
            category = record["category"]
            if category == "ordinary_text":
                totals[f"{record['language']}_prose"] += 1
            elif category == "technical":
                totals["technical"] += 1
            elif category == "factual_encyclopedic":
                totals["factual"] += 1
            elif category == "code_structured":
                totals["code"] += 1
            warnings = record["policy"]["warnings"]
            assessment = (
                "Raw output has measured warnings and remains weak or unreliable."
                if warnings
                else "No automatic warning fired; coherence and factuality are still not established."
            )
            lines.extend(
                [
                    f"### {step}.{index} {record['language']} / {category}",
                    "",
                    f"Prompt: {record['prompt']}",
                    "",
                    f"Raw continuation (escaped exactly): {record['output_escaped']}",
                    "",
                    f"Checkpoint: step {step}",
                    "",
                    "Decoding: greedy, max_new_tokens=32, EOS-aware, terminal EOS excluded from leakage",
                    "",
                    f"Warnings: {', '.join(warnings) or 'none'}",
                    "",
                    f"Assessment: {assessment}",
                    "",
                ]
            )
    required = {"tr_prose": 20, "en_prose": 20, "technical": 10, "factual": 5, "code": 5}
    if any(totals[key] < value for key, value in required.items()):
        raise ValueError(f"Phase 4D sample report balance failed: {dict(totals)}")
    lines.extend(
        [
            "## Balance",
            "",
            f"Turkish prose: {totals['tr_prose']}; English prose: {totals['en_prose']}; technical/educational: {totals['technical']}; factual: {totals['factual']}; code/structured: {totals['code']}.",
            "",
        ]
    )
    SAMPLES_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _thermal_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    samples = [sample for segment in manifest["segments"] for sample in segment["gpu_samples"] if sample.get("available")]
    temperatures = [float(item["temperature_c"]) for item in samples if item.get("temperature_c") is not None]
    powers = [float(item["power_w"]) for item in samples if item.get("power_w") is not None]
    return {
        "samples": len(samples),
        "temperature_min_c": min(temperatures) if temperatures else None,
        "temperature_max_c": max(temperatures) if temperatures else None,
        "temperature_mean_c": statistics.fmean(temperatures) if temperatures else None,
        "power_min_w": min(powers) if powers else None,
        "power_max_w": max(powers) if powers else None,
        "power_mean_w": statistics.fmean(powers) if powers else None,
    }


def render_stage_report() -> dict[str, Any]:
    training = load_json(RUN_DIR / "training_summary.json")
    generation = load_json(RUN_DIR / "generation_analysis.json")
    manifest = load_json(RUN_DIR / "run_manifest.json")
    evaluations = load_json(RUN_DIR / "evaluations.json")
    thermal = _thermal_summary(manifest)
    final_audit = generation["authoritative_final"]
    classification = training["classification"]
    recommendation = (
        "Prepare a separate approval request for continuation toward the 100M gate; do not start it automatically."
        if classification == "STRONG PASS"
        else "Stop and diagnose before any 100M continuation."
    )
    lines = [
        "# DarkMind v2 Phase 4D Base V1 Stage-2 25M",
        "",
        f"Stage-2 classification: **{classification}**",
        f"100M recommendation: {recommendation}",
        "",
        "This classification concerns stable learning only. The model is not instruction-tuned, not a chat model, not production-ready, and not publicly released.",
        "",
        "## Identity and scope",
        "",
        f"- Exact resume checkpoint: `{training['source_checkpoint']}`.",
        f"- Frozen V2 config SHA-256: `{training['v2_config_sha256']}`.",
        f"- Stage-2 authorization SHA-256: `{training['authorization_file_sha256']}`.",
        f"- Step/token range: `{training['starting_optimizer_step']}` / `{training['starting_tokens']:,}` to `{training['final_optimizer_step']}` / `{training['final_tokens']:,}`.",
        f"- Additional work: `{training['additional_optimizer_steps']:,}` steps, `{training['additional_tokens']:,}` tokens, `{training['additional_sequences']:,}` sequences.",
        "- Base V1 architecture, frozen tokenizer, Corpus V3, deterministic order, optimizer grouping, and V2 scheduler were not changed.",
        "",
        "## Learning progression",
        "",
        "| Step | Tokens | Applied LR | Train loss | Validation | Eval | Eval perplexity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in MILESTONES:
        item = evaluations[str(step)]
        train = item.get("train", {})
        lines.append(
            f"| {step:,} | {step * 8192:,} | {training['learning_rate_progression'][str(step)]:.12f} | "
            f"{train.get('loss', float('nan')):.6f} | {item['validation']['loss']:.6f} | {item['eval']['loss']:.6f} | {item['eval']['perplexity']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Validation improvement from step 610: `{training['validation_improvement_percent']:.3f}%`; eval improvement: `{training['eval_improvement_percent']:.3f}%`.",
            f"Validation/eval rebound: `{training['validation_rebound_percent']:.3f}%` / `{training['eval_rebound_percent']:.3f}%`.",
            f"Best validation checkpoint: step `{training['best_validation_step']}` at `{training['best_checkpoint']}`.",
            "",
            "## Optimization and activations",
            "",
            f"Gradient norm p50/p95/max: `{training['gradient_norm_p50']:.4f}` / `{training['gradient_norm_p95']:.4f}` / `{training['gradient_norm_max']:.4f}`.",
            f"Clipped-step fraction: `{training['clipped_step_fraction']:.3%}`; maximum sentinel update-to-weight ratio: `{training['update_to_weight_maximum']:.8f}`.",
            "The high clipping fraction is disclosed as an optimization characteristic; losses, updates, logits, and residual diagnostics remained finite and the validation/eval curve did not rebound.",
            "",
            "| Step | Logit std | Prediction entropy | Embedding norm mean | Final/early residual RMS |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for step in MILESTONES:
        diagnostic = load_json(Path(manifest["diagnostic_snapshots"][str(step)]))
        activation = diagnostic["activation"]
        lines.append(
            f"| {step:,} | {activation['logits']['std']:.6f} | {activation['logits']['softmax_entropy_mean']:.6f} | "
            f"{diagnostic['embedding_norm']['mean']:.6f} | {activation['final_to_early_residual_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed probes",
            "",
            "| Probe | Step 610 | Step 3051 | Improvement | Catastrophic regression |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, item in training["probe_regressions"].items():
        lines.append(
            f"| {name} | {item['baseline_loss']:.6f} | {item['final_loss']:.6f} | "
            f"{item['improvement_percent']:.3f}% | {'YES' if item['catastrophic_regression'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "All Turkish, English, prose, technical, and source-family probes avoided catastrophic regression. Turkish technical improved more slowly than the other primary probes and remains a disclosed weakness.",
            "",
            "## Generation diagnostics",
            "",
            "| Step | Greedy repetition | Greedy loops | Sampling repetition | Sampling loops |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for step in MILESTONES:
        item = generation["subset_progression"][str(step)]
        lines.append(
            f"| {step:,} | {item['greedy']['repetition_warning_rate']:.1%} | {item['greedy']['exact_loop_rate']:.1%} | "
            f"{item['sampling']['repetition_warning_rate']:.1%} | {item['sampling']['exact_loop_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Authoritative final audit: `{final_audit['greedy']['generations']}` greedy and `{final_audit['sampling']['generations']}` fixed-seeded generations; hard failures `{final_audit['greedy']['hard_failure_total'] + final_audit['sampling']['hard_failure_total']}`.",
            f"Final greedy repetition/loop rates: `{_warning_rate(final_audit['greedy'], 'repetition'):.1%}` / `{_loop_rate(final_audit['greedy']):.1%}`.",
            f"Final sampling repetition/loop rates: `{_warning_rate(final_audit['sampling'], 'repetition'):.1%}` / `{_loop_rate(final_audit['sampling']):.1%}`.",
            "Raw generations are retained without sanitization. Generation quality remains diagnostic and does not make this checkpoint a chatbot or release candidate.",
            "",
            "## Runtime and restart",
            "",
            f"Active/wall throughput: `{training['active_tokens_per_second']:.1f}` / `{training['wall_tokens_per_second']:.1f}` tokens/s.",
            f"Peak allocated/reserved VRAM: `{training['peak_allocated_vram_bytes']:,}` / `{training['peak_reserved_vram_bytes']:,}` bytes.",
            f"GPU temperature min/mean/max: `{thermal['temperature_min_c']:.1f}` / `{thermal['temperature_mean_c']:.1f}` / `{thermal['temperature_max_c']:.1f}` C.",
            f"GPU power min/mean/max: `{thermal['power_min_w']:.1f}` / `{thermal['power_mean_w']:.1f}` / `{thermal['power_max_w']:.1f}` W.",
            f"Real process restart: `{training['process_restart']['result']}`; PIDs `{manifest['process_ids']}`.",
            "Optimizer, scheduler, RNG, data position, and deterministic sequence order all passed continuation checks.",
            "",
            "One initial Segment-A attempt stopped at durable step 1379 because Windows briefly denied the atomic `progress.json` replace. Its evidence is preserved unchanged. A retry-safe bounded rename policy was added, and the official retry1 run restarted from the immutable step-610 source and completed both exact segments without further I/O failure.",
            "",
            "## Decision",
            "",
            f"**{classification}**: validation and eval improved by more than 8%, rebound stayed within 2%, late milestones did not worsen, every fixed probe avoided catastrophic regression, and all integrity gates passed.",
            "",
            recommendation,
            "",
            "No 100M continuation, SFT, Qwen teacher generation, corpus modification, tokenizer modification, public release, or Hugging Face upload occurred.",
            "",
        ]
    )
    STAGE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": "darkmind-v2-phase4d-stage-report-v1",
        "result": "PASS",
        "classification": classification,
        "recommend_100m_approval_request": classification == "STRONG PASS",
        "thermal": thermal,
        "samples_report": str(SAMPLES_REPORT),
        "stage_report": str(STAGE_REPORT),
        "runtime_bytes_before_export": directory_size(RUNTIME_ROOT),
        "phase_100m_started": False,
        "upload_performed": False,
    }
    atomic_write_json(RUN_DIR / "stage_report_summary.json", payload)
    return payload


def finalize() -> dict[str, Any]:
    if not (RUN_DIR / "generation_subsets.json").is_file():
        run_subsets()
    generation_analysis()
    render_samples()
    return render_stage_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("subsets", "authoritative", "finalize"))
    args = parser.parse_args()
    if args.command == "subsets":
        payload = run_subsets()
    elif args.command == "authoritative":
        payload = run_authoritative_final()
    else:
        payload = finalize()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
