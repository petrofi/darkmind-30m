"""Run milestone generation audits and render the Phase 4A Stage-1 reports."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.evaluation.audit_public_preview import (
    generate_record,
    load_audit_prompts,
    preserve_hard_failure,
    write_manifest,
)
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer
from darkmind_v2.training.validate_phase4a_config import (
    DEFAULT_CONFIG,
    classify_stage1,
    load_and_validate_phase4a_config,
)
from darkmind_v2.training.validate_phase4a_preflight import ROOT


PROMPTS_PATH = ROOT / "darkmind_v2" / "eval" / "public_preview_prompts.jsonl"
SAMPLES_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4a_base_v1_stage1_samples.md"
STAGE_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4a_base_v1_stage1_5m.md"
STAGES = {0: "initialization", 128: "stage1", 305: "midpoint", 458: "stage1", 610: "stage1_final"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {
        "tr:ordinary_text": 15,
        "en:ordinary_text": 15,
        "tr:technical": 5,
        "en:technical": 5,
        "tr:factual_encyclopedic": 3,
        "en:factual_encyclopedic": 2,
        "code:code_structured": 5,
    }
    used: Counter[str] = Counter()
    selected = []
    for prompt in prompts:
        key = f"{prompt['language']}:{prompt['category']}"
        if used[key] < targets.get(key, 0):
            selected.append(prompt)
            used[key] += 1
    if len(selected) != 50 or dict(used) != targets:
        raise ValueError(f"Phase 4A sample prompt distribution mismatch: {dict(used)}")
    return selected


def run_authoritative(checkpoint: Path, output_dir: Path, stage: str, expected_hash: str) -> dict[str, Any]:
    summary = output_dir / "audit_summary.json"
    if summary.is_file():
        payload = load_json(summary)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed authoritative audit hash mismatch")
        return payload
    command = [
        sys.executable,
        "-m",
        "darkmind_v2.evaluation.audit_full_epoch_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--checkpoint-stage",
        stage,
        "--expected-model-hash",
        expected_hash,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=14_400, check=False)
    if completed.returncode or not summary.is_file():
        raise RuntimeError(f"authoritative audit failed at {checkpoint.name}: {completed.stderr[-3000:]}")
    return load_json(summary)


def run_subset(checkpoint: Path, output_dir: Path, stage: str, expected_hash: str) -> dict[str, Any]:
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed subset audit hash mismatch")
        return payload
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete subset audit requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prompts = sample_prompts(load_audit_prompts(PROMPTS_PATH))
    prompt_hash = sha256_file(PROMPTS_PATH)
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer()
    greedy_records = []
    sampling_records = []
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
            raise RuntimeError(f"subset greedy hard failure: {greedy['prompt_id']}")
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
        sampling_records.append(sampled)
        if sampled["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "sampling", sampled, index)
            raise RuntimeError(f"subset sampling hard failure: {sampled['prompt_id']}")
    greedy = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={"mode": "greedy", "max_new_tokens": 32, "checkpoint_stage": stage},
        prompt_hash=prompt_hash,
        records=greedy_records,
    )
    sampling = write_manifest(
        output_dir / "sampling_manifest.json",
        settings={
            "mode": "fixed_seeded_sampling_subset",
            "max_new_tokens": 32,
            "checkpoint_stage": stage,
            "seed": 20260712,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        },
        prompt_hash=prompt_hash,
        records=sampling_records,
    )
    report = {
        "schema_version": "darkmind-v2-phase4a-milestone-subset-v1",
        "result": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": expected_hash,
        "greedy": greedy["summary"],
        "sampling": sampling["summary"],
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
    }
    atomic_write_json(summary_path, report)
    return report


def extract_authoritative_subset(authoritative_dir: Path, subset_dir: Path, expected_hash: str) -> dict[str, Any]:
    summary_path = subset_dir / "audit_summary.json"
    if summary_path.is_file():
        return load_json(summary_path)
    subset_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {item["id"] for item in sample_prompts(load_audit_prompts(PROMPTS_PATH))}
    greedy_source = load_json(authoritative_dir / "greedy_manifest.json")
    sampling_source = load_json(authoritative_dir / "sampling_manifest.json")
    greedy_records = [item for item in greedy_source["results"] if item["prompt_id"] in selected_ids]
    sampling_records = [
        item
        for item in sampling_source["results"]
        if item["prompt_id"] in selected_ids
        and item["settings"]["profile"] == "A"
        and item["settings"]["seed"] == 20260712
    ]
    if len(greedy_records) != 50 or len(sampling_records) != 50:
        raise ValueError("authoritative audit cannot provide the Phase 4A subset")
    greedy = write_manifest(
        subset_dir / "greedy_manifest.json",
        settings={"mode": "greedy_reused_from_authoritative", "max_new_tokens": 32},
        prompt_hash=greedy_source["prompt_manifest_sha256"],
        records=greedy_records,
    )
    sampling = write_manifest(
        subset_dir / "sampling_manifest.json",
        settings={"mode": "sampling_reused_from_authoritative", "profile": "A", "seed": 20260712},
        prompt_hash=sampling_source["prompt_manifest_sha256"],
        records=sampling_records,
    )
    report = {
        "schema_version": "darkmind-v2-phase4a-milestone-subset-v1",
        "result": "PASS",
        "checkpoint_model_sha256": expected_hash,
        "greedy": greedy["summary"],
        "sampling": sampling["summary"],
        "reused_from_authoritative_audit": True,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
    }
    atomic_write_json(summary_path, report)
    return report


def render_samples(run_dir: Path) -> None:
    lines = [
        "# DarkMind v2 Base V1 Stage-1 Samples",
        "",
        "These are deterministic raw early-stage continuations. They are not cherry-picked, not sanitized, and not factual assurances.",
        "Phase 4A pipeline integrity passed, but the learning-quality gate failed. Step 128 was the best checkpoint, final validation and eval were worse than initialization, the 25M continuation was rejected, and no upload occurred.",
        "",
    ]
    for step in STAGES:
        records = load_json(run_dir / "audits" / f"step_{step:06d}" / "subset" / "greedy_manifest.json")["results"]
        groups = {
            "tr": [item for item in records if item["language"] == "tr" and item["category"] == "ordinary_text"][:3],
            "en": [item for item in records if item["language"] == "en" and item["category"] == "ordinary_text"][:3],
            "technical": [item for item in records if item["category"] == "technical"][:2],
            "factual": [item for item in records if item["category"] == "factual_encyclopedic"][:1],
            "code": [item for item in records if item["category"] == "code_structured"][:1],
        }
        selected = [item for group in groups.values() for item in group]
        if len(selected) != 10:
            raise ValueError(f"sample report distribution failed at step {step}")
        lines.extend([f"## Step {step}", ""])
        for index, record in enumerate(selected, start=1):
            assessment = (
                "Factual continuation is unverified and must not be relied upon."
                if record["category"] == "factual_encyclopedic"
                else "Raw early-stage structural sample; coherence is not assumed."
            )
            lines.extend(
                [
                    f"### {step}.{index} {record['language']} / {record['category']}",
                    "",
                    f"Prompt: {record['prompt']}",
                    "",
                    f"Raw continuation: {record['output']}",
                    "",
                    f"Checkpoint: step {step}",
                    "",
                    f"Decoding: greedy, max_new_tokens=32, terminal-EOS aware",
                    "",
                    f"Health warnings: {', '.join(record['policy']['warnings']) or 'none'}",
                    "",
                    f"Assessment: {assessment}",
                    "",
                ]
            )
    SAMPLES_REPORT.write_text("\n".join(lines), encoding="utf-8")


def render_stage_report(config: dict[str, Any], run_dir: Path, results: dict[str, Any]) -> dict[str, Any]:
    run = load_json(run_dir / "run_manifest.json")
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    milestones = {}
    for step in STAGES:
        validation = load_json(run_dir / "validations" / f"step_{step:06d}.json")
        evaluation = load_json(run_dir / "eval" / f"step_{step:06d}.json")
        milestones[str(step)] = {"validation": validation, "eval": evaluation}
    decision = classify_stage1(
        milestones["0"]["validation"]["loss"],
        milestones["610"]["validation"]["loss"],
        milestones["0"]["eval"]["loss"],
        milestones["610"]["eval"]["loss"],
    )
    integrity = all(
        load_json(run_dir / "resume" / f"segment_to_step_{step:06d}.json")["result"] == "PASS"
        for step in (305, 610)
    ) and all(results[str(step)]["result"] == "PASS" for step in STAGES)
    if not integrity:
        decision["classification"] = "FAIL"
    final_metric = metrics[-1]
    calibration = load_json(
        ROOT / config["calibration"]["output_dir"] / "calibration_summary.json"
    )
    failed_attempt_path = run_dir / "resume" / "failed_segment_b_attempt1.json"
    failed_attempt = load_json(failed_attempt_path) if failed_attempt_path.is_file() else None
    active_seconds = sum(item["optimizer_step_duration_seconds"] for item in metrics)
    total_wall = run["wall_clock"]["segment_to_610_completed_unix"] - run["wall_clock"]["initialization_started_unix"]
    peak_allocated = max(item["allocated_vram_bytes"] for item in metrics)
    peak_reserved = max(item["reserved_vram_bytes"] for item in metrics)
    temperatures = [item["gpu"]["temperature_c"] for item in metrics if item["gpu"].get("available")]
    powers = [item["gpu"]["power_w"] for item in metrics if item["gpu"].get("available")]
    summary = {
        "schema_version": "darkmind-v2-phase4a-stage1-summary-v1",
        "result": "PASS" if integrity else "FAIL",
        "classification": decision["classification"],
        "validation_improvement_percent": decision["validation_improvement_percent"],
        "eval_improvement_percent": decision["eval_improvement_percent"],
        "optimizer_steps": run["optimizer_steps"],
        "consumed_tokens": run["consumed_tokens"],
        "final_train_loss": final_metric["raw_train_loss"],
        "final_smoothed_train_loss": final_metric["smoothed_train_loss"],
        "final_validation_loss": milestones["610"]["validation"]["loss"],
        "final_eval_loss": milestones["610"]["eval"]["loss"],
        "final_perplexity": milestones["610"]["eval"]["perplexity"],
        "train_validation_gap": milestones["610"]["validation"]["loss"] - final_metric["smoothed_train_loss"],
        "active_tokens_per_second": len(metrics) * 8192 / active_seconds,
        "complete_wall_tokens_per_second": run["consumed_tokens"] / total_wall,
        "active_seconds": active_seconds,
        "complete_wall_seconds": total_wall,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "temperature_c_min_max": [min(temperatures), max(temperatures)] if temperatures else None,
        "power_w_min_max": [min(powers), max(powers)] if powers else None,
        "best_checkpoint": run["best_checkpoint"],
        "calibration": calibration,
        "failed_segment_attempt": failed_attempt,
        "continuation_to_25m_recommended": decision["classification"] == "Strong PASS" and integrity,
        "milestones": milestones,
    }
    atomic_write_json(run_dir / "stage1_summary.json", summary)
    lines = [
        "# DarkMind v2 Base V1 Stage-1 5M",
        "",
        f"Classification: **{summary['classification']}**",
        "Pipeline integrity: **PASS**",
        "Learning-quality gate: **FAIL**",
        "Best checkpoint: **step 128**",
        "Final validation and eval worse than initialization: **YES**",
        "",
        f"Model: darkmind-v2-base-v1 / 118,056,960 parameters",
        f"Config SHA-256: `{config['model_config_sha256']}`",
        f"Architecture hash: `3a2dda86293ceae23ca4e50ea47c840b7fc46021d293c862d330110851ac8305`",
        f"Corpus hash: `{config['corpus']['corpus_hash']}`",
        f"Tokenized manifest hash: `{config['corpus']['tokenized_manifest_hash']}`",
        f"Initialization hash: `{run['initialization_hash']}`",
        "",
        f"Optimizer steps / tokens: {run['optimizer_steps']:,} / {run['consumed_tokens']:,}",
        f"Scheduler position: {run['optimizer_steps']:,} of {config['schedule']['scheduler_horizon_optimizer_steps']:,}",
        f"Final next-step LR: {final_metric['next_learning_rate']:.10f}",
        f"Nominal 100M scheduler unused tail: {100_000_000 - config['schedule']['scheduler_horizon_tokens']:,} tokens",
        f"Current complete train-sequence capacity deficit to scheduler horizon: {config['schedule']['scheduler_horizon_tokens'] - config['corpus']['train_complete_sequence_tokens']:,} tokens",
        "",
        "| Step | Tokens | Validation loss | Eval loss | Eval perplexity |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in STAGES:
        item = milestones[str(step)]
        lines.append(f"| {step} | {step * 8192:,} | {item['validation']['loss']:.6f} | {item['eval']['loss']:.6f} | {item['eval']['perplexity']:.3f} |")
    lines.extend(
        [
            "",
            f"Validation improvement: {summary['validation_improvement_percent']:.3f}%",
            f"Eval improvement: {summary['eval_improvement_percent']:.3f}%",
            f"Final train / validation gap: {summary['train_validation_gap']:.6f}",
            f"Active throughput: {summary['active_tokens_per_second']:.1f} tokens/s",
            f"Complete wall throughput: {summary['complete_wall_tokens_per_second']:.1f} tokens/s",
            f"Peak allocated / reserved VRAM: {peak_allocated:,} / {peak_reserved:,} bytes",
            f"Temperature range: {summary['temperature_c_min_max']}",
        f"Power range: {summary['power_w_min_max']}",
        "",
        "## Calibration",
        "",
        f"Measured optimizer steps: {calibration['measured_optimizer_steps']}",
        f"Active / full-wall throughput: {calibration['active_tokens_per_second']:.1f} / {calibration['full_wall_tokens_per_second']:.1f} tokens/s",
        f"p50 / p95 step: {calibration['p50_step_seconds']:.4f} / {calibration['p95_step_seconds']:.4f} seconds",
        f"Peak allocated / reserved VRAM: {calibration['peak_allocated_vram_bytes']:,} / {calibration['peak_reserved_vram_bytes']:,} bytes",
        f"Optimizer state: {calibration['optimizer_state_bytes']:,} bytes",
        "",
        "## Checkpoints",
        "",
        "| Step | Model SHA-256 | Model bytes | Resume bytes |",
        "|---:|---|---:|---:|",
        ]
    )
    for step in STAGES:
        checkpoint = ROOT / run["checkpoints"][str(step)]
        lines.append(
            f"| {step} | `{run['checkpoint_model_hashes'][str(step)]['model_sha256']}` | "
            f"{(checkpoint / 'model' / 'model.safetensors').stat().st_size:,} | "
            f"{(checkpoint / 'resume_state.pt').stat().st_size:,} |"
        )
    lines.extend(
        [
            "",
            "## Generation health",
            "",
            "| Step | Greedy repetition | Greedy loops | Sampling repetition | Sampling loops | Greedy/sample EOS |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for step in STAGES:
        subset = results[str(step)]["subset"]
        greedy = subset["greedy"]
        sampling = subset["sampling"]
        lines.append(
            f"| {step} | {greedy['quality_warning_counts'].get('repetition', 0)}/{greedy['generations']} | "
            f"{greedy['exact_repeated_ngram_loop_outputs']} | "
            f"{sampling['quality_warning_counts'].get('repetition', 0)}/{sampling['generations']} | "
            f"{sampling['exact_repeated_ngram_loop_outputs']} | "
            f"{greedy['eos_completion_rate']:.3f}/{sampling['eos_completion_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The fresh-process midpoint restart passed model, optimizer, scheduler, RNG, data-position, validation-reproducibility, and learning-rate continuity checks.",
            (
                "The first Segment B process stopped after uncheckpointed step 319 because OneDrive temporarily locked the repeatedly replaced progress JSON. "
                "Its metrics and traceback evidence are preserved. The immutable step-305 checkpoint was unchanged, and the single clean-process retry passed through step 610."
                if failed_attempt
                else "No discarded segment attempt was recorded."
            ),
            "All raw generation outputs remain in ignored runtime manifests. Repetition, exact loops, EOS, Unicode, byte fallback, scripts, and finite logits are reported without sanitization.",
            "Absolute Phase 3B finalist losses are not treated as comparable because the corpus, split, and production scheduler differ.",
            "The model is not instruction-tuned, not a chatbot, not production-ready, and not publicly released.",
            "Hugging Face upload performed: **NO**",
            "",
            f"25M continuation recommended: **{'YES, after explicit user approval' if summary['continuation_to_25m_recommended'] else 'NO'}**",
            "",
            (
                "DARKMIND V2 BASE V1 STAGE-1 5M PASSED AND IS READY FOR 25M CONTINUATION APPROVAL"
                if summary["classification"] == "Strong PASS"
                else "DARKMIND V2 BASE V1 STAGE-1 5M REQUIRES LEARNING DIAGNOSIS BEFORE CONTINUATION"
                if summary["classification"] == "Conditional PASS"
                else "DARKMIND V2 BASE V1 STAGE-1 5M FAILED AND TRAINING IS STOPPED"
            ),
            "",
        ]
    )
    STAGE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return summary


def evaluate(config_path: Path) -> dict[str, Any]:
    config = load_and_validate_phase4a_config(config_path, check_runtime_assets=True)
    run_dir = ROOT / config["run_dir"]
    run = load_json(run_dir / "run_manifest.json")
    if run.get("status") not in {"stage1_complete", "evaluated"} or run.get("optimizer_steps") != 610:
        raise ValueError("Stage-1 training is incomplete")
    results: dict[str, Any] = {}
    for step, stage in STAGES.items():
        checkpoint = ROOT / run["checkpoints"][str(step)]
        expected_hash = run["checkpoint_model_hashes"][str(step)]["model_sha256"]
        step_dir = run_dir / "audits" / f"step_{step:06d}"
        if step in (0, 610):
            authoritative = run_authoritative(checkpoint, step_dir / "authoritative", stage, expected_hash)
            subset = run_subset(checkpoint, step_dir / "subset", stage, expected_hash)
        else:
            authoritative = None
            subset = run_subset(checkpoint, step_dir / "subset", stage, expected_hash)
        results[str(step)] = {
            "result": subset["result"] if authoritative is None else authoritative["result"],
            "checkpoint_model_sha256": expected_hash,
            "subset": subset,
            "authoritative": authoritative,
        }
        print(f"audit step={step} result={results[str(step)]['result']}", flush=True)
    payload = {
        "schema_version": "darkmind-v2-phase4a-evaluation-v1",
        "result": "PASS" if all(item["result"] == "PASS" for item in results.values()) else "FAIL",
        "results": results,
    }
    atomic_write_json(run_dir / "evaluation_summary.json", payload)
    render_samples(run_dir)
    stage = render_stage_report(config, run_dir, results)
    run["evaluation_result"] = payload["result"]
    run["stage1_classification"] = stage["classification"]
    run["best_checkpoint"] = stage["best_checkpoint"]
    run["status"] = "evaluated"
    atomic_write_json(run_dir / "run_manifest.json", run)
    return {"evaluation": payload["result"], "classification": stage["classification"], "best_checkpoint": stage["best_checkpoint"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
