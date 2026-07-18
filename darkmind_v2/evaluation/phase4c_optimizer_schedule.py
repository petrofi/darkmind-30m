"""Evaluate Phase 4C diagnostic arms and immutable confirmation generations."""

from __future__ import annotations

import argparse
import json
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
from darkmind_v2.training.phase4c_diagnostics import ROOT, RUNTIME_ROOT, TOKENIZER_INPUT, atomic_write_json, ensure_runtime_path
from darkmind_v2.training.phase4c_training import MILESTONES, normalize_diagnostic_summary


PROMPTS_PATH = ROOT / "darkmind_v2" / "eval" / "public_preview_prompts.jsonl"
GATES_PATH = ROOT / "darkmind_v2" / "config" / "public_research_preview_gates.json"
ARM_NAMES = (
    "arm1_global_lr1e4_current_groups",
    "arm2_global_lr75e6_current_groups",
    "arm3_global_lr1e4_corrected_groups",
    "arm4_staged_decay_corrected_groups",
    "arm5_depth_scaled_init_staged",
)
CONFIRMATION_NAME = "base_v1_stage1_5m_v2_confirmation"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {
        "tr:ordinary_text": 2,
        "en:ordinary_text": 2,
        "tr:technical": 2,
        "en:technical": 2,
        "tr:factual_encyclopedic": 2,
        "en:factual_encyclopedic": 2,
        "code:code_structured": 2,
    }
    used: Counter[str] = Counter()
    selected = []
    for prompt in prompts:
        key = f"{prompt['language']}:{prompt['category']}"
        if used[key] < targets.get(key, 0):
            selected.append(prompt)
            used[key] += 1
    if len(selected) != 14 or dict(used) != targets:
        raise ValueError(f"Phase 4C subset prompt distribution mismatch: {dict(used)}")
    return selected


def run_subset(checkpoint: Path, output_dir: Path, stage: str, expected_hash: str) -> dict[str, Any]:
    output_dir = ensure_runtime_path(output_dir)
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = _load(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4C subset hash mismatch")
        return payload
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete Phase 4C subset requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prompts = subset_prompts(load_audit_prompts(PROMPTS_PATH))
    prompt_hash = sha256_file(PROMPTS_PATH)
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer(TOKENIZER_INPUT)
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
            raise RuntimeError(f"Phase 4C subset greedy hard failure: {greedy['prompt_id']}")
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
            raise RuntimeError(f"Phase 4C subset sampling hard failure: {sampled['prompt_id']}")
    greedy_manifest = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={"mode": "greedy", "max_new_tokens": 32, "checkpoint_stage": stage},
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
        },
        prompt_hash=prompt_hash,
        records=sampling_records,
    )
    report = {
        "schema_version": "darkmind-v2-phase4c-milestone-subset-v1",
        "result": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": expected_hash,
        "prompt_count_per_mode": len(prompts),
        "greedy": greedy_manifest["summary"],
        "sampling": sampling_manifest["summary"],
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
        "tokenizer_dir": str(TOKENIZER_INPUT),
    }
    atomic_write_json(summary_path, report)
    del model
    torch.cuda.empty_cache()
    return report


def _rate(summary: dict[str, Any], key: str) -> float:
    if key == "loops":
        return summary["exact_repeated_ngram_loop_outputs"] / summary["generations"]
    return summary["quality_warning_counts"].get("repetition", 0) / summary["generations"]


def evaluate_run(run_name: str, *, authoritative_final: bool) -> dict[str, Any]:
    run_dir = ensure_runtime_path(RUNTIME_ROOT / "runs" / run_name)
    training = _load(run_dir / "training_summary.json")
    subsets = {}
    for step in MILESTONES:
        checkpoint = Path(training["checkpoints"][str(step)])
        checkpoint_hash = training["checkpoint_hashes"][str(step)]
        stage = "initialization" if step == 0 else ("stage1_final" if step == 610 else "stage1")
        subsets[str(step)] = run_subset(
            checkpoint,
            run_dir / "audits" / f"step_{step:06d}" / "subset",
            stage,
            checkpoint_hash,
        )
        print(f"run={run_name} subset_step={step} result=PASS", flush=True)
    authoritative = None
    if authoritative_final:
        checkpoint = Path(training["checkpoints"]["610"])
        output_dir = run_dir / "audits" / "step_000610" / "authoritative"
        summary_path = output_dir / "audit_summary.json"
        authoritative = _load(summary_path) if summary_path.is_file() else audit_checkpoint(
            checkpoint,
            GATES_PATH,
            PROMPTS_PATH,
            output_dir,
            checkpoint_stage="stage1_final",
            expected_model_hash=training["checkpoint_hashes"]["610"],
            tokenizer_dir=TOKENIZER_INPUT,
        )
        torch.cuda.empty_cache()
    snapshot_payloads = {
        step: _load(Path(path)) for step, path in training["diagnostic_snapshots"].items()
    }
    normalized = normalize_diagnostic_summary(training, snapshot_payloads)
    normalized["original_stability"] = training["stability"]
    normalized["generation_subsets"] = subsets
    normalized["authoritative_final"] = authoritative
    normalized["generation_quality_is_diagnostic_at_5m"] = True
    normalized["generation_hard_failure"] = any(
        subset[mode]["hard_failure_total"]
        for subset in subsets.values()
        for mode in ("greedy", "sampling")
    ) > 0
    initial = subsets["0"]
    final = subsets["610"]
    normalized["generation_change"] = {
        mode: {
            "repetition_rate_delta": _rate(final[mode], "repetition") - _rate(initial[mode], "repetition"),
            "exact_loop_rate_delta": _rate(final[mode], "loops") - _rate(initial[mode], "loops"),
            "eos_rate_delta": final[mode]["eos_completion_rate"] - initial[mode]["eos_completion_rate"],
        }
        for mode in ("greedy", "sampling")
    }
    atomic_write_json(run_dir / "evaluation_summary.json", normalized)
    return normalized


def _mean_loss(summary: dict[str, Any]) -> float:
    return statistics.fmean((summary["final_validation_loss"], summary["final_eval_loss"]))


def select_policy(summaries: dict[str, dict[str, Any]]) -> str | None:
    stable = [name for name, item in summaries.items() if item["stability"] == "stable" and not item["generation_hard_failure"]]
    if not stable:
        return None
    best_mean = min(_mean_loss(summaries[name]) for name in stable)
    within_one_percent = [name for name in stable if _mean_loss(summaries[name]) <= best_mean * 1.01]
    return min(
        within_one_percent,
        key=lambda name: (
            summaries[name]["optimizer_grouping"] != "corrected_adamw_v1",
            summaries[name]["schedule"]["name"] != "warmup_cosine_global",
            summaries[name]["schedule"]["peak_learning_rate"],
            _mean_loss(summaries[name]),
        ),
    )


def analyze() -> dict[str, Any]:
    summaries = {name: _load(RUNTIME_ROOT / "runs" / name / "evaluation_summary.json") for name in ARM_NAMES}
    selected = select_policy(summaries)
    authoritative = summaries[selected]["authoritative_final"] if selected else None
    payload = {
        "schema_version": "darkmind-v2-phase4c-policy-analysis-v1",
        "result": "PASS",
        "arms": summaries,
        "stable_policy_found": selected is not None,
        "selected_arm": selected,
        "selected_policy": {
            "optimizer_grouping": summaries[selected]["optimizer_grouping"],
            "initialization_policy": summaries[selected]["initialization_policy"]["name"],
            "schedule": summaries[selected]["schedule"],
            "sequence_order": summaries[selected]["sequence_order"],
        } if selected else None,
        "selected_authoritative_generation_complete": authoritative is not None,
        "selection_rule": (
            "stable integrity first; best mean val/eval; within 1% prefer corrected grouping, simpler global schedule, then lower LR"
        ),
        "generation_quality_role": "diagnostic at 5M; encoding/special-token hard failures remain integrity gates",
        "phase_25m_authorized": False,
    }
    atomic_write_json(RUNTIME_ROOT / "diagnostics" / "policy_analysis.json", payload)
    return payload


def authoritative_best() -> dict[str, Any]:
    analysis = analyze()
    selected = analysis["selected_arm"]
    if selected is None:
        raise RuntimeError("no stable Phase 4C arm is available for authoritative generation")
    return evaluate_run(selected, authoritative_final=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate-all-subsets")
    commands.add_parser("authoritative-best")
    commands.add_parser("analyze")
    commands.add_parser("evaluate-confirmation")
    args = parser.parse_args()
    if args.command == "evaluate-all-subsets":
        payload = {name: evaluate_run(name, authoritative_final=False)["stability"] for name in ARM_NAMES}
    elif args.command == "authoritative-best":
        summary = authoritative_best()
        payload = {"run": summary["arm_name"], "authoritative_complete": summary["authoritative_final"] is not None}
    elif args.command == "evaluate-confirmation":
        summary = evaluate_run(CONFIRMATION_NAME, authoritative_final=True)
        payload = {"run": CONFIRMATION_NAME, "stability": summary["stability"], "authoritative_complete": True}
    else:
        analysis = analyze()
        payload = {key: value for key, value in analysis.items() if key != "arms"}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
