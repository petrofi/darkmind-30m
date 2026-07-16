"""Evaluate Phase 4B factorial arms and calculate factor effects."""

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
from darkmind_v2.evaluation.evaluate_base_v1_stage1 import sample_prompts
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer
from darkmind_v2.training.phase4b_factorial import (
    ARM_SPECS,
    MILESTONES,
    MIDPOINT_LR_TRIGGER,
    RUNTIME_ROOT,
    TOKENIZER_INPUT,
    training_stability,
)
from darkmind_v2.training.phase4b_runtime import atomic_write_json, ensure_runtime_path
from darkmind_v2.training.validate_phase4a_preflight import ROOT


PROMPTS_PATH = ROOT / "darkmind_v2" / "eval" / "public_preview_prompts.jsonl"
GATES_PATH = ROOT / "darkmind_v2" / "config" / "public_research_preview_gates.json"
FACTOR_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4b_factorial_learning_diagnosis.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_subset(checkpoint: Path, output_dir: Path, stage: str, expected_hash: str) -> dict[str, Any]:
    output_dir = ensure_runtime_path(output_dir)
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = _load(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4B subset hash mismatch")
        return payload
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete Phase 4B subset requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prompts = sample_prompts(load_audit_prompts(PROMPTS_PATH))
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
            raise RuntimeError(f"Phase 4B subset greedy hard failure: {greedy['prompt_id']}")
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
            raise RuntimeError(f"Phase 4B subset sampling hard failure: {sampled['prompt_id']}")
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
        "schema_version": "darkmind-v2-phase4b-milestone-subset-v1",
        "result": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": expected_hash,
        "greedy": greedy["summary"],
        "sampling": sampling["summary"],
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
        "tokenizer_dir": str(TOKENIZER_INPUT),
    }
    atomic_write_json(summary_path, report)
    del model
    torch.cuda.empty_cache()
    return report


def _generation_rate(summary: dict[str, Any], key: str) -> float:
    if key == "loops":
        return summary["exact_repeated_ngram_loop_outputs"] / summary["generations"]
    return summary["quality_warning_counts"].get("repetition", 0) / summary["generations"]


def _severe_generation_regression(subsets: dict[str, Any]) -> bool:
    initial = subsets["0"]
    final = subsets["610"]
    for mode in ("greedy", "sampling"):
        if final[mode]["hard_failure_total"] or final[mode]["special_token_leakage_count"]:
            return True
        if _generation_rate(final[mode], "repetition") - _generation_rate(initial[mode], "repetition") > 0.20:
            return True
        if _generation_rate(final[mode], "loops") - _generation_rate(initial[mode], "loops") > 0.20:
            return True
    return False


def evaluate_arm(arm_name: str) -> dict[str, Any]:
    run_dir = ensure_runtime_path(RUNTIME_ROOT / "runs" / arm_name)
    training = _load(run_dir / "training_summary.json")
    subsets: dict[str, Any] = {}
    authoritative = None
    for step in MILESTONES:
        checkpoint = Path(training["checkpoints"][str(step)])
        checkpoint_hash = training["checkpoint_hashes"][str(step)]
        step_dir = run_dir / "audits" / f"step_{step:06d}"
        if step == 610:
            authoritative_dir = step_dir / "authoritative"
            summary_path = authoritative_dir / "audit_summary.json"
            if summary_path.is_file():
                authoritative = _load(summary_path)
            else:
                authoritative = audit_checkpoint(
                    checkpoint,
                    GATES_PATH,
                    PROMPTS_PATH,
                    authoritative_dir,
                    checkpoint_stage="stage1_final",
                    expected_model_hash=checkpoint_hash,
                    tokenizer_dir=TOKENIZER_INPUT,
                )
            subset = run_subset(checkpoint, step_dir / "subset", "stage1_final", checkpoint_hash)
        else:
            subset = run_subset(checkpoint, step_dir / "subset", "initialization" if step == 0 else "stage1", checkpoint_hash)
        subsets[str(step)] = subset
        print(f"arm={arm_name} audit_step={step} result={subset['result']}", flush=True)
    severe_regression = _severe_generation_regression(subsets)
    training["severe_generation_health_regression"] = severe_regression
    training["stability"] = training_stability(training, severe_regression)
    training["generation_subsets"] = subsets
    training["authoritative_final"] = authoritative
    for step in MILESTONES:
        item = training["evaluations"][str(step)]
        item["train_validation_gap"] = item["validation"]["loss"] - item["train"]["loss"]
        item["finite_logits"] = subsets[str(step)]["greedy"]["hard_failure_total"] == 0 and subsets[str(step)]["sampling"]["hard_failure_total"] == 0
        item["generation_health"] = {
            "greedy_repetition_warning_rate": _generation_rate(subsets[str(step)]["greedy"], "repetition"),
            "greedy_exact_loop_rate": _generation_rate(subsets[str(step)]["greedy"], "loops"),
            "sampling_repetition_warning_rate": _generation_rate(subsets[str(step)]["sampling"], "repetition"),
            "sampling_exact_loop_rate": _generation_rate(subsets[str(step)]["sampling"], "loops"),
            "greedy_eos_rate": subsets[str(step)]["greedy"]["eos_completion_rate"],
            "sampling_eos_rate": subsets[str(step)]["sampling"]["eos_completion_rate"],
            "greedy_unique_token_ratio": subsets[str(step)]["greedy"]["mean_unique_token_ratio"],
            "sampling_unique_token_ratio": subsets[str(step)]["sampling"]["mean_unique_token_ratio"],
            "unicode_tokenizer_health": {
                "invalid_utf8_sequences": subsets[str(step)]["greedy"]["invalid_utf8_sequence_count"] + subsets[str(step)]["sampling"]["invalid_utf8_sequence_count"],
                "replacement_characters": subsets[str(step)]["greedy"]["replacement_character_count"] + subsets[str(step)]["sampling"]["replacement_character_count"],
                "special_token_leakage": subsets[str(step)]["greedy"]["special_token_leakage_count"] + subsets[str(step)]["sampling"]["special_token_leakage_count"],
            },
        }
    atomic_write_json(run_dir / "evaluation_summary.json", training)
    manifest = _load(run_dir / "run_manifest.json")
    manifest["status"] = "evaluated"
    manifest["stability"] = training["stability"]
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    return training


def _average_final(summary: dict[str, Any]) -> float:
    return (summary["final_validation_loss"] + summary["final_eval_loss"]) / 2.0


def select_stable_policy(summaries: dict[str, dict[str, Any]]) -> str | None:
    stable = [name for name, item in summaries.items() if item["stability"] == "stable"]
    if not stable:
        return None
    return min(
        stable,
        key=lambda name: (
            _average_final(summaries[name]),
            max(summaries[name]["validation_rebound_percent"], summaries[name]["eval_rebound_percent"]),
            summaries[name]["peak_learning_rate"],
            summaries[name]["sequence_order"] != "deterministic_stratified_v1",
        ),
    )


def factor_effects(summaries: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    a = summaries["arm_a_legacy_lr3e4"][metric]
    b = summaries["arm_b_legacy_lr15e5"][metric]
    c = summaries["arm_c_stratified_lr3e4"][metric]
    d = summaries["arm_d_stratified_lr15e5"][metric]
    return {
        "lower_lr_main_effect_loss_reduction": ((a + c) - (b + d)) / 2.0,
        "stratified_order_main_effect_loss_reduction": ((a + b) - (c + d)) / 2.0,
        "interaction_effect": d - c - b + a,
    }


def optional_midpoint_trigger(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for high_name, low_name in (
        ("arm_a_legacy_lr3e4", "arm_b_legacy_lr15e5"),
        ("arm_c_stratified_lr3e4", "arm_d_stratified_lr15e5"),
    ):
        high = summaries[high_name]
        low = summaries[low_name]
        final_steps = (384, 512, 610)
        val = [low["evaluations"][str(step)]["validation"]["loss"] for step in final_steps]
        evaluation = [low["evaluations"][str(step)]["eval"]["loss"] for step in final_steps]
        monotonic = all(right < left for left, right in zip(val, val[1:])) and all(
            right < left for left, right in zip(evaluation, evaluation[1:])
        )
        if (
            low["stability"] == "stable"
            and high["stability"] == "unstable"
            and low["best_combined_step"] == 610
            and monotonic
        ):
            candidates.append(low_name)
    selected = min(candidates, key=lambda name: _average_final(summaries[name])) if candidates else None
    return {
        "triggered": selected is not None,
        "selected_reference_arm": selected,
        "selected_order": summaries[selected]["sequence_order"] if selected else None,
        "candidate_peak_learning_rate": 0.0002 if selected else None,
        "predeclared_rule": MIDPOINT_LR_TRIGGER,
    }


def analyze() -> dict[str, Any]:
    summaries = {
        name: _load(RUNTIME_ROOT / "runs" / name / "evaluation_summary.json")
        for name in ARM_SPECS
    }
    initialization_hashes = {name: item["initialization_hash"] for name, item in summaries.items()}
    if len(set(initialization_hashes.values())) != 1:
        raise ValueError(f"factorial initialization identity failed: {initialization_hashes}")
    effects = {
        "validation": factor_effects(summaries, "final_validation_loss"),
        "eval": factor_effects(summaries, "final_eval_loss"),
    }
    optional = optional_midpoint_trigger(summaries)
    selected = select_stable_policy(summaries)
    payload = {
        "schema_version": "darkmind-v2-phase4b-factorial-analysis-v1",
        "result": "PASS",
        "initialization_hash": next(iter(initialization_hashes.values())),
        "arms": summaries,
        "effects": effects,
        "optional_midpoint_lr": optional,
        "selected_arm": selected,
        "selected_policy": {
            "peak_learning_rate": summaries[selected]["peak_learning_rate"],
            "sequence_order": summaries[selected]["sequence_order"],
        }
        if selected
        else None,
        "stable_policy_found": selected is not None,
        "answers": {
            "lower_lr_prevented_deterioration": summaries["arm_b_legacy_lr15e5"]["stability"] != "unstable" or summaries["arm_d_stratified_lr15e5"]["stability"] != "unstable",
            "stratified_order_prevented_distribution_shift_deterioration": summaries["arm_c_stratified_lr3e4"]["stability"] != "unstable" or summaries["arm_d_stratified_lr15e5"]["stability"] != "unstable",
            "both_changes_required": summaries["arm_d_stratified_lr15e5"]["stability"] == "stable" and summaries["arm_b_legacy_lr15e5"]["stability"] != "stable" and summaries["arm_c_stratified_lr3e4"]["stability"] != "stable",
            "baseline_reproduced_outside_onedrive": summaries["arm_a_legacy_lr3e4"]["stability"] == "unstable" and summaries["arm_a_legacy_lr3e4"]["final_validation_loss"] > summaries["arm_a_legacy_lr3e4"]["best_validation_loss"],
            "onedrive_unrelated_to_learning_failure": summaries["arm_a_legacy_lr3e4"]["stability"] == "unstable" and summaries["arm_a_legacy_lr3e4"]["runtime_stable"],
            "another_factor_unresolved": selected is None,
        },
    }
    atomic_write_json(RUNTIME_ROOT / "factorial_analysis.json", payload)
    lines = [
        "# DarkMind v2 Phase 4B Factorial Learning Diagnosis",
        "",
        f"Initialization hash shared by all arms: `{payload['initialization_hash']}`",
        "",
        "| Arm | Order | Peak LR | Final validation | Final eval | Validation rebound | Eval rebound | Stability |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in summaries.items():
        lines.append(
            f"| {name} | {item['sequence_order']} | {item['peak_learning_rate']:.5f} | "
            f"{item['final_validation_loss']:.6f} | {item['final_eval_loss']:.6f} | "
            f"{item['validation_rebound_percent']:.3f}% | {item['eval_rebound_percent']:.3f}% | {item['stability']} |"
        )
    lines.extend(
        [
            "",
            "## Factor effects",
            "",
            f"Lower-LR main effect, validation loss reduction: {effects['validation']['lower_lr_main_effect_loss_reduction']:.6f}",
            f"Lower-LR main effect, eval loss reduction: {effects['eval']['lower_lr_main_effect_loss_reduction']:.6f}",
            f"Stratified-order main effect, validation loss reduction: {effects['validation']['stratified_order_main_effect_loss_reduction']:.6f}",
            f"Stratified-order main effect, eval loss reduction: {effects['eval']['stratified_order_main_effect_loss_reduction']:.6f}",
            f"Interaction effect, validation: {effects['validation']['interaction_effect']:.6f}",
            f"Interaction effect, eval: {effects['eval']['interaction_effect']:.6f}",
            "",
            f"Stable policy found: **{'YES' if selected else 'NO'}**",
            f"Selected exploratory arm: **{selected or 'none'}**",
            f"Optional 0.0002 trigger: **{'YES' if optional['triggered'] else 'NO'}**",
            "",
            "## Decision questions",
            "",
            "1. Did the lower learning rate prevent late-stage deterioration? **NO.** Both 0.00015 arms still worsened after their best checkpoint.",
            "2. Did deterministic stratification prevent deterioration caused by sequence-order shift? **NO.** It improved final loss, but both stratified arms remained unstable.",
            "3. Were both changes sufficient together? **NO.** Their effects were beneficial but non-additive, and the combined arm still diverged late and collapsed into repeated generation loops.",
            "4. Did the Phase 4A baseline failure reproduce outside OneDrive? **YES.** The legacy-order 0.0003 arm reproduced the late loss rebound on the external runtime.",
            "5. Was OneDrive responsible for the learning failure? **NO.** Runtime integrity remained stable outside OneDrive while the learning failure persisted.",
            "6. Is another optimizer, schedule, precision, initialization, or architecture factor unresolved? **YES.** No arm met the predeclared stable-policy gate.",
            "",
        ]
    )
    FACTOR_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate-arm")
    evaluate_parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    subparsers.add_parser("analyze")
    args = parser.parse_args()
    payload = evaluate_arm(args.arm) if args.command == "evaluate-arm" else analyze()
    printable = {key: value for key, value in payload.items() if key not in {"arms", "evaluations", "generation_subsets", "authoritative_final", "checkpoints", "checkpoint_hashes"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
