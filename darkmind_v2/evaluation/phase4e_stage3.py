"""Run controlled Phase 4E generation audits without making release claims."""

from __future__ import annotations

import argparse
import json
import time
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
from darkmind_v2.evaluation.phase4d_stage2 import subset_prompts
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer
from darkmind_v2.training.phase4e_stage3 import (
    FINAL_STEP,
    GATE_50_STEP,
    GATE_75_STEP,
    ROOT,
    RUN_DIR,
    TOKENIZER_INPUT,
    atomic_write_json,
    load_json,
)


PROMPTS_PATH = ROOT / "darkmind_v2" / "eval" / "public_preview_prompts.jsonl"
GATES_PATH = ROOT / "darkmind_v2" / "config" / "public_research_preview_gates.json"
AUDIT_STEPS = (GATE_50_STEP, GATE_75_STEP, FINAL_STEP)
PHASE4D_ANALYSIS = Path(r"C:\DarkMindRuntime\phase4d\runs\base_v1_stage2_25m_v2_retry1\generation_analysis.json")


def checkpoint_and_hash(step: int) -> tuple[Path, str]:
    if step not in AUDIT_STEPS:
        raise ValueError(f"unsupported Phase 4E generation checkpoint: {step}")
    manifest = load_json(RUN_DIR / "run_manifest.json")
    checkpoint = Path(manifest["checkpoints"][str(step)])
    value = manifest["checkpoint_hashes"][str(step)]
    expected_hash = value if isinstance(value, str) else value["model_sha256"]
    if sha256_file(checkpoint / "model" / "model.safetensors") != expected_hash:
        raise ValueError(f"Phase 4E milestone model hash mismatch: {step}")
    return checkpoint, expected_hash


def run_subset(step: int) -> dict[str, Any]:
    checkpoint, expected_hash = checkpoint_and_hash(step)
    output_dir = RUN_DIR / "audits" / f"step_{step:06d}" / "subset"
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4E subset hash mismatch")
        return payload
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"incomplete Phase 4E subset requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = subset_prompts(load_audit_prompts(PROMPTS_PATH))
    prompt_hash = sha256_file(PROMPTS_PATH)
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer(TOKENIZER_INPUT)
    started = time.perf_counter()
    greedy_records = []
    sampled_records = []
    for index, prompt in enumerate(prompts, start=1):
        greedy = generate_record(
            model,
            tokenizer,
            prompt,
            max_new_tokens=32,
            do_sample=False,
            profile_name="greedy",
            seed=None,
            checkpoint_stage="stage1_final",
        )
        greedy_records.append(greedy)
        if greedy["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "greedy", greedy, index)
            raise RuntimeError(f"Phase 4E greedy hard failure at step {step}: {greedy['prompt_id']}")
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
            checkpoint_stage="stage1_final",
        )
        sampled_records.append(sampled)
        if sampled["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "sampling", sampled, index)
            raise RuntimeError(f"Phase 4E sampling hard failure at step {step}: {sampled['prompt_id']}")
    greedy_manifest = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={
            "mode": "greedy",
            "max_new_tokens": 32,
            "checkpoint_stage": "stage1_final",
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
            "checkpoint_stage": "stage1_final",
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
        "schema_version": "darkmind-v2-phase4e-milestone-generation-subset-v1",
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
        "release_quality_claimed": False,
    }
    atomic_write_json(summary_path, report)
    del model
    torch.cuda.empty_cache()
    return report


def run_authoritative_final() -> dict[str, Any]:
    checkpoint, expected_hash = checkpoint_and_hash(FINAL_STEP)
    output_dir = RUN_DIR / "audits" / f"step_{FINAL_STEP:06d}" / "authoritative"
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        if payload.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError("completed Phase 4E authoritative hash mismatch")
        return payload
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
        raise ValueError("Phase 4E authoritative generation counts changed")
    if report["greedy"]["hard_failure_total"] + report["sampling"]["hard_failure_total"] != 0:
        raise RuntimeError("Phase 4E authoritative generation hard failure")
    return report


def _warning_rate(summary: dict[str, Any], key: str) -> float:
    return summary["quality_warning_counts"].get(key, 0) / max(summary["generations"], 1)


def _loop_rate(summary: dict[str, Any]) -> float:
    return summary["exact_repeated_ngram_loop_outputs"] / max(summary["generations"], 1)


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generations": summary["generations"],
        "repetition_warning_rate": _warning_rate(summary, "repetition"),
        "exact_loop_rate": _loop_rate(summary),
        "eos_completion_rate": summary["eos_completion_rate"],
        "mean_unique_token_ratio": summary["mean_unique_token_ratio"],
        "empty_output_count": summary["empty_output_count"],
        "hard_failure_total": summary["hard_failure_total"],
        "invalid_utf8_sequence_count": summary["invalid_utf8_sequence_count"],
        "replacement_character_count": summary["replacement_character_count"],
        "mojibake_output_count": summary["mojibake_output_count"],
        "special_token_leakage_count": summary["special_token_leakage_count"],
    }


def generation_analysis() -> dict[str, Any]:
    final = run_authoritative_final()
    subsets = {str(step): run_subset(step) for step in AUDIT_STEPS}
    baseline = load_json(PHASE4D_ANALYSIS)["authoritative_final"]
    payload = {
        "schema_version": "darkmind-v2-phase4e-generation-analysis-v1",
        "result": "PASS",
        "baseline_25m": {mode: _metrics(baseline[mode]) for mode in ("greedy", "sampling")},
        "subset_progression": {
            step: {mode: _metrics(item[mode]) for mode in ("greedy", "sampling")}
            for step, item in subsets.items()
        },
        "authoritative_final": final,
        "final": {mode: _metrics(final[mode]) for mode in ("greedy", "sampling")},
        "hard_failure_total": final["greedy"]["hard_failure_total"] + final["sampling"]["hard_failure_total"],
        "raw_outputs_retained": True,
        "sanitization_performed": False,
        "terminal_eos_policy_corrected": True,
        "generation_quality_role": "diagnostic only; Base V1 is not a chatbot or release candidate",
        "upload_performed": False,
    }
    atomic_write_json(RUN_DIR / "generation_analysis.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    subset = commands.add_parser("subset")
    subset.add_argument("--step", type=int, choices=AUDIT_STEPS, required=True)
    commands.add_parser("authoritative")
    commands.add_parser("analysis")
    args = parser.parse_args()
    if args.command == "subset":
        payload = run_subset(args.step)
    elif args.command == "authoritative":
        payload = run_authoritative_final()
    else:
        payload = generation_analysis()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
