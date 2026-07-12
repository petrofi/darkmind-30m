"""Compare initial, midpoint, final, and best DarkMind v2 Stage-1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.corpus.detect_mojibake import detect_text
from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash
from darkmind_v2.evaluation.generate_fixed_prompts import generate_fixed_prompts, load_prompts
from darkmind_v2.evaluation.validate_generation_health import (
    classify_generation_health,
    enforce_generation_policy,
)
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer, verify_frozen_tokenizer
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_tiny_stage1 import evaluate_loss
from darkmind_v2.training.validate_stage1_config import load_and_validate_stage1_config


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[int], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_generations(
    model: torch.nn.Module,
    tokenizer: FrozenTokenizer,
    *,
    checkpoint_stage: str,
    do_sample: bool,
    output_path: Path,
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompts = load_prompts(Path("darkmind_v2/eval/fixed_base_prompts.jsonl"))
    manifest = generate_fixed_prompts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        seed=seed,
    )
    warning_counts: dict[str, int] = {}
    hard_counts: dict[str, int] = {}
    token_lengths = []
    character_lengths = []
    unknown_tokens = 0
    for item in manifest["results"]:
        policy = classify_generation_health(
            item["generation"],
            item["token_ids"],
            checkpoint_stage=checkpoint_stage,
            maximum_repetition_ratio=item["health"]["maximum_repetition_ratio"],
            token_trace=item["token_trace"],
        )
        item["policy"] = policy
        token_lengths.append(len(item["token_ids"]))
        character_lengths.append(len(item["generation"]))
        unknown_tokens += item["token_ids"].count(tokenizer.unk_token_id)
        for warning in policy["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        for failure in policy["hard_failures"]:
            hard_counts[failure] = hard_counts.get(failure, 0) + 1
    enforce_generation_policy(manifest["results"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation output: {output_path}")
    atomic_write_json(output_path, manifest)
    return {
        "mode": "seeded_sampling" if do_sample else "greedy",
        "content_hash": manifest["deterministic_content_hash"],
        "prompts": len(manifest["results"]),
        "p50_generation_tokens": percentile(token_lengths, 0.50),
        "p95_generation_tokens": percentile(token_lengths, 0.95),
        "p50_generation_characters": percentile(character_lengths, 0.50),
        "p95_generation_characters": percentile(character_lengths, 0.95),
        "empty_outputs": sum(not item["generation"].strip() for item in manifest["results"]),
        "warning_counts": dict(sorted(warning_counts.items())),
        "hard_failure_counts": dict(sorted(hard_counts.items())),
        "unknown_tokens": unknown_tokens,
        "output": str(output_path),
    }


def tokenizer_health_audit(tokenizer: FrozenTokenizer) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path("darkmind_v2/tokenizer/tokenizer_eval_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_roundtrip_failures = 0
    valid_unknown_tokens = 0
    hostile_records = 0
    hostile_detected = 0
    for record in records:
        text = record["text"]
        if record["expected_valid"]:
            token_ids = tokenizer.encode(text)
            valid_unknown_tokens += token_ids.count(tokenizer.unk_token_id)
            valid_roundtrip_failures += int(tokenizer.decode(token_ids) != text)
        else:
            hostile_records += 1
            findings = bool(detect_text(text)) or "\ufffd" in text or any(
                pattern in text for pattern in record.get("forbidden_patterns", [])
            )
            hostile_detected += int(findings)
    return {
        "valid_records": len(records) - hostile_records,
        "valid_roundtrip_failures": valid_roundtrip_failures,
        "valid_unknown_tokens": valid_unknown_tokens,
        "hostile_records": hostile_records,
        "hostile_detected": hostile_detected,
    }


def evaluate_checkpoint(
    label: str,
    checkpoint_dir: Path,
    *,
    config: dict[str, Any],
    validation_data: TokenShardDataset,
    eval_data: TokenShardDataset,
    tokenizer: FrozenTokenizer,
    run_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model = load_model_package(checkpoint_dir / "model", device="cuda")
    profile = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["selected_profile"]
    sequence_length = config["data"]["sequence_length"]
    validation_sequences = validation_data.total_tokens // sequence_length
    eval_sequences = eval_data.total_tokens // sequence_length
    validation = evaluate_loss(
        model,
        validation_data,
        sequences=validation_sequences,
        sequence_length=sequence_length,
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    evaluation = evaluate_loss(
        model,
        eval_data,
        sequences=eval_sequences,
        sequence_length=sequence_length,
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    checkpoint_stage = {
        "initial": "initialization",
        "step_64": "stage1",
        "midpoint": "midpoint",
        "step_192": "stage1",
        "final": "stage1_final",
    }[label]
    greedy = evaluate_generations(
        model,
        tokenizer,
        checkpoint_stage=checkpoint_stage,
        do_sample=False,
        output_path=evaluation_dir / f"{label}_greedy.json",
        seed=config["seed"],
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
    )
    sampling = evaluate_generations(
        model,
        tokenizer,
        checkpoint_stage=checkpoint_stage,
        do_sample=True,
        output_path=evaluation_dir / f"{label}_sampling.json",
        seed=config["seed"],
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
    )
    return {
        "label": label,
        "checkpoint": str(checkpoint_dir),
        "model_weight_sha256": sha256_file(checkpoint_dir / "model" / "model.safetensors"),
        "validation": validation,
        "eval": evaluation,
        "greedy": greedy,
        "seeded_sampling": sampling,
    }


def evaluate_run(config_path: Path, evaluation_dir: Path | None = None) -> dict[str, Any]:
    config = load_and_validate_stage1_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    evaluation_dir = evaluation_dir or run_dir / "evaluations" / "byte_trace_policy_v1"
    summary_path = evaluation_dir / "stage1_evaluation.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation: {summary_path}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "stage1_complete":
        raise ValueError("Stage-1 run is not complete")
    verify_frozen_tokenizer()
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    if tokenized_manifest_hash(tokenized_dir) != run_manifest["tokenized_corpus_manifest_sha256"]:
        raise ValueError("tokenized manifest changed before evaluation")
    validation_data = TokenShardDataset(tokenized_dir, "validation")
    eval_data = TokenShardDataset(tokenized_dir, "eval")
    tokenizer = FrozenTokenizer()
    checkpoints = {
        "initial": Path(run_manifest["initial_checkpoint"]),
        "step_64": run_dir / "checkpoints" / "step_000064_tokens_000262144",
        "midpoint": run_dir / "checkpoints" / "step_000128_tokens_000524288",
        "step_192": run_dir / "checkpoints" / "step_000192_tokens_000786432",
        "final": run_dir / "checkpoints" / "step_000256_tokens_001048576",
    }
    results = {
        label: evaluate_checkpoint(
            label,
            path,
            config=config,
            validation_data=validation_data,
            eval_data=eval_data,
            tokenizer=tokenizer,
            run_dir=run_dir,
            evaluation_dir=evaluation_dir,
        )
        for label, path in checkpoints.items()
    }
    best_label = min(results, key=lambda item: (results[item]["validation"]["loss"], item))
    best_result = results[best_label]

    metrics = read_metrics(run_dir / "metrics.jsonl")
    training_metrics = [item for item in metrics if item.get("event") == "optimizer_step"]
    raw_losses = [float(item["raw_training_loss"]) for item in training_metrics]
    gradients = [float(item["gradient_norm"]) for item in training_metrics]
    throughputs = [float(item["tokens_per_second"]) for item in training_metrics]
    initial_window = raw_losses[:16]
    final_window = raw_losses[-16:]
    midpoint_metric = next(item for item in training_metrics if item["optimizer_step"] == 128)
    final_metric = training_metrics[-1]
    summary = {
        "optimizer_steps": final_metric["optimizer_step"],
        "consumed_tokens": final_metric["consumed_tokens"],
        "initial_training_loss": raw_losses[0],
        "midpoint_training_loss": midpoint_metric["raw_training_loss"],
        "final_training_loss": final_metric["raw_training_loss"],
        "initial_window_median_train_loss": statistics.median(initial_window),
        "final_window_median_train_loss": statistics.median(final_window),
        "mean_tokens_per_second": statistics.mean(throughputs),
        "median_tokens_per_second": statistics.median(throughputs),
        "peak_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in training_metrics),
        "peak_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in training_metrics),
        "finite_training_losses": all(math.isfinite(value) for value in raw_losses),
        "finite_gradients": all(math.isfinite(value) for value in gradients),
        "initial_validation_loss": results["initial"]["validation"]["loss"],
        "midpoint_validation_loss": results["midpoint"]["validation"]["loss"],
        "final_validation_loss": results["final"]["validation"]["loss"],
        "best_validation_loss": best_result["validation"]["loss"],
        "best_validation_label": best_label,
        "best_validation_checkpoint": best_result["checkpoint"],
        "best_eval_loss": best_result["eval"]["loss"],
        "final_eval_loss": results["final"]["eval"]["loss"],
        "final_perplexity": results["final"]["validation"]["perplexity"],
    }
    tokenizer_audit = tokenizer_health_audit(tokenizer)
    preliminary_gates = {
        "exact_tokens": summary["consumed_tokens"] == 1_048_576,
        "exact_optimizer_steps": summary["optimizer_steps"] == 256,
        "finite_training_loss": summary["finite_training_losses"],
        "finite_gradients": summary["finite_gradients"],
        "train_loss_window_improved": (
            summary["final_window_median_train_loss"] < summary["initial_window_median_train_loss"]
        ),
        "validation_loss_improved": summary["final_validation_loss"] < summary["initial_validation_loss"],
        "midpoint_reload_pass": json.loads(
            (run_dir / "checkpoint_reload_midpoint.json").read_text(encoding="utf-8")
        )["result"] == "PASS",
        "final_reload_pass": json.loads(
            (run_dir / "checkpoint_reload_final.json").read_text(encoding="utf-8")
        )["result"] == "PASS",
        "tokenizer_hash_unchanged": True,
        "tokenized_manifest_unchanged": True,
        "valid_roundtrips": tokenizer_audit["valid_roundtrip_failures"] == 0,
        "unknown_tokens_zero": tokenizer_audit["valid_unknown_tokens"] == 0,
        "generation_hard_failures_zero": all(
            not result[mode]["hard_failure_counts"]
            for result in results.values()
            for mode in ("greedy", "seeded_sampling")
        ),
        "local_huggingface_reload": None,
    }
    report_core = {
        "schema_version": "darkmind-v2-stage1-evaluation-byte-trace-v1",
        "summary": summary,
        "checkpoints": results,
        "best_checkpoint": {
            "label": best_label,
            "checkpoint": best_result["checkpoint"],
            "validation_loss": best_result["validation"]["loss"],
            "eval_loss": best_result["eval"]["loss"],
            "model_weight_sha256": best_result["model_weight_sha256"],
        },
        "tokenizer_health": tokenizer_audit,
        "acceptance_gates": preliminary_gates,
    }
    report = {**report_core, "deterministic_content_hash": canonical_json_hash(report_core)}
    atomic_write_json(summary_path, report)
    atomic_write_json(evaluation_dir / "summarized_metrics.json", summary)
    checkpoint_manifest = {
        label: {
            "path": str(path),
            "model_sha256": results[label]["model_weight_sha256"],
            "metadata_sha256": sha256_file(path / "checkpoint_metadata.json"),
        }
        for label, path in checkpoints.items()
    }
    atomic_write_json(evaluation_dir / "checkpoint_manifest.json", checkpoint_manifest)
    atomic_write_json(
        evaluation_dir / "model_provenance.json",
        {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "tokenizer_model_sha256": EXPECTED_HASHES["tokenizer.model"],
            "tokenized_manifest_sha256": run_manifest["tokenized_corpus_manifest_sha256"],
            "source_corpus_manifest_sha256": run_manifest["source_corpus_manifest_sha256"],
            "final_model_sha256": results["final"]["model_weight_sha256"],
            "best_model_sha256": best_result["model_weight_sha256"],
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1_r2.json"))
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()
    report = evaluate_run(args.config, args.evaluation_dir)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(json.dumps({"acceptance_gates": report["acceptance_gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
