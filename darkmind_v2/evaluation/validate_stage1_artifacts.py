"""Validate immutable Stage-1 checkpoints, corpus provenance, and throughput accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.data_pipeline.validate_full_tokenized_corpus import validate_full_tokenized_corpus
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer


CHECKPOINTS = {
    "initial": ("initial_step_000000", 0, 0),
    "step_64": ("step_000064_tokens_000262144", 64, 262_144),
    "midpoint": ("step_000128_tokens_000524288", 128, 524_288),
    "step_192": ("step_000192_tokens_000786432", 192, 786_432),
    "final": ("step_000256_tokens_001048576", 256, 1_048_576),
}


def checkpoint_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def validate_checkpoint(
    path: Path,
    *,
    expected_step: int,
    expected_tokens: int,
    expected_model_config_hash: str,
    expected_data_hash: str,
) -> dict[str, Any]:
    failures = []
    required = {"model/config.json", "model/model.safetensors", "checkpoint_metadata.json", "resume_state.pt"}
    present = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
    missing = sorted(required - present)
    if missing:
        failures.append(f"missing checkpoint files: {missing}")
    metadata = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    resume = torch.load(path / "resume_state.pt", map_location="cpu", weights_only=False)
    state = resume["training_state"]
    scheduler = resume["scheduler"]
    optimizer = resume["optimizer"]
    if state["step"] != expected_step:
        failures.append("optimizer step counter mismatch")
    if state["tokens_seen"] != expected_tokens:
        failures.append("consumed token counter mismatch")
    if state["data_position"] != expected_tokens:
        failures.append("data position reset or mismatch")
    if int(scheduler.get("last_epoch", -1)) != expected_step:
        failures.append("scheduler step mismatch")
    if expected_step == 0 and optimizer.get("state"):
        failures.append("initial optimizer state is unexpectedly populated")
    if expected_step > 0 and not optimizer.get("state"):
        failures.append("trained checkpoint optimizer state is empty")
    if not {"python", "numpy", "torch_cpu"}.issubset(resume.get("rng", {})):
        failures.append("RNG state is incomplete")
    if metadata.get("model_config_hash") != expected_model_config_hash:
        failures.append("model config provenance mismatch")
    if metadata.get("tokenizer_model_sha256") != EXPECTED_HASHES["tokenizer.model"]:
        failures.append("tokenizer provenance mismatch")
    if metadata.get("tokenized_data_manifest_hash") != expected_data_hash:
        failures.append("corpus manifest provenance mismatch")
    model_hash = sha256_file(path / "model" / "model.safetensors")
    config_hash = sha256_file(path / "model" / "config.json")
    if model_hash != metadata.get("model_files", {}).get("model_sha256"):
        failures.append("model weight hash mismatch")
    if config_hash != metadata.get("model_files", {}).get("config_sha256"):
        failures.append("model config file hash mismatch")
    if metadata.get("training_state") != state:
        failures.append("metadata and resume training states differ")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "path": str(path),
        "step": state["step"],
        "consumed_tokens": state["tokens_seen"],
        "data_position": state["data_position"],
        "scheduler_last_epoch": scheduler.get("last_epoch"),
        "optimizer_state_present": bool(optimizer.get("state")),
        "rng_state_present": bool(resume.get("rng")),
        "model_sha256": model_hash,
        "checkpoint_tree_sha256": checkpoint_tree_hash(path),
    }


def audit_corpus_text(processed_dir: Path) -> dict[str, Any]:
    files = ["tokenizer_train.txt", "tokenizer_validation.txt", "tokenizer_eval.txt"]
    replacement_characters = 0
    lines = 0
    failures = []
    for filename in files:
        path = processed_dir / filename
        try:
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                for line in handle:
                    lines += 1
                    replacement_characters += line.count("\ufffd")
        except UnicodeDecodeError as exc:
            failures.append(f"strict UTF-8 decode failed for {filename}: {exc}")
    if replacement_characters:
        failures.append(f"source corpus contains U+FFFD: {replacement_characters}")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "files": files,
        "lines": lines,
        "replacement_characters": replacement_characters,
    }


def audit_throughput(run_dir: Path) -> dict[str, Any]:
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [item for item in metrics if item.get("event") == "optimizer_step"]
    active_training_seconds = sum(float(item["step_duration_seconds"]) for item in steps)
    segment_files = [
        run_dir / "segment_to_000524288.json",
        run_dir / "segment_to_001048576.json",
    ]
    total_wall_seconds = sum(
        float(json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"])
        for path in segment_files
    )
    overhead_seconds = total_wall_seconds - active_training_seconds
    tokens = int(steps[-1]["consumed_tokens"])
    calibration = json.loads(
        Path("darkmind_v2/data/phase2a/profiling/tiny_stage1_calibration.json").read_text(encoding="utf-8")
    )["attempts"][0]
    return {
        "canonical_metric": "effective training tokens divided by recorded end-to-end segment wall time",
        "total_training_wall_seconds": total_wall_seconds,
        "active_training_seconds": active_training_seconds,
        "evaluation_generation_checkpoint_seconds": overhead_seconds,
        "effective_wall_tokens_per_second": tokens / total_wall_seconds,
        "effective_active_training_tokens_per_second": tokens / active_training_seconds,
        "optimizer_steps_per_wall_second": len(steps) / total_wall_seconds,
        "optimizer_steps": len(steps),
        "tokens": tokens,
        "reported_mean_step_tokens_per_second": sum(float(item["tokens_per_second"]) for item in steps) / len(steps),
        "calibration_tokens_per_second": calibration["tokens_per_second"],
        "calibration_step_seconds": calibration["step_duration_seconds"],
        "explanation": (
            "The approximately 61k metric averages per-step effective throughput after warm-up and excludes "
            "validation, generation, checkpointing, and reloads. The approximately 10k calibration was one "
            "cold optimizer step and includes first-step CUDA/kernel warm-up. The canonical full-run metric "
            "uses both recorded segment wall times and therefore includes periodic overhead."
        ),
    }


def validate(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable integrity report: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    tokenizer_manifest = verify_frozen_tokenizer()
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    processed_dir = Path("darkmind_v2/data/phase1b/processed")
    corpus = validate_full_tokenized_corpus(tokenized_dir, processed_dir)
    checkpoints = {}
    for label, (directory, step, tokens) in CHECKPOINTS.items():
        checkpoints[label] = validate_checkpoint(
            run_dir / "checkpoints" / directory,
            expected_step=step,
            expected_tokens=tokens,
            expected_model_config_hash=run_manifest["model_config_hash"],
            expected_data_hash=run_manifest["tokenized_corpus_manifest_sha256"],
        )
    failures = []
    if corpus["result"] != "PASS":
        failures.extend(corpus["failures"])
    if any(item["result"] != "PASS" for item in checkpoints.values()):
        failures.append("one or more checkpoints failed integrity validation")
    corpus_text = audit_corpus_text(processed_dir)
    if corpus_text["result"] != "PASS":
        failures.extend(corpus_text["failures"])
    throughput = audit_throughput(run_dir)
    finite_metrics = all(
        math.isfinite(float(item[key]))
        for item in [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("event") == "optimizer_step"
        ]
        for key in ("raw_training_loss", "gradient_norm")
    )
    if not finite_metrics:
        failures.append("non-finite loss or gradient in training metrics")
    payload = {
        "schema_version": "darkmind-v2-stage1-integrity-v1",
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "tokenizer": {
            "result": "PASS",
            "name": tokenizer_manifest["tokenizer_name"],
            "hashes": EXPECTED_HASHES,
        },
        "corpus": corpus,
        "corpus_text": corpus_text,
        "checkpoints": checkpoints,
        "resume_invariants": {
            "scheduler_did_not_restart": all(
                item["scheduler_last_epoch"] == item["step"] for item in checkpoints.values()
            ),
            "data_state_did_not_reset": all(
                item["data_position"] == item["consumed_tokens"] for item in checkpoints.values()
            ),
        },
        "finite_training_loss_and_gradients": finite_metrics,
        "throughput": throughput,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1_r2.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/byte_trace_policy_v1/artifact_integrity.json"
        ),
    )
    args = parser.parse_args()
    report = validate(args.config, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
