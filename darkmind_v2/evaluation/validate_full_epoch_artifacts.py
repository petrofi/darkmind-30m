"""Fail-closed final integrity validation for all Phase 2C artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash, sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.token_shard_dataset import tokenized_manifest_hash
from darkmind_v2.training.validate_full_epoch_config import (
    EXPECTED_MILESTONES,
    learning_rate_for_step,
    load_and_validate_full_epoch_config,
)


TOKENS_BY_STEP = {
    0: 0,
    256: 1_048_576,
    717: 2_936_832,
    1434: 5_873_664,
    2150: 8_806_400,
    2867: 11_743_232,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_name(step: int) -> str:
    return f"step_{step:06d}_tokens_{TOKENS_BY_STEP[step]:09d}"


def validate(config_path: Path, export_dir: Path) -> dict[str, Any]:
    config = load_and_validate_full_epoch_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = load_json(run_dir / "run_manifest.json")
    failures: list[str] = []
    checks: dict[str, Any] = {}
    if run_manifest.get("status") != "full_epoch_complete":
        failures.append("run manifest is not full_epoch_complete")

    verify_frozen_tokenizer()
    checks["frozen_tokenizer_hashes"] = EXPECTED_HASHES
    actual_manifest_hash = tokenized_manifest_hash(Path(config["data"]["tokenized_dir"]))
    if actual_manifest_hash != run_manifest["tokenized_corpus_manifest_sha256"]:
        failures.append("tokenized corpus manifest hash mismatch")
    corpus_validation = load_json(run_dir / "manifests" / "tokenized_corpus_validation.json")
    if corpus_validation.get("result") != "PASS" or corpus_validation.get("failures"):
        failures.append("saved tokenized corpus validation failed")
    excluded = load_json(run_dir / "manifests" / "excluded_tail.json")
    ordering = load_json(run_dir / "manifests" / "sequence_ordering.json")
    if excluded.get("tokens") != 994 or excluded.get("token_start") != 11_743_232:
        failures.append("excluded-tail manifest mismatch")
    if ordering.get("tokens_consumed") != 11_743_232 or ordering.get("wraparound") is not False:
        failures.append("sequence-ordering manifest mismatch")

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training = [item for item in metrics if item.get("event") == "optimizer_step"]
    if len(training) != 2867 or [item["optimizer_step"] for item in training] != list(range(1, 2868)):
        failures.append("optimizer-step sequence is not exactly 1..2867")
    for item in training:
        step = int(item["optimizer_step"])
        if item["consumed_tokens"] != step * 4096 or item["data_position"] != step * 4096:
            failures.append(f"data-position continuity failure at step {step}")
            break
        if not math.isclose(float(item["learning_rate"]), learning_rate_for_step(step, config), abs_tol=1e-15):
            failures.append(f"learning-rate continuity failure at step {step}")
            break
        if not math.isfinite(float(item["raw_training_loss"])) or not math.isfinite(float(item["gradient_norm"])):
            failures.append(f"non-finite training metric at step {step}")
            break
    checks["training_steps"] = len(training)
    checks["final_data_position"] = training[-1]["data_position"] if training else None

    checkpoint_reports: dict[str, Any] = {}
    for step in EXPECTED_MILESTONES:
        checkpoint = run_dir / "checkpoints" / checkpoint_name(step)
        required = {"checkpoint_metadata.json", "resume_state.pt", "model"}
        missing = sorted(name for name in required if not (checkpoint / name).exists())
        if missing:
            failures.append(f"checkpoint {step} missing: {missing}")
            continue
        metadata = load_json(checkpoint / "checkpoint_metadata.json")
        model_path = checkpoint / "model" / "model.safetensors"
        actual_model_hash = sha256_file(model_path)
        if actual_model_hash != metadata["model_files"]["model_sha256"]:
            failures.append(f"checkpoint {step} model hash mismatch")
        state = metadata["training_state"]
        expected_tokens = TOKENS_BY_STEP[step]
        if (state["step"], state["tokens_seen"], state["data_position"]) != (step, expected_tokens, expected_tokens):
            failures.append(f"checkpoint {step} training state mismatch")
        resume = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
        if resume.get("format") != "darkmind-v2-internal-resume-state-v1":
            failures.append(f"checkpoint {step} resume format mismatch")
        if resume["scheduler"].get("last_epoch") != step:
            failures.append(f"checkpoint {step} scheduler epoch mismatch")
        if not resume.get("rng"):
            failures.append(f"checkpoint {step} RNG state missing")
        if step and not resume["optimizer"]["state"]:
            failures.append(f"checkpoint {step} optimizer state missing")
        model = load_model_package(checkpoint / "model", device="cpu")
        if model.num_parameters() != 9_369_088 or not model.embeddings_are_tied():
            failures.append(f"checkpoint {step} model reload mismatch")
        checkpoint_reports[str(step)] = {
            "path": str(checkpoint),
            "model_sha256": actual_model_hash,
            "optimizer_state_present": bool(resume["optimizer"]["state"]),
            "scheduler_last_epoch": resume["scheduler"]["last_epoch"],
            "rng_state_present": bool(resume["rng"]),
            "model_reload_pass": True,
        }
        del model, resume
    checks["checkpoints"] = checkpoint_reports

    resume_report = load_json(run_dir / "midpoint_process_restart_validation.json")
    if resume_report.get("result") != "PASS" or not all(resume_report.get("checks", {}).values()):
        failures.append("midpoint process-restart evidence failed")
    checks["midpoint_resume"] = resume_report

    audit_reports: dict[str, Any] = {}
    for step in EXPECTED_MILESTONES:
        audit_dir = run_dir / "evaluations" / f"public_preview_v2_step_{step:06d}"
        audit = load_json(audit_dir / "audit_summary.json")
        greedy = load_json(audit_dir / "greedy_manifest.json")
        sampling = load_json(audit_dir / "sampling_manifest.json")
        if audit.get("result") != "PASS":
            failures.append(f"audit {step} failed")
        if audit["greedy"]["hard_failure_total"] or audit["sampling"]["hard_failure_total"]:
            failures.append(f"audit {step} contains hard failures")
        if len(greedy["results"]) != 200 or len(sampling["results"]) != 500:
            failures.append(f"audit {step} generation count mismatch")
        if sha256_file(audit_dir / "greedy_manifest.json") != audit["greedy_manifest_sha256"]:
            failures.append(f"audit {step} greedy manifest hash mismatch")
        if sha256_file(audit_dir / "sampling_manifest.json") != audit["sampling_manifest_sha256"]:
            failures.append(f"audit {step} sampling manifest hash mismatch")
        audit_reports[str(step)] = {
            "checkpoint_model_sha256": audit["checkpoint_model_sha256"],
            "greedy_hard_failures": audit["greedy"]["hard_failure_total"],
            "sampling_hard_failures": audit["sampling"]["hard_failure_total"],
            "greedy_generations": len(greedy["results"]),
            "sampling_generations": len(sampling["results"]),
        }
    checks["audits"] = audit_reports

    summary = load_json(run_dir / "evaluations" / "full_epoch_summary.json")
    summary_core = {key: value for key, value in summary.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(summary_core) != summary.get("deterministic_content_hash"):
        failures.append("full-epoch summary deterministic hash mismatch")
    offline = load_json(run_dir / "evaluations" / "huggingface_offline_validation.json")
    if offline.get("result") != "PASS" or offline.get("failures"):
        failures.append("offline Hugging Face validation failed")
    export_hashes = load_json(export_dir / "file_hashes.json")
    bad_export_hashes = [
        filename for filename, expected in export_hashes.items()
        if sha256_file(export_dir / filename) != expected
    ]
    if bad_export_hashes:
        failures.append(f"local export hash mismatch: {bad_export_hashes}")
    checks["offline_huggingface"] = offline
    checks["verified_export_files"] = len(export_hashes)

    return {
        "schema_version": "darkmind-v2-phase2c-artifact-integrity-v1",
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "checks": checks,
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
        "tokenized_corpus_manifest_sha256": actual_manifest_hash,
        "export_dir": str(export_dir),
        "upload_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_full_epoch.json"))
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/exports/darkmind-v2-tiny-full-epoch"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2c/runs/tiny_full_epoch_seed20260712_v1/"
            "evaluations/artifact_integrity.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite integrity report: {args.output}")
    report = validate(args.config, args.export_dir)
    atomic_write_json(args.output, report)
    print(json.dumps({
        "result": report["result"],
        "failures": report["failures"],
        "training_steps": report["checks"]["training_steps"],
        "final_data_position": report["checks"]["final_data_position"],
        "validated_checkpoints": len(report["checks"]["checkpoints"]),
        "validated_audits": len(report["checks"]["audits"]),
        "verified_export_files": report["checks"]["verified_export_files"],
    }, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
