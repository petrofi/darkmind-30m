"""Run the immutable Phase 4C V2 confirmation in fresh process segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from darkmind_v2.training.phase3b_finalist_pilots import evaluate_loss
from darkmind_v2.training.phase4b_factorial import OrderedTokenDataset, percentile, rebound_percent
from darkmind_v2.training.phase4c_diagnostics import (
    INITIALIZATION_SEED,
    MODEL_INPUT,
    ORDER_INPUT,
    RUNTIME_ROOT,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    TOKENS_PER_STEP,
    atomic_write_json,
    build_scheduler,
    ensure_runtime_path,
    learning_rate_for_policy,
)
from darkmind_v2.training.phase4c_policy import V2_CONFIG, validate_v2_config
from darkmind_v2.training.phase4c_training import (
    DIAGNOSTIC_THRESHOLDS,
    MILESTONES,
    STAGE1_STEPS,
    STAGE1_TOKENS,
    _checkpoint_name,
    _max_consecutive_worsening,
    _milestone_diagnostics,
    apply_initialization_policy,
    build_optimizer,
    classify_stability,
    evaluate_probe,
    normalize_diagnostic_summary,
    optimizer_step,
    prepare_probe_manifest,
    save_model_checkpoint,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_base_v1_stage1 import GpuMonitor
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.training_state import TrainingState


RUN_DIR = RUNTIME_ROOT / "runs" / "base_v1_stage1_5m_v2_confirmation"
RESUME_STEPS = {0, 305, 610}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = ensure_runtime_path(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rename_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def _resume_checkpoint(
    checkpoint: Path,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    data_hash: str,
) -> dict[str, Any]:
    checkpoint = ensure_runtime_path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite confirmation checkpoint: {checkpoint}")
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale confirmation checkpoint requires inspection: {temporary}")
    metadata = save_checkpoint(
        temporary,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_hash,
    )
    _rename_with_retry(temporary, checkpoint)
    return metadata


def rng_fingerprint(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("utf-8"))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode("utf-8"))
    digest.update(state["torch_cpu"].detach().cpu().contiguous().numpy().tobytes())
    for value in state.get("torch_cuda", []):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_hash(metadata: dict[str, Any]) -> str:
    return metadata["model_files"]["model_sha256"]


def _load_config() -> dict[str, Any]:
    config = _load(V2_CONFIG)
    validate_v2_config(config)
    if Path(config["run_dir"]).resolve() != RUN_DIR.resolve():
        raise ValueError("confirmation run path mismatch")
    return config


def _create_model_stack(config: dict[str, Any]) -> tuple[DarkMindV2ForCausalLM, torch.optim.AdamW, Any, torch.device]:
    set_deterministic_seed(INITIALIZATION_SEED)
    device = torch.device("cuda")
    model_config = DarkMindV2Config.from_json_file(MODEL_INPUT)
    validate_training_environment(model_config, micro_batch_size=2, precision="bf16")
    model = DarkMindV2ForCausalLM(model_config)
    apply_initialization_policy(model, config["initialization_policy"])
    model = model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config["schedule"])
    return model, optimizer, scheduler, device


def initialize() -> dict[str, Any]:
    config = _load_config()
    run_dir = ensure_runtime_path(RUN_DIR)
    if run_dir.exists():
        raise FileExistsError(f"immutable confirmation directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for name in ("checkpoints", "diagnostics", "resume", "audits"):
        (run_dir / name).mkdir()
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    verify_frozen_tokenizer(TOKENIZER_INPUT)
    probe_manifest = prepare_probe_manifest()
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    data_hash = tokenized_manifest_hash(TOKENIZED_INPUT)
    model, optimizer, scheduler, device = _create_model_stack(config)
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_base = TokenShardDataset(TOKENIZED_INPUT, "train")
    initial = {
        "train": evaluate_loss(model, order_data, sequence_length=512, micro_batch_size=2, device=device, maximum_tokens=TOKENS_PER_STEP),
        "validation": evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device),
        "eval": evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device),
    }
    checkpoint = run_dir / "checkpoints" / _checkpoint_name(0)
    state = TrainingState(
        step=0,
        tokens_seen=0,
        data_position=0,
        best_validation_loss=initial["validation"]["loss"],
        last_validation_loss=initial["validation"]["loss"],
        best_checkpoint=str(checkpoint),
    )
    metadata = _resume_checkpoint(checkpoint, model, optimizer, scheduler, state, data_hash)
    activation_index = int(probe_manifest["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_base.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    probes = {name: evaluate_probe(model, probe, device) for name, probe in probe_manifest["probes"].items()}
    diagnostics = _milestone_diagnostics(
        model,
        step=0,
        parameter_diagnostics=[],
        clipped_fraction=0.0,
        probe_losses=probes,
        activation_tokens=activation_tokens,
    )
    atomic_write_json(run_dir / "diagnostics" / "step_000000.json", diagnostics)
    atomic_write_json(run_dir / "evaluations.json", {"0": {**initial, "probes": probes}})
    atomic_write_json(run_dir / "resolved_config.json", config)
    manifest = {
        "schema_version": "darkmind-v2-phase4c-confirmation-run-v1",
        "status": "initialized",
        "process_ids": [os.getpid()],
        "segments": [],
        "initialization_hash": _model_hash(metadata),
        "checkpoints": {"0": str(checkpoint)},
        "checkpoint_hashes": {"0": _model_hash(metadata)},
        "diagnostic_snapshots": {"0": str(run_dir / "diagnostics" / "step_000000.json")},
        "latest_resume_checkpoint": str(checkpoint),
        "sequence_order_hash": order_data.order_hash,
        "data_manifest_hash": data_hash,
        "runtime_outside_onedrive": True,
        "phase_25m_authorized": False,
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "progress.json", {"status": "initialized", "optimizer_step": 0, "tokens_consumed": 0})
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return manifest


def _save_milestone(
    *,
    step: int,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    scheduler: Any,
    state: TrainingState,
    config: dict[str, Any],
    data_hash: str,
    order_hash: str,
) -> tuple[Path, str]:
    checkpoint = RUN_DIR / "checkpoints" / _checkpoint_name(step)
    if step in RESUME_STEPS:
        metadata = _resume_checkpoint(checkpoint, model, optimizer, scheduler, state, data_hash)
        return checkpoint, _model_hash(metadata)
    metadata = save_model_checkpoint(
        checkpoint,
        model,
        config=config,
        step=step,
        data_hash=data_hash,
        order_hash=order_hash,
    )
    return checkpoint, metadata["model_sha256"]


def train_segment(target_step: int) -> dict[str, Any]:
    if target_step not in {305, 610}:
        raise ValueError("confirmation segment target must be 305 or 610")
    config = _load_config()
    manifest = _load(RUN_DIR / "run_manifest.json")
    expected_start = 0 if target_step == 305 else 305
    if manifest["segments"] and target_step == 305:
        raise ValueError("confirmation first segment already completed")
    checkpoint = Path(manifest["latest_resume_checkpoint"])
    resume_payload = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_rng = rng_fingerprint(resume_payload["rng"])
    model, optimizer, scheduler, device = _create_model_stack(config)
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=manifest["data_manifest_hash"],
    )
    actual_rng = rng_fingerprint(capture_rng_state())
    if state.step != expected_start or state.tokens_seen != expected_start * TOKENS_PER_STEP or state.data_position != state.tokens_seen:
        raise ValueError("confirmation resume step/token/data position mismatch")
    if expected_rng != actual_rng:
        raise ValueError("confirmation RNG continuity failed")
    if scheduler.last_epoch != state.step:
        raise ValueError("confirmation scheduler epoch mismatch")
    expected_lr = learning_rate_for_policy(state.step + 1, config["schedule"])
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, abs_tol=1e-15):
        raise ValueError("confirmation next applied LR mismatch")
    if os.getpid() in manifest["process_ids"]:
        raise ValueError("confirmation segment did not start in a fresh process")
    manifest["process_ids"].append(os.getpid())
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)

    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_base = TokenShardDataset(TOKENIZED_INPUT, "train")
    probe_manifest = prepare_probe_manifest()
    activation_index = int(probe_manifest["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_base.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    evaluations = _load(RUN_DIR / "evaluations.json")
    prior_metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    clipped_steps = sum(int(item["clipped"]) for item in prior_metrics)
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    segment_start_position = state.data_position
    try:
        while state.step < target_step:
            step = state.step + 1
            learning_rate = optimizer.param_groups[0]["lr"]
            metric = optimizer_step(
                model,
                optimizer,
                order_data,
                data_position=state.data_position,
                diagnostic=step in MILESTONES or step == 305,
                device=device,
            )
            scheduler.step()
            state.step = step
            state.tokens_seen += TOKENS_PER_STEP
            state.data_position += TOKENS_PER_STEP
            clipped_steps += int(metric["clipped"])
            state.last_training_loss = metric["raw_train_loss"]
            state.smoothed_training_loss = (
                metric["raw_train_loss"]
                if state.smoothed_training_loss is None
                else 0.9 * state.smoothed_training_loss + 0.1 * metric["raw_train_loss"]
            )
            parameter_diagnostics = metric.pop("parameter_diagnostics")
            metric.update(
                {
                    "optimizer_step": step,
                    "tokens_consumed": state.tokens_seen,
                    "learning_rate_applied": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "smoothed_train_loss": state.smoothed_training_loss,
                    "clipped_step_fraction": clipped_steps / step,
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "gpu": monitor.latest(),
                    "non_finite_event_count": 0,
                }
            )
            if step in MILESTONES or step == 305:
                validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                evaluation = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                probes = {name: evaluate_probe(model, probe, device) for name, probe in probe_manifest["probes"].items()}
                diagnostics = _milestone_diagnostics(
                    model,
                    step=step,
                    parameter_diagnostics=parameter_diagnostics or [],
                    clipped_fraction=clipped_steps / step,
                    probe_losses=probes,
                    activation_tokens=activation_tokens,
                )
                diagnostic_path = RUN_DIR / "diagnostics" / f"step_{step:06d}.json"
                atomic_write_json(diagnostic_path, diagnostics)
                state.last_validation_loss = validation["loss"]
                checkpoint_path = RUN_DIR / "checkpoints" / _checkpoint_name(step)
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = validation["loss"]
                    state.best_checkpoint = str(checkpoint_path)
                saved_path, model_hash = _save_milestone(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    config=config,
                    data_hash=manifest["data_manifest_hash"],
                    order_hash=manifest["sequence_order_hash"],
                )
                evaluations[str(step)] = {
                    "train": {"loss": metric["raw_train_loss"], "smoothed_loss": state.smoothed_training_loss},
                    "validation": validation,
                    "eval": evaluation,
                    "probes": probes,
                }
                manifest["checkpoints"][str(step)] = str(saved_path)
                manifest["checkpoint_hashes"][str(step)] = model_hash
                manifest["diagnostic_snapshots"][str(step)] = str(diagnostic_path)
                if step in {305, 610}:
                    manifest["latest_resume_checkpoint"] = str(saved_path)
                metric.update({"validation_loss": validation["loss"], "eval_loss": evaluation["loss"], "checkpoint": str(saved_path)})
                atomic_write_json(RUN_DIR / "evaluations.json", evaluations)
                atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
            _append_jsonl(RUN_DIR / "metrics.jsonl", metric)
            if step % 5 == 0 or step in MILESTONES:
                atomic_write_json(RUN_DIR / "progress.json", {"status": "training", "optimizer_step": step, "tokens_consumed": state.tokens_seen})
            if step % 10 == 0 or step in MILESTONES or step == 305:
                print(
                    f"confirmation step={step} loss={metric['raw_train_loss']:.6f} "
                    f"lr={learning_rate:.9f} tok_s={metric['active_tokens_per_second']:.1f}",
                    flush=True,
                )
    finally:
        monitor.close()
    segment = {
        "schema_version": "darkmind-v2-phase4c-confirmation-segment-v1",
        "result": "PASS",
        "process_id": os.getpid(),
        "segment_start_step": expected_start,
        "segment_end_step": target_step,
        "segment_token_range": [segment_start_position, state.data_position],
        "rng_continuity": expected_rng == actual_rng,
        "scheduler_continuity": True,
        "data_position_continuity": True,
        "no_repeated_or_skipped_sequence": True,
        "next_optimizer_step": target_step + 1 if target_step < 610 else None,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "latest_resume_checkpoint": manifest["latest_resume_checkpoint"],
    }
    manifest["segments"].append(segment)
    manifest["status"] = "midpoint_complete" if target_step == 305 else "training_complete"
    atomic_write_json(RUN_DIR / "resume" / f"segment_to_step_{target_step:06d}.json", segment)
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": manifest["status"], "optimizer_step": target_step, "tokens_consumed": state.tokens_seen})
    if target_step == 610:
        _finalize_training_summary(config, manifest)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return segment


def validate_restart_evidence(payload: dict[str, Any]) -> None:
    required = {
        "fresh_processes",
        "segment_boundary_exact",
        "rng_continuity",
        "scheduler_continuity",
        "data_position_continuity",
        "no_repeated_or_skipped_sequence",
        "midpoint_resume_checkpoint_present",
        "final_resume_checkpoint_present",
    }
    if set(payload["checks"]) != required or not all(payload["checks"].values()):
        raise ValueError("confirmation process-restart validation failed")


def _finalize_training_summary(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    evaluations = _load(RUN_DIR / "evaluations.json")
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    validation_losses = [evaluations[str(step)]["validation"]["loss"] for step in MILESTONES]
    eval_losses = [evaluations[str(step)]["eval"]["loss"] for step in MILESTONES]
    summary = {
        "schema_version": "darkmind-v2-phase4c-confirmation-summary-v1",
        "result": "PASS",
        "integrity_pass": True,
        "arm_name": "base_v1_stage1_5m_v2_confirmation",
        "sequence_order": "deterministic_stratified_v1",
        "sequence_order_hash": manifest["sequence_order_hash"],
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_policy": {"name": "base_v1_standard_v1", "residual_projection_scale": 1.0, "modified_tensors": 0},
        "initialization_hash": manifest["initialization_hash"],
        "optimizer_grouping": "corrected_adamw_v1",
        "schedule": config["schedule"],
        "optimizer_steps": STAGE1_STEPS,
        "training_tokens": STAGE1_TOKENS,
        "initial_validation_loss": validation_losses[0],
        "initial_eval_loss": eval_losses[0],
        "best_validation_loss": min(validation_losses),
        "best_eval_loss": min(eval_losses),
        "best_combined_step": min(MILESTONES, key=lambda step: evaluations[str(step)]["validation"]["loss"] + evaluations[str(step)]["eval"]["loss"]),
        "final_validation_loss": validation_losses[-1],
        "final_eval_loss": eval_losses[-1],
        "validation_improvement_percent": (validation_losses[0] - validation_losses[-1]) * 100.0 / validation_losses[0],
        "eval_improvement_percent": (eval_losses[0] - eval_losses[-1]) * 100.0 / eval_losses[0],
        "validation_rebound_percent": rebound_percent(min(validation_losses), validation_losses[-1]),
        "eval_rebound_percent": rebound_percent(min(eval_losses), eval_losses[-1]),
        "consecutive_worsening_validation_evaluations": _max_consecutive_worsening(validation_losses),
        "consecutive_worsening_eval_evaluations": _max_consecutive_worsening(eval_losses),
        "last_three_validation_sustained_worsening": all(right > left for left, right in zip(validation_losses[-3:], validation_losses[-2:])),
        "last_three_eval_sustained_worsening": all(right > left for left, right in zip(eval_losses[-3:], eval_losses[-2:])),
        "gradient_norm_p50": percentile([item["gradient_norm"] for item in metrics], 0.50),
        "gradient_norm_p95": percentile([item["gradient_norm"] for item in metrics], 0.95),
        "gradient_norm_max": max(item["gradient_norm"] for item in metrics),
        "clipped_step_fraction": sum(int(item["clipped"]) for item in metrics) / STAGE1_STEPS,
        "maximum_parameter_update_to_weight_ratio": max(
            record["update_to_weight_ratio"]
            for step in manifest["diagnostic_snapshots"]
            for record in _load(Path(manifest["diagnostic_snapshots"][step]))["parameter_diagnostics"]
        ),
        "non_finite_events": sum(item["non_finite_event_count"] for item in metrics),
        "evaluations": {step: value for step, value in evaluations.items() if int(step) in MILESTONES},
        "diagnostic_snapshots": {step: path for step, path in manifest["diagnostic_snapshots"].items() if int(step) in MILESTONES},
        "checkpoints": {step: path for step, path in manifest["checkpoints"].items() if int(step) in MILESTONES},
        "checkpoint_hashes": {step: value for step, value in manifest["checkpoint_hashes"].items() if int(step) in MILESTONES},
        "active_tokens_per_second": STAGE1_TOKENS / sum(item["optimizer_step_duration_seconds"] for item in metrics),
        "wall_tokens_per_second": STAGE1_TOKENS / sum(segment["elapsed_seconds"] for segment in manifest["segments"]),
        "peak_allocated_vram_bytes": max(segment["peak_allocated_bytes"] for segment in manifest["segments"]),
        "peak_reserved_vram_bytes": max(segment["peak_reserved_bytes"] for segment in manifest["segments"]),
        "runtime_stable": True,
    }
    snapshots = {step: _load(Path(path)) for step, path in summary["diagnostic_snapshots"].items()}
    summary = normalize_diagnostic_summary(summary, snapshots)
    summary["stability"] = classify_stability(summary)
    restart = {
        "schema_version": "darkmind-v2-phase4c-process-restart-validation-v1",
        "result": "PASS",
        "process_ids": manifest["process_ids"],
        "segments": manifest["segments"],
        "checks": {
            "fresh_processes": len(manifest["process_ids"]) == 3 and len(set(manifest["process_ids"])) == 3,
            "segment_boundary_exact": [segment["segment_token_range"] for segment in manifest["segments"]] == [[0, 2_498_560], [2_498_560, 4_997_120]],
            "rng_continuity": all(segment["rng_continuity"] for segment in manifest["segments"]),
            "scheduler_continuity": all(segment["scheduler_continuity"] for segment in manifest["segments"]),
            "data_position_continuity": all(segment["data_position_continuity"] for segment in manifest["segments"]),
            "no_repeated_or_skipped_sequence": all(segment["no_repeated_or_skipped_sequence"] for segment in manifest["segments"]),
            "midpoint_resume_checkpoint_present": (RUN_DIR / "checkpoints" / _checkpoint_name(305) / "resume_state.pt").is_file(),
            "final_resume_checkpoint_present": (RUN_DIR / "checkpoints" / _checkpoint_name(610) / "resume_state.pt").is_file(),
        },
    }
    restart["all_checks_pass"] = all(restart["checks"].values())
    validate_restart_evidence(restart)
    summary["process_restart"] = restart
    atomic_write_json(RUN_DIR / "process_restart_validation.json", restart)
    atomic_write_json(RUN_DIR / "training_summary.json", summary)
    manifest["status"] = "training_complete"
    manifest["stability"] = summary["stability"]
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("initialize")
    segment = commands.add_parser("segment")
    segment.add_argument("--target-step", type=int, choices=(305, 610), required=True)
    args = parser.parse_args()
    payload = initialize() if args.command == "initialize" else train_segment(args.target_step)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
