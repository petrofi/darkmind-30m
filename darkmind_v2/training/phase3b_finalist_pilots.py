"""Equal-data learning-rate calibration and forced-resume Phase 3B pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_phase3b_pilot_config import (
    EXPECTED_TOKENIZED_MANIFEST_HASH,
    load_and_validate_phase3b_pilot_config,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "darkmind_v2" / "config" / "phase3b_finalist_pilot.json"
RUNTIME_ROOT = ROOT / "darkmind_v2" / "data" / "phase3b"
CALIBRATION_ROOT = RUNTIME_ROOT / "calibration"
SELECTED_LR_PATH = CALIBRATION_ROOT / "selected_learning_rates.json"
REPORT_DIR = ROOT / "darkmind_v2" / "reports"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finalist_config(candidate: str, contract: dict[str, Any]) -> DarkMindV2Config:
    payload = json.loads((ROOT / contract["candidates"][candidate]).read_text(encoding="utf-8"))
    payload.update(
        {
            "seed": contract["initialization_seed"],
            "attention_implementation": contract["attention_implementation"],
            "gradient_checkpointing": contract["gradient_checkpointing"],
        }
    )
    return DarkMindV2Config(**payload)


def learning_rate_for_step(
    step: int,
    *,
    peak: float,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if not 1 <= step <= total_steps:
        raise ValueError("learning-rate step is outside the authorized horizon")
    if step <= warmup_steps:
        return peak * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    minimum = peak * minimum_ratio
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: DarkMindV2ForCausalLM, peak_lr: float, contract: dict[str, Any]) -> torch.optim.AdamW:
    values = contract["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(values["beta1"], values["beta2"]),
        weight_decay=values["weight_decay"],
        foreach=False,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_for_step(
            min(epoch + 1, total_steps),
            peak=peak_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=minimum_ratio,
        )
        / peak_lr,
    )


@torch.no_grad()
def evaluate_loss(
    model: DarkMindV2ForCausalLM,
    dataset: TokenShardDataset,
    *,
    sequence_length: int,
    micro_batch_size: int,
    device: torch.device,
    maximum_tokens: int | None = None,
) -> dict[str, float | int]:
    available_sequences = dataset.total_tokens // sequence_length
    requested_sequences = (
        available_sequences
        if maximum_tokens is None
        else min(available_sequences, maximum_tokens // sequence_length)
    )
    if requested_sequences <= 0:
        raise ValueError("evaluation view has no complete sequence")
    model.eval()
    weighted_loss = 0.0
    completed = 0
    while completed < requested_sequences:
        batch_size = min(micro_batch_size, requested_sequences - completed)
        batch = dataset.batch(
            offset=completed * sequence_length,
            micro_batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise FloatingPointError("non-finite evaluation loss")
        weighted_loss += float(output.loss) * batch_size
        completed += batch_size
    model.train()
    loss = weighted_loss / requested_sequences
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 80.0)),
        "sequences": requested_sequences,
        "evaluated_tokens": requested_sequences * sequence_length,
    }


def train_optimizer_step(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    dataset: TokenShardDataset,
    *,
    data_position: int,
    contract: dict[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    data = contract["data"]
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for micro_step in range(data["gradient_accumulation_steps"]):
        offset = data_position + micro_step * data["micro_batch_size"] * data["sequence_length"]
        batch = dataset.batch(
            offset=offset,
            micro_batch_size=data["micro_batch_size"],
            sequence_length=data["sequence_length"],
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("non-finite Phase 3B training loss")
            loss = output.loss / data["gradient_accumulation_steps"]
        loss.backward()
        losses.append(float(output.loss.detach()))
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=contract["optimizer"]["gradient_clipping"]
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite Phase 3B gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    return statistics.fmean(losses), float(gradient_norm.detach())


def slope(values: list[float]) -> float:
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def calibration_worker(candidate: str, learning_rate: float, output: Path, config_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration result: {output}")
    progress_path = output.with_suffix(".progress.json")
    contract = load_and_validate_phase3b_pilot_config(config_path)
    atomic_write_json(progress_path, {"operation": "contract_validated", "candidate": candidate})
    set_deterministic_seed(contract["initialization_seed"])
    verify_frozen_tokenizer()
    device = torch.device("cuda")
    model_config = finalist_config(candidate, contract)
    validate_training_environment(
        model_config,
        micro_batch_size=contract["data"]["micro_batch_size"],
        precision=contract["precision"],
    )
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    atomic_write_json(progress_path, {"operation": "model_created", "candidate": candidate})
    optimizer = build_optimizer(model, learning_rate, contract)
    calibration = contract["calibration"]
    scheduler = build_scheduler(
        optimizer,
        peak_lr=learning_rate,
        total_steps=calibration["optimizer_steps"],
        warmup_steps=calibration["warmup_optimizer_steps"],
        minimum_ratio=contract["schedule"]["minimum_learning_rate_ratio"],
    )
    data_dir = ROOT / contract["data"]["tokenized_dir"]
    train_data = TokenShardDataset(data_dir, "train")
    validation_data = TokenShardDataset(data_dir, "validation")
    atomic_write_json(progress_path, {"operation": "before_initial_validation", "candidate": candidate})
    initial = evaluate_loss(
        model,
        validation_data,
        sequence_length=contract["data"]["sequence_length"],
        micro_batch_size=contract["data"]["micro_batch_size"],
        device=device,
        maximum_tokens=calibration["validation_subset_tokens"],
    )
    atomic_write_json(
        progress_path,
        {"operation": "initial_validation_complete", "loss": initial["loss"], "candidate": candidate},
    )
    losses: list[float] = []
    gradients: list[float] = []
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(calibration["optimizer_steps"]):
        atomic_write_json(
            progress_path,
            {"operation": "before_optimizer_step", "optimizer_step": step + 1, "candidate": candidate},
        )
        step_started = time.perf_counter()
        loss, gradient = train_optimizer_step(
            model,
            optimizer,
            train_data,
            data_position=step * contract["data"]["effective_tokens_per_optimizer_step"],
            contract=contract,
            device=device,
        )
        scheduler.step()
        losses.append(loss)
        gradients.append(gradient)
        durations.append(time.perf_counter() - step_started)
        atomic_write_json(
            progress_path,
            {
                "operation": "optimizer_step_complete",
                "optimizer_step": step + 1,
                "loss": loss,
                "gradient_norm": gradient,
                "candidate": candidate,
            },
        )
    wall_seconds = time.perf_counter() - started
    atomic_write_json(progress_path, {"operation": "before_final_validation", "candidate": candidate})
    final = evaluate_loss(
        model,
        validation_data,
        sequence_length=contract["data"]["sequence_length"],
        micro_batch_size=contract["data"]["micro_batch_size"],
        device=device,
        maximum_tokens=calibration["validation_subset_tokens"],
    )
    atomic_write_json(
        progress_path,
        {"operation": "final_validation_complete", "loss": final["loss"], "candidate": candidate},
    )
    overshoot = final["loss"] >= initial["loss"] or statistics.fmean(losses[-8:]) > statistics.fmean(losses[-16:-8]) * 1.05
    result = {
        "schema_version": "darkmind-v2-phase3b-lr-calibration-result-v1",
        "candidate": candidate,
        "learning_rate": learning_rate,
        "seed": contract["initialization_seed"],
        "tokens": calibration["tokens"],
        "optimizer_steps": calibration["optimizer_steps"],
        "initial_validation_loss": initial["loss"],
        "final_validation_loss": final["loss"],
        "validation_improvement": initial["loss"] - final["loss"],
        "training_loss_slope_per_step": slope(losses),
        "initial_training_loss": losses[0],
        "final_training_loss": losses[-1],
        "gradient_norm_min": min(gradients),
        "gradient_norm_max": max(gradients),
        "gradient_norm_p95": sorted(gradients)[math.ceil(0.95 * len(gradients)) - 1],
        "non_finite_events": 0,
        "tokens_per_second": calibration["tokens"] / wall_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "wall_seconds": wall_seconds,
        "overshoot_warning": overshoot,
        "stable": not overshoot and all(math.isfinite(value) for value in losses + gradients),
        "data_range": [0, calibration["tokens"]],
        "result": "PASS",
    }
    atomic_write_json(output, result)
    atomic_write_json(progress_path, {"operation": "worker_complete", "candidate": candidate})
    return result


def run_calibrations(config_path: Path) -> dict[str, Any]:
    contract = load_and_validate_phase3b_pilot_config(config_path)
    if SELECTED_LR_PATH.exists():
        raise FileExistsError(f"refusing to overwrite learning-rate selection: {SELECTED_LR_PATH}")
    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"C": [], "D": []}
    for candidate in results:
        for learning_rate in contract["calibration"]["learning_rates"]:
            label = f"candidate_{candidate.lower()}_lr_{learning_rate:.4f}".replace(".", "p")
            output = CALIBRATION_ROOT / f"{label}.json"
            if output.is_file():
                existing = json.loads(output.read_text(encoding="utf-8"))
                expected = {
                    "candidate": candidate,
                    "learning_rate": learning_rate,
                    "tokens": contract["calibration"]["tokens"],
                    "optimizer_steps": contract["calibration"]["optimizer_steps"],
                    "result": "PASS",
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"incompatible completed calibration result: {output}")
                results[candidate].append(existing)
                continue
            command = [
                sys.executable,
                "-m",
                "darkmind_v2.training.phase3b_finalist_pilots",
                "calibration-worker",
                "--candidate",
                candidate,
                "--learning-rate",
                str(learning_rate),
                "--output",
                str(output),
                "--config",
                str(config_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
            if completed.returncode or not output.is_file():
                progress_path = output.with_suffix(".progress.json")
                progress = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file()
                    else None
                )
                raise RuntimeError(
                    f"calibration worker failed for {candidate} lr={learning_rate}: "
                    f"exit={completed.returncode} progress={progress} stderr={completed.stderr[-2000:]}"
                )
            results[candidate].append(json.loads(output.read_text(encoding="utf-8")))
    selections: dict[str, float] = {}
    for candidate, records in results.items():
        stable = [record for record in records if record["stable"] and record["non_finite_events"] == 0]
        if not stable:
            raise RuntimeError(f"no stable learning-rate calibration for Candidate {candidate}")
        selected = min(stable, key=lambda item: item["final_validation_loss"])
        selections[candidate] = selected["learning_rate"]
    payload = {
        "schema_version": "darkmind-v2-phase3b-lr-calibration-selection-v1",
        "results": results,
        "selected_learning_rates": selections,
        "selection_policy": [
            "finite and stable behavior",
            "strongest validation improvement",
            "no exploding-gradient trend",
            "no overshoot warning",
        ],
        "result": "PASS",
    }
    atomic_write_json(SELECTED_LR_PATH, payload)
    lines = [
        "# Phase 3B Learning-Rate Calibration",
        "",
        "Each run used the same seed, first 524,288 ordered train tokens, validation subset, optimizer, "
        "effective batch, warmup, and cosine horizon.",
        "",
        "| Candidate | Peak LR | Initial val | Final val | Improvement | Slope/step | Grad p95 | tok/s | VRAM GiB | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate, records in results.items():
        for item in records:
            lines.append(
                f"| {candidate} | {item['learning_rate']:.4f} | {item['initial_validation_loss']:.6f} | "
                f"{item['final_validation_loss']:.6f} | {item['validation_improvement']:.6f} | "
                f"{item['training_loss_slope_per_step']:.6f} | {item['gradient_norm_p95']:.4f} | "
                f"{item['tokens_per_second']:,.1f} | {item['peak_reserved_bytes'] / 2**30:.2f} | "
                f"{'PASS' if item['stable'] else 'WARN'} |"
            )
    lines.extend(
        [
            "",
            f"Selected Candidate C peak LR: `{selections['C']:.4f}`",
            "",
            f"Selected Candidate D peak LR: `{selections['D']:.4f}`",
            "",
        ]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase3b_learning_rate_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def checkpoint_name(step: int, tokens: int) -> str:
    return f"step_{step:06d}_tokens_{tokens:09d}"


def rng_fingerprint(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode("utf-8"))
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(numpy_state[2:]).encode("utf-8"))
    digest.update(state["torch_cpu"].cpu().numpy().tobytes())
    for tensor in state.get("torch_cuda", []):
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def initialize_pilot(candidate: str, config_path: Path) -> dict[str, Any]:
    contract = load_and_validate_phase3b_pilot_config(config_path)
    selection = json.loads(SELECTED_LR_PATH.read_text(encoding="utf-8"))
    peak_lr = float(selection["selected_learning_rates"][candidate])
    run_dir = ROOT / contract["runs"][candidate]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable Phase 3B run: {run_dir}")
    data_dir = ROOT / contract["data"]["tokenized_dir"]
    manifest = json.loads((data_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("deterministic_content_hash") != EXPECTED_TOKENIZED_MANIFEST_HASH:
        raise ValueError("validated tokenized corpus content hash changed")
    verify_frozen_tokenizer()
    set_deterministic_seed(contract["initialization_seed"])
    device = torch.device("cuda")
    model_config = finalist_config(candidate, contract)
    validate_training_environment(
        model_config,
        micro_batch_size=contract["data"]["micro_batch_size"],
        precision=contract["precision"],
    )
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, peak_lr, contract)
    scheduler = build_scheduler(
        optimizer,
        peak_lr=peak_lr,
        total_steps=contract["pilot"]["maximum_optimizer_steps"],
        warmup_steps=contract["schedule"]["warmup_optimizer_steps"],
        minimum_ratio=contract["schedule"]["minimum_learning_rate_ratio"],
    )
    validation_data = TokenShardDataset(data_dir, "validation")
    eval_data = TokenShardDataset(data_dir, "eval")
    initial_validation = evaluate_loss(
        model,
        validation_data,
        sequence_length=contract["data"]["sequence_length"],
        micro_batch_size=contract["data"]["micro_batch_size"],
        device=device,
    )
    initial_eval = evaluate_loss(
        model,
        eval_data,
        sequence_length=contract["data"]["sequence_length"],
        micro_batch_size=contract["data"]["micro_batch_size"],
        device=device,
    )
    for name in ("checkpoints", "validations", "eval", "audits", "resume"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    checkpoint = run_dir / "checkpoints" / checkpoint_name(0, 0)
    state = TrainingState(
        step=0,
        tokens_seen=0,
        data_position=0,
        best_validation_loss=float(initial_validation["loss"]),
        last_validation_loss=float(initial_validation["loss"]),
        best_checkpoint=str(checkpoint),
    )
    metadata = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=tokenized_manifest_hash(data_dir),
    )
    atomic_write_json(run_dir / "validations" / "step_000000.json", initial_validation)
    atomic_write_json(run_dir / "eval" / "step_000000.json", initial_eval)
    run_manifest = {
        "schema_version": "darkmind-v2-phase3b-finalist-run-v1",
        "candidate": candidate,
        "parameters": model.parameter_count(),
        "peak_learning_rate": peak_lr,
        "initialization_seed": contract["initialization_seed"],
        "tokenized_corpus_manifest_sha256": tokenized_manifest_hash(data_dir),
        "tokenized_corpus_content_hash": manifest["deterministic_content_hash"],
        "data_order": "contiguous offsets from zero, identical for C and D",
        "pilot_tokens": contract["pilot"]["maximum_total_training_tokens"],
        "pilot_steps": contract["pilot"]["maximum_optimizer_steps"],
        "checkpoint_steps": contract["pilot"]["checkpoint_steps"],
        "initial_checkpoint": str(checkpoint),
        "latest_checkpoint": str(checkpoint),
        "best_checkpoint": str(checkpoint),
        "best_validation_loss": initial_validation["loss"],
        "checkpoints": {"0": str(checkpoint)},
        "segment_ranges": [],
        "checkpoint_model_hashes": {"0": metadata["model_files"]},
        "status": "initialized",
    }
    atomic_write_json(run_dir / "resolved_model_config.json", model_config.architecture_dict())
    atomic_write_json(run_dir / "resolved_pilot_contract.json", contract)
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest


def train_segment(candidate: str, checkpoint: Path, target_step: int, config_path: Path) -> dict[str, Any]:
    contract = load_and_validate_phase3b_pilot_config(config_path)
    if target_step not in {
        contract["pilot"]["segment_a_target_step"],
        contract["pilot"]["segment_b_target_step"],
    }:
        raise ValueError("target is outside the authorized forced-restart boundaries")
    run_dir = ROOT / contract["runs"][candidate]
    progress_path = run_dir / "resume" / f"segment_to_step_{target_step:06d}.progress.json"
    atomic_write_json(progress_path, {"operation": "contract_validated", "candidate": candidate})
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    peak_lr = float(run_manifest["peak_learning_rate"])
    data_dir = ROOT / contract["data"]["tokenized_dir"]
    data_hash = tokenized_manifest_hash(data_dir)
    if data_hash != run_manifest["tokenized_corpus_manifest_sha256"]:
        raise ValueError("tokenized corpus changed after pilot initialization")
    verify_frozen_tokenizer()
    set_deterministic_seed(contract["initialization_seed"])
    device = torch.device("cuda")
    model_config = finalist_config(candidate, contract)
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    atomic_write_json(progress_path, {"operation": "model_created", "candidate": candidate})
    optimizer = build_optimizer(model, peak_lr, contract)
    scheduler = build_scheduler(
        optimizer,
        peak_lr=peak_lr,
        total_steps=contract["pilot"]["maximum_optimizer_steps"],
        warmup_steps=contract["schedule"]["warmup_optimizer_steps"],
        minimum_ratio=contract["schedule"]["minimum_learning_rate_ratio"],
    )
    atomic_write_json(progress_path, {"operation": "before_resume_state_read", "candidate": candidate})
    resume_payload = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    atomic_write_json(progress_path, {"operation": "resume_state_read", "candidate": candidate})
    expected_rng_fingerprint = rng_fingerprint(resume_payload["rng"])
    atomic_write_json(progress_path, {"operation": "before_checkpoint_load", "candidate": candidate})
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=data_hash,
    )
    atomic_write_json(
        progress_path,
        {"operation": "checkpoint_loaded", "candidate": candidate, "optimizer_step": state.step},
    )
    actual_rng_fingerprint = rng_fingerprint(capture_rng_state())
    tokens_per_step = contract["data"]["effective_tokens_per_optimizer_step"]
    if (state.tokens_seen, state.data_position) != (state.step * tokens_per_step, state.step * tokens_per_step):
        raise ValueError("resume state step/token/data position mismatch")
    if state.step >= target_step:
        raise ValueError("checkpoint already reached the requested segment target")
    expected_lr = learning_rate_for_step(
        state.step + 1,
        peak=peak_lr,
        total_steps=contract["pilot"]["maximum_optimizer_steps"],
        warmup_steps=contract["schedule"]["warmup_optimizer_steps"],
        minimum_ratio=contract["schedule"]["minimum_learning_rate_ratio"],
    )
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, abs_tol=1e-15):
        raise ValueError("scheduler did not resume at the exact next optimizer step")
    if scheduler.last_epoch != state.step:
        raise ValueError("scheduler epoch does not match resumed optimizer step")
    if expected_rng_fingerprint != actual_rng_fingerprint:
        raise ValueError("RNG continuity failed on fresh-process resume")
    atomic_write_json(
        progress_path,
        {
            "operation": "resume_continuity_validated",
            "candidate": candidate,
            "optimizer_step": state.step,
            "next_optimizer_step": state.step + 1,
            "data_position": state.data_position,
        },
    )

    train_data = TokenShardDataset(data_dir, "train")
    validation_data = TokenShardDataset(data_dir, "validation")
    eval_data = TokenShardDataset(data_dir, "eval")
    segment_start_step = state.step
    segment_start_position = state.data_position
    checkpoint_steps = set(contract["pilot"]["checkpoint_steps"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while state.step < target_step:
        atomic_write_json(
            progress_path,
            {
                "operation": "before_optimizer_step",
                "candidate": candidate,
                "optimizer_step": state.step + 1,
                "data_position": state.data_position,
            },
        )
        step_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        loss, gradient_norm = train_optimizer_step(
            model,
            optimizer,
            train_data,
            data_position=state.data_position,
            contract=contract,
            device=device,
        )
        scheduler.step()
        state.step += 1
        state.tokens_seen += tokens_per_step
        state.data_position += tokens_per_step
        atomic_write_json(
            progress_path,
            {
                "operation": "optimizer_step_complete",
                "candidate": candidate,
                "optimizer_step": state.step,
                "data_position": state.data_position,
            },
        )
        if state.data_position > contract["pilot"]["maximum_total_training_tokens"]:
            raise ValueError("pilot attempted to exceed its authorized token budget")
        state.last_training_loss = loss
        state.smoothed_training_loss = (
            loss if state.smoothed_training_loss is None else 0.9 * state.smoothed_training_loss + 0.1 * loss
        )
        duration = time.perf_counter() - step_started
        metric: dict[str, Any] = {
            "optimizer_step": state.step,
            "consumed_tokens": state.tokens_seen,
            "data_position": state.data_position,
            "training_loss": loss,
            "smoothed_training_loss": state.smoothed_training_loss,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "step_duration_seconds": duration,
            "tokens_per_second": tokens_per_step / duration,
        }
        if state.step in checkpoint_steps:
            validation = evaluate_loss(
                model,
                validation_data,
                sequence_length=contract["data"]["sequence_length"],
                micro_batch_size=contract["data"]["micro_batch_size"],
                device=device,
            )
            evaluation = evaluate_loss(
                model,
                eval_data,
                sequence_length=contract["data"]["sequence_length"],
                micro_batch_size=contract["data"]["micro_batch_size"],
                device=device,
            )
            checkpoint_path = run_dir / "checkpoints" / checkpoint_name(state.step, state.tokens_seen)
            state.last_validation_loss = float(validation["loss"])
            if state.best_validation_loss is None or float(validation["loss"]) < state.best_validation_loss:
                state.best_validation_loss = float(validation["loss"])
                state.best_checkpoint = str(checkpoint_path)
            metadata = save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state=state,
                tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
                data_manifest_hash=data_hash,
            )
            model_path = checkpoint_path / "model" / "model.safetensors"
            if sha256_file(model_path) != metadata["model_files"]["model_sha256"]:
                raise ValueError("saved safetensors hash mismatch")
            atomic_write_json(run_dir / "validations" / f"step_{state.step:06d}.json", validation)
            atomic_write_json(run_dir / "eval" / f"step_{state.step:06d}.json", evaluation)
            run_manifest["checkpoints"][str(state.step)] = str(checkpoint_path)
            run_manifest["checkpoint_model_hashes"][str(state.step)] = metadata["model_files"]
            run_manifest["latest_checkpoint"] = str(checkpoint_path)
            run_manifest["best_checkpoint"] = state.best_checkpoint
            run_manifest["best_validation_loss"] = state.best_validation_loss
            metric.update(
                {
                    "validation_loss": validation["loss"],
                    "eval_loss": evaluation["loss"],
                    "checkpoint": str(checkpoint_path),
                }
            )
            atomic_write_json(run_dir / "run_manifest.json", run_manifest)
        append_jsonl(run_dir / "metrics.jsonl", metric)
        if state.step % 25 == 0 or state.step in checkpoint_steps:
            print(
                f"candidate={candidate} step={state.step} tokens={state.tokens_seen} "
                f"loss={loss:.6f} tok_s={metric['tokens_per_second']:.1f}",
                flush=True,
            )

    expected_end = target_step * tokens_per_step
    if (state.tokens_seen, state.data_position) != (expected_end, expected_end):
        raise ValueError("segment ended at an inconsistent data position")
    segment_range = [segment_start_position, state.data_position]
    prior_ranges = run_manifest["segment_ranges"]
    if prior_ranges and prior_ranges[-1][1] != segment_range[0]:
        raise ValueError("segment ranges contain a repeated or skipped token")
    if not prior_ranges and segment_range[0] != 0:
        raise ValueError("first segment did not begin at token zero")
    run_manifest["segment_ranges"].append(segment_range)
    run_manifest["optimizer_steps"] = state.step
    run_manifest["consumed_tokens"] = state.tokens_seen
    run_manifest["data_position"] = state.data_position
    run_manifest["status"] = "midpoint_complete" if target_step == 305 else "pilot_complete"
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    summary = {
        "schema_version": "darkmind-v2-phase3b-pilot-segment-v1",
        "candidate": candidate,
        "segment_start_step": segment_start_step,
        "segment_end_step": state.step,
        "segment_token_range": segment_range,
        "next_optimizer_step": state.step + 1 if state.step < 610 else None,
        "rng_continuity": expected_rng_fingerprint == actual_rng_fingerprint,
        "scheduler_continuity": True,
        "data_position_continuity": True,
        "no_repeated_or_skipped_sequence": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "latest_checkpoint": run_manifest["latest_checkpoint"],
        "result": "PASS",
    }
    atomic_write_json(run_dir / "resume" / f"segment_to_step_{target_step:06d}.json", summary)
    atomic_write_json(
        progress_path,
        {"operation": "segment_complete", "candidate": candidate, "optimizer_step": state.step},
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate-all")
    calibrate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    worker = commands.add_parser("calibration-worker")
    worker.add_argument("--candidate", choices=("C", "D"), required=True)
    worker.add_argument("--learning-rate", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--candidate", choices=("C", "D"), required=True)
    initialize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    segment = commands.add_parser("segment")
    segment.add_argument("--candidate", choices=("C", "D"), required=True)
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--target-step", type=int, required=True)
    segment.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "calibrate-all":
        result = run_calibrations(args.config)
    elif args.command == "calibration-worker":
        result = calibration_worker(args.candidate, args.learning_rate, args.output, args.config)
    elif args.command == "initialize":
        result = initialize_pilot(args.candidate, args.config)
    else:
        result = train_segment(args.candidate, args.checkpoint, args.target_step, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
