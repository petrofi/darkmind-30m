"""Calibrate and train the immutable Base V1 Stage-1 gate on Corpus V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import (
    atomic_write_json as _atomic_write_json,
    canonical_json_hash,
    sha256_file,
)
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from darkmind_v2.training.phase3b_finalist_pilots import append_jsonl, evaluate_loss, rng_fingerprint
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_phase4a_config import (
    DEFAULT_CONFIG,
    STAGE1_STEPS,
    STAGE1_TOKENS,
    TOKENS_PER_STEP,
    learning_rate_for_step,
    load_and_validate_phase4a_config,
)
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    ROOT,
    sha256_file as preflight_sha256_file,
)


PREFLIGHT_PASS2 = ROOT / "darkmind_v2" / "data" / "phase4a" / "preflight" / "pass2.json"


def atomic_write_json(path: Path, payload: Any) -> None:
    for attempt in range(20):
        try:
            _atomic_write_json(path, payload)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def build_optimizer(model: DarkMindV2ForCausalLM, config: dict[str, Any]) -> torch.optim.AdamW:
    values = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["schedule"]["peak_learning_rate"]),
        betas=(float(values["beta1"]), float(values["beta2"])),
        eps=float(values["epsilon"]),
        weight_decay=float(values["weight_decay"]),
        foreach=False,
        fused=False,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    peak = float(config["schedule"]["peak_learning_rate"])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_for_step(
            min(epoch + 1, int(config["schedule"]["scheduler_horizon_optimizer_steps"])), config
        )
        / peak,
    )


def load_model_config(config: dict[str, Any]) -> DarkMindV2Config:
    model_config = DarkMindV2Config.from_json_file(ROOT / config["model_config"])
    if model_config_hash(model_config) != EXPECTED_ARCHITECTURE_HASH:
        raise ValueError("frozen Base V1 architecture hash mismatch")
    return model_config


def verify_preflight_snapshot() -> dict[str, Any]:
    report = json.loads(PREFLIGHT_PASS2.read_text(encoding="utf-8"))
    if report.get("result") != "PASS" or report.get("cross_pass_asset_identity") is not True:
        raise ValueError("two-pass immutable preflight is not complete")
    tokenized_dir = ROOT / "darkmind_v2" / "data" / "phase3c1" / "tokenized" / "tranche1_v2"
    for filename, expected in report["corpus"]["asset_snapshot"].items():
        path = tokenized_dir / filename
        stat = path.stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Corpus V3 runtime file metadata changed: {filename}")
        if filename.endswith(".bin") and preflight_sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Corpus V3 shard hash changed: {filename}")
    return report


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def gpu_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            timeout=10,
        ).strip().splitlines()[0]
        name, utilization, temperature, power, used, total, driver = [item.strip() for item in output.split(",")]
        return {
            "available": True,
            "name": name,
            "utilization_percent": float(utilization),
            "temperature_c": float(temperature),
            "power_w": float(power),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "driver_version": driver,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


class GpuMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = gpu_snapshot()
        self.samples: list[dict[str, Any]] = [self._latest]
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(2.0):
            sample = gpu_snapshot()
            with self._lock:
                self._latest = sample
                self.samples.append(sample)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "gpu": gpu_snapshot(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_short": git_output("status", "--short"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def sequence_provenance(config: dict[str, Any]) -> dict[str, Any]:
    tokenized_dir = ROOT / config["corpus"]["tokenized_dir"]
    manifest = json.loads((tokenized_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    train_shards = [item for item in manifest["shards"] if item["split"] == "train"]
    shard_global_starts: dict[str, int] = {}
    total = 0
    for shard in train_shards:
        shard_global_starts[shard["filename"]] = total
        total += int(shard["tokens"])
    targets: dict[tuple[str, int], list[str]] = {}
    milestones: dict[str, dict[str, Any]] = {}
    for step in config["evaluation"]["milestone_steps"]:
        consumed_sequences = step * TOKENS_PER_STEP // config["data"]["sequence_length"]
        entry: dict[str, Any] = {
            "optimizer_step": step,
            "consumed_tokens": step * TOKENS_PER_STEP,
            "consumed_sequence_range": [0, consumed_sequences],
            "last_consumed_sequence_index": consumed_sequences - 1 if consumed_sequences else None,
            "next_sequence_index": consumed_sequences,
        }
        milestones[str(step)] = entry
        for label, sequence_index in (
            ("last_consumed_document", consumed_sequences - 1 if consumed_sequences else None),
            ("next_document", consumed_sequences),
        ):
            if sequence_index is None:
                continue
            global_offset = sequence_index * config["data"]["sequence_length"]
            for shard in train_shards:
                start = shard_global_starts[shard["filename"]]
                if start <= global_offset < start + shard["tokens"]:
                    targets.setdefault((shard["filename"], global_offset - start), []).append(f"{step}:{label}")
                    break
    with (tokenized_dir / manifest["document_boundaries"]["filename"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            boundary = json.loads(line)
            if boundary["split"] != "train":
                continue
            shard = boundary["shard"]
            relevant = [item for item in targets if item[0] == shard and boundary["start_offset"] <= item[1] < boundary["end_offset"]]
            for key in relevant:
                for label in targets.pop(key):
                    step, field = label.split(":", 1)
                    milestones[step][field] = {
                        "document_id": boundary["id"],
                        "source_id": boundary["allocation_source_id"],
                        "language": boundary["language"],
                        "category": boundary["category"],
                        "text_sha256": boundary["text_sha256"],
                        "shard": shard,
                        "local_token_offset": key[1],
                    }
            if not targets:
                break
    if targets:
        raise ValueError(f"failed to resolve milestone provenance: {sorted(targets)[:3]}")
    core = {
        "schema_version": "darkmind-v2-phase4a-sequence-order-v1",
        "data_order_seed": config["data"]["data_order_seed"],
        "rule": config["data"]["sequence_order"],
        "sequence_length": config["data"]["sequence_length"],
        "available_complete_sequences": config["corpus"]["train_complete_sequence_tokens"] // config["data"]["sequence_length"],
        "available_complete_sequence_tokens": config["corpus"]["train_complete_sequence_tokens"],
        "unused_train_tail_tokens": config["corpus"]["train_tail_tokens"],
        "no_replacement": True,
        "no_wrap": True,
        "milestones": milestones,
    }
    return {**core, "deterministic_content_hash": canonical_json_hash(core)}


def checkpoint_name(step: int, tokens: int) -> str:
    return f"step_{step:06d}_tokens_{tokens:09d}"


def save_checkpoint_atomic(
    checkpoint: Path,
    *,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    data_hash: str,
    sequence_order_hash: str,
) -> dict[str, Any]:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite immutable checkpoint: {checkpoint}")
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete checkpoint requires inspection: {temporary}")
    metadata = save_checkpoint(
        temporary,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_hash,
    )
    from safetensors import safe_open

    weights = temporary / "model" / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise ValueError("empty safetensors checkpoint")
    files = {
        relative(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    }
    checkpoint_manifest = {
        "schema_version": "darkmind-v2-phase4a-checkpoint-v1",
        "result": "PASS",
        "optimizer_step": state.step,
        "consumed_tokens": state.tokens_seen,
        "next_sequence_index": state.data_position // 512,
        "learning_rate_for_next_step": optimizer.param_groups[0]["lr"],
        "scheduler_last_epoch": scheduler.last_epoch,
        "sequence_order_hash": sequence_order_hash,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "files": files,
    }
    atomic_write_json(temporary / "checkpoint_manifest.json", checkpoint_manifest)
    os.replace(temporary, checkpoint)
    metadata["checkpoint_manifest"] = checkpoint_manifest
    return metadata


def profiled_optimizer_step(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    dataset: TokenShardDataset,
    *,
    data_position: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    data_wait = 0.0
    transfer_time = 0.0
    source_sequence_indices: list[int] = []
    started = time.perf_counter()
    for micro_step in range(config["data"]["gradient_accumulation_steps"]):
        offset = data_position + micro_step * config["data"]["micro_batch_size"] * config["data"]["sequence_length"]
        read_started = time.perf_counter()
        values = dataset.read(offset, config["data"]["micro_batch_size"] * config["data"]["sequence_length"])
        data_wait += time.perf_counter() - read_started
        transfer_started = time.perf_counter()
        batch = torch.from_numpy(values).to(device=device, dtype=torch.long).view(
            config["data"]["micro_batch_size"], config["data"]["sequence_length"]
        )
        torch.cuda.synchronize(device)
        transfer_time += time.perf_counter() - transfer_started
        first = offset // config["data"]["sequence_length"]
        source_sequence_indices.extend(range(first, first + config["data"]["micro_batch_size"]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("non-finite Phase 4A training loss")
            (output.loss / config["data"]["gradient_accumulation_steps"]).backward()
        losses.append(float(output.loss.detach()))
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["gradient_clipping"])
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite Phase 4A gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    return {
        "raw_train_loss": statistics.fmean(losses),
        "gradient_norm": float(gradient_norm.detach()),
        "optimizer_step_duration_seconds": duration,
        "data_loader_wait_seconds": data_wait,
        "host_to_device_seconds": transfer_time,
        "active_tokens_per_second": TOKENS_PER_STEP / duration,
        "source_sequence_indices": source_sequence_indices,
    }


def calibration(config_path: Path) -> dict[str, Any]:
    config = load_and_validate_phase4a_config(config_path, check_runtime_assets=True)
    verify_preflight_snapshot()
    verify_frozen_tokenizer()
    output_dir = ROOT / config["calibration"]["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output_dir}")
    output_dir.mkdir(parents=True)
    set_deterministic_seed(config["initialization_seed"])
    device = torch.device("cuda")
    model_config = load_model_config(config)
    validate_training_environment(model_config, micro_batch_size=2, precision="bf16")
    wall_started = time.perf_counter()
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    dataset = TokenShardDataset(ROOT / config["corpus"]["tokenized_dir"], "train")
    warmup = config["calibration"]["warmup_optimizer_steps"]
    measured = config["calibration"]["measured_optimizer_steps"]
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, Any]] = []
    non_finite = 0
    try:
        for index in range(warmup + measured):
            lr = optimizer.param_groups[0]["lr"]
            metric = profiled_optimizer_step(
                model, optimizer, dataset, data_position=index * TOKENS_PER_STEP, config=config, device=device
            )
            scheduler.step()
            metric.update(
                {
                    "optimizer_step": index + 1,
                    "phase": "warmup" if index < warmup else "measured",
                    "learning_rate": lr,
                    "gpu": monitor.latest(),
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "optimizer_state_bytes": optimizer_state_bytes(optimizer),
                    "non_finite_events": non_finite,
                }
            )
            append_jsonl(output_dir / "metrics.jsonl", metric)
            if index >= warmup:
                records.append(metric)
            print(f"calibration step={index + 1}/{warmup + measured} loss={metric['raw_train_loss']:.6f} tok_s={metric['active_tokens_per_second']:.1f}", flush=True)
    finally:
        monitor.close()
    durations = [item["optimizer_step_duration_seconds"] for item in records]
    elapsed = time.perf_counter() - wall_started
    temperatures = [item["temperature_c"] for item in monitor.samples if item.get("available")]
    powers = [item["power_w"] for item in monitor.samples if item.get("available")]
    summary = {
        "schema_version": "darkmind-v2-phase4a-real-data-calibration-v1",
        "result": "PASS",
        "warmup_optimizer_steps": warmup,
        "measured_optimizer_steps": measured,
        "measured_tokens": measured * TOKENS_PER_STEP,
        "finite_forward_loss": all(math.isfinite(item["raw_train_loss"]) for item in records),
        "finite_backward_gradients": all(math.isfinite(item["gradient_norm"]) for item in records),
        "optimizer_success": True,
        "gradient_norm_min": min(item["gradient_norm"] for item in records),
        "gradient_norm_max": max(item["gradient_norm"] for item in records),
        "active_tokens_per_second": statistics.fmean(item["active_tokens_per_second"] for item in records),
        "full_wall_tokens_per_second": measured * TOKENS_PER_STEP / elapsed,
        "p50_step_seconds": percentile(durations, 0.50),
        "p95_step_seconds": percentile(durations, 0.95),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "mean_data_loader_wait_seconds": statistics.fmean(item["data_loader_wait_seconds"] for item in records),
        "mean_host_to_device_seconds": statistics.fmean(item["host_to_device_seconds"] for item in records),
        "gpu_temperature_c_min_max": [min(temperatures), max(temperatures)] if temperatures else None,
        "gpu_power_w_min_max": [min(powers), max(powers)] if powers else None,
        "oom": False,
        "process_stable": True,
        "non_finite_events": non_finite,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def initialize(config_path: Path) -> dict[str, Any]:
    config = load_and_validate_phase4a_config(config_path, check_runtime_assets=True)
    preflight = verify_preflight_snapshot()
    verify_frozen_tokenizer()
    calibration_path = ROOT / config["calibration"]["output_dir"] / "calibration_summary.json"
    calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration_report.get("result") != "PASS":
        raise ValueError("real-data calibration did not pass")
    run_dir = ROOT / config["run_dir"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable Phase 4A run: {run_dir}")
    for name in ("checkpoints", "validations", "eval", "audits", "resume", "provenance"):
        (run_dir / name).mkdir(parents=True, exist_ok=name != "checkpoints")
    atomic_write_json(run_dir / "resolved_model_config.json", json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8")))
    atomic_write_json(run_dir / "resolved_training_config.json", config)
    atomic_write_json(run_dir / "environment.json", environment_report())
    atomic_write_json(run_dir / "preflight_identity.json", {"pass1": "darkmind_v2/data/phase4a/preflight/pass1.json", "pass2": "darkmind_v2/data/phase4a/preflight/pass2.json", "cross_pass_asset_identity": preflight["cross_pass_asset_identity"]})
    order = sequence_provenance(config)
    atomic_write_json(run_dir / "sequence_order_manifest.json", order)
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    run_manifest: dict[str, Any] = {
        "schema_version": "darkmind-v2-phase4a-stage1-run-v1",
        "status": "initializing",
        "model": "darkmind-v2-base-v1",
        "fresh_deterministic_initialization": True,
        "finalist_pilot_weights_reused": False,
        "tiny_model_weights_reused": False,
        "corpus": "Corpus V3 aggregate 100M",
        "authorized_tokens": STAGE1_TOKENS,
        "authorized_optimizer_steps": STAGE1_STEPS,
        "scheduler_horizon_optimizer_steps": config["schedule"]["scheduler_horizon_optimizer_steps"],
        "scheduler_horizon_tokens": config["schedule"]["scheduler_horizon_tokens"],
        "instruction_tuned": False,
        "chatbot": False,
        "public_release_eligible": False,
        "huggingface_upload_authorized": False,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty_state": git_output("status", "--short"),
        "sequence_order_hash": order["deterministic_content_hash"],
        "checkpoints": {},
        "checkpoint_model_hashes": {},
        "segment_ranges": [],
        "wall_clock": {"initialization_started_unix": time.time()},
    }
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    set_deterministic_seed(config["initialization_seed"])
    device = torch.device("cuda")
    model_config = load_model_config(config)
    validate_training_environment(model_config, micro_batch_size=2, precision="bf16")
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    data_dir = ROOT / config["corpus"]["tokenized_dir"]
    train_data = TokenShardDataset(data_dir, "train")
    validation_data = TokenShardDataset(data_dir, "validation")
    eval_data = TokenShardDataset(data_dir, "eval")
    initial_train = evaluate_loss(model, train_data, sequence_length=512, micro_batch_size=2, device=device, maximum_tokens=TOKENS_PER_STEP)
    initial_validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
    initial_eval = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
    checkpoint = run_dir / "checkpoints" / checkpoint_name(0, 0)
    state = TrainingState(step=0, tokens_seen=0, data_position=0, best_validation_loss=float(initial_validation["loss"]), last_validation_loss=float(initial_validation["loss"]), best_checkpoint=relative(checkpoint))
    metadata = save_checkpoint_atomic(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        data_hash=tokenized_manifest_hash(data_dir),
        sequence_order_hash=order["deterministic_content_hash"],
    )
    atomic_write_json(run_dir / "validations" / "step_000000.json", initial_validation)
    atomic_write_json(run_dir / "eval" / "step_000000.json", initial_eval)
    atomic_write_json(run_dir / "provenance" / "initial_train_loss.json", initial_train)
    run_manifest.update(
        {
            "status": "initialized",
            "parameters": model.parameter_count(),
            "initialization_hash": metadata["model_files"]["model_sha256"],
            "initial_checkpoint": relative(checkpoint),
            "latest_checkpoint": relative(checkpoint),
            "best_checkpoint": relative(checkpoint),
            "best_validation_loss": initial_validation["loss"],
            "checkpoints": {"0": relative(checkpoint)},
            "checkpoint_model_hashes": {"0": metadata["model_files"]},
            "initial_train_loss": initial_train["loss"],
            "initial_validation_loss": initial_validation["loss"],
            "initial_eval_loss": initial_eval["loss"],
            "wall_clock": {**run_manifest["wall_clock"], "initialization_completed_unix": time.time()},
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return run_manifest


def train_segment(config_path: Path, checkpoint: Path, target_step: int) -> dict[str, Any]:
    config = load_and_validate_phase4a_config(config_path, check_runtime_assets=True)
    if target_step not in (305, 610):
        raise ValueError("target must be the forced midpoint or Stage-1 stop")
    verify_preflight_snapshot()
    verify_frozen_tokenizer()
    run_dir = ROOT / config["run_dir"]
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    order = json.loads((run_dir / "sequence_order_manifest.json").read_text(encoding="utf-8"))
    progress_path = run_dir / "resume" / f"segment_to_step_{target_step:06d}.progress.json"
    atomic_write_json(progress_path, {"operation": "fresh_process_started", "pid": os.getpid(), "target_step": target_step})
    data_dir = ROOT / config["corpus"]["tokenized_dir"]
    data_hash = tokenized_manifest_hash(data_dir)
    set_deterministic_seed(config["initialization_seed"])
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(load_model_config(config)).to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    resume_payload = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_rng = rng_fingerprint(resume_payload["rng"])
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=data_hash,
    )
    actual_rng = rng_fingerprint(capture_rng_state())
    if expected_rng != actual_rng:
        raise ValueError("RNG continuity failed")
    if state.tokens_seen != state.step * TOKENS_PER_STEP or state.data_position != state.tokens_seen:
        raise ValueError("checkpoint step/token/data position mismatch")
    if scheduler.last_epoch != state.step:
        raise ValueError("scheduler state did not resume")
    expected_lr = learning_rate_for_step(state.step + 1, config)
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, abs_tol=1e-15):
        raise ValueError("learning-rate continuity failed")
    if state.step >= target_step:
        raise ValueError("checkpoint already reached target")
    validation_data = TokenShardDataset(data_dir, "validation")
    stored_validation = json.loads((run_dir / "validations" / f"step_{state.step:06d}.json").read_text(encoding="utf-8"))
    reloaded_validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
    if not math.isclose(reloaded_validation["loss"], stored_validation["loss"], abs_tol=1e-7):
        raise ValueError("validation loss is not reproducible after fresh-process reload")
    atomic_write_json(
        progress_path,
        {
            "operation": "resume_continuity_validated",
            "loaded_step": state.step,
            "next_optimizer_step": state.step + 1,
            "consumed_tokens": state.tokens_seen,
            "next_sequence_index": state.data_position // 512,
            "rng_continuity": True,
            "scheduler_continuity": True,
            "validation_reproducible": True,
        },
    )
    train_data = TokenShardDataset(data_dir, "train")
    eval_data = TokenShardDataset(data_dir, "eval")
    milestones = set(config["evaluation"]["milestone_steps"])
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    segment_started = time.perf_counter()
    segment_start_step = state.step
    segment_start_position = state.data_position
    checkpoint_seconds = 0.0
    evaluation_seconds = 0.0
    non_finite_events = 0
    try:
        while state.step < target_step:
            learning_rate = optimizer.param_groups[0]["lr"]
            metric = profiled_optimizer_step(model, optimizer, train_data, data_position=state.data_position, config=config, device=device)
            scheduler.step()
            state.step += 1
            state.tokens_seen += TOKENS_PER_STEP
            state.data_position += TOKENS_PER_STEP
            if state.step > STAGE1_STEPS or state.tokens_seen > STAGE1_TOKENS:
                raise ValueError("Stage-1 authorization exceeded")
            state.last_training_loss = metric["raw_train_loss"]
            state.smoothed_training_loss = metric["raw_train_loss"] if state.smoothed_training_loss is None else 0.9 * state.smoothed_training_loss + 0.1 * metric["raw_train_loss"]
            metric.update(
                {
                    "optimizer_step": state.step,
                    "tokens_consumed": state.tokens_seen,
                    "sequence_index_start": state.data_position // 512 - 16,
                    "sequence_index_end_exclusive": state.data_position // 512,
                    "smoothed_train_loss": state.smoothed_training_loss,
                    "learning_rate": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "gpu": monitor.latest(),
                    "non_finite_event_count": non_finite_events,
                    "evaluation_seconds": 0.0,
                    "checkpoint_seconds": 0.0,
                }
            )
            if state.step in milestones:
                eval_started = time.perf_counter()
                validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                evaluation = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                elapsed_eval = time.perf_counter() - eval_started
                evaluation_seconds += elapsed_eval
                checkpoint_path = run_dir / "checkpoints" / checkpoint_name(state.step, state.tokens_seen)
                state.last_validation_loss = float(validation["loss"])
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = float(validation["loss"])
                    state.best_checkpoint = relative(checkpoint_path)
                checkpoint_started = time.perf_counter()
                metadata = save_checkpoint_atomic(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    data_hash=data_hash,
                    sequence_order_hash=order["deterministic_content_hash"],
                )
                elapsed_checkpoint = time.perf_counter() - checkpoint_started
                checkpoint_seconds += elapsed_checkpoint
                atomic_write_json(run_dir / "validations" / f"step_{state.step:06d}.json", validation)
                atomic_write_json(run_dir / "eval" / f"step_{state.step:06d}.json", evaluation)
                run_manifest["checkpoints"][str(state.step)] = relative(checkpoint_path)
                run_manifest["checkpoint_model_hashes"][str(state.step)] = metadata["model_files"]
                run_manifest["latest_checkpoint"] = relative(checkpoint_path)
                run_manifest["best_checkpoint"] = state.best_checkpoint
                run_manifest["best_validation_loss"] = state.best_validation_loss
                metric.update(
                    {
                        "validation_loss": validation["loss"],
                        "eval_loss": evaluation["loss"],
                        "validation_perplexity": validation["perplexity"],
                        "eval_perplexity": evaluation["perplexity"],
                        "evaluation_seconds": elapsed_eval,
                        "checkpoint_seconds": elapsed_checkpoint,
                        "checkpoint": relative(checkpoint_path),
                    }
                )
                atomic_write_json(run_dir / "run_manifest.json", run_manifest)
            append_jsonl(run_dir / "metrics.jsonl", metric)
            if state.step % 5 == 0 or state.step in milestones:
                atomic_write_json(progress_path, {"operation": "optimizer_step_complete", "optimizer_step": state.step, "tokens_consumed": state.tokens_seen, "next_sequence_index": state.data_position // 512})
            if state.step % 10 == 0 or state.step in milestones:
                print(f"step={state.step} tokens={state.tokens_seen} loss={metric['raw_train_loss']:.6f} tok_s={metric['active_tokens_per_second']:.1f}", flush=True)
    finally:
        monitor.close()
    segment_range = [segment_start_position, state.data_position]
    if run_manifest["segment_ranges"] and run_manifest["segment_ranges"][-1][1] != segment_range[0]:
        raise ValueError("segment sequence range repeated or skipped")
    if not run_manifest["segment_ranges"] and segment_range[0] != 0:
        raise ValueError("first segment did not start at token zero")
    run_manifest["segment_ranges"].append(segment_range)
    run_manifest["optimizer_steps"] = state.step
    run_manifest["consumed_tokens"] = state.tokens_seen
    run_manifest["next_sequence_index"] = state.data_position // 512
    run_manifest["status"] = "midpoint_complete" if target_step == 305 else "stage1_complete"
    run_manifest["wall_clock"][f"segment_to_{target_step}_completed_unix"] = time.time()
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    temperatures = [item["temperature_c"] for item in monitor.samples if item.get("available")]
    powers = [item["power_w"] for item in monitor.samples if item.get("available")]
    summary = {
        "schema_version": "darkmind-v2-phase4a-segment-v1",
        "result": "PASS",
        "fresh_process_pid": os.getpid(),
        "segment_start_step": segment_start_step,
        "segment_end_step": state.step,
        "segment_token_range": segment_range,
        "next_optimizer_step": state.step + 1 if state.step < STAGE1_STEPS else None,
        "next_sequence_index": state.data_position // 512,
        "rng_continuity": True,
        "scheduler_continuity": True,
        "data_position_continuity": True,
        "validation_reproducible": True,
        "no_repeated_or_skipped_sequence": True,
        "no_data_wrap": True,
        "elapsed_seconds": time.perf_counter() - segment_started,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_temperature_c_min_max": [min(temperatures), max(temperatures)] if temperatures else None,
        "gpu_power_w_min_max": [min(powers), max(powers)] if powers else None,
        "latest_checkpoint": run_manifest["latest_checkpoint"],
    }
    atomic_write_json(run_dir / "resume" / f"segment_to_step_{target_step:06d}.json", summary)
    atomic_write_json(progress_path, {"operation": "segment_complete_process_will_exit", "optimizer_step": state.step, "pid": os.getpid()})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "initialize"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    segment = commands.add_parser("segment")
    segment.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--target-step", type=int, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibration(args.config)
    elif args.command == "initialize":
        initialize(args.config)
    else:
        train_segment(args.config, args.checkpoint, args.target_step)


if __name__ == "__main__":
    main()
