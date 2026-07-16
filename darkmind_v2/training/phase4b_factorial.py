"""Run controlled Phase 4B Base V1 learning-rate and data-order experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import load_model_package, save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.phase3b_finalist_pilots import evaluate_loss
from darkmind_v2.training.phase4b_runtime import (
    DATA_ORDER_SEED,
    INPUT_ROOT,
    MODEL_INPUT,
    ORDER_ROOT,
    RUNTIME_ROOT,
    SEQUENCE_LENGTH,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    atomic_write_json,
    ensure_runtime_path,
    percentile,
    sha256_file,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_base_v1_stage1 import GpuMonitor, build_optimizer, build_scheduler
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.validate_phase4a_config import learning_rate_for_step
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
    ROOT,
)


INITIALIZATION_SEED = 20260712
TOKENS_PER_STEP = 8192
STAGE1_STEPS = 610
STAGE1_TOKENS = 4_997_120
MILESTONES = (0, 64, 128, 256, 384, 512, 610)
ARM_SPECS = {
    "arm_a_legacy_lr3e4": {"order": "legacy_order_v1", "peak_learning_rate": 0.0003},
    "arm_b_legacy_lr15e5": {"order": "legacy_order_v1", "peak_learning_rate": 0.00015},
    "arm_c_stratified_lr3e4": {"order": "deterministic_stratified_v1", "peak_learning_rate": 0.0003},
    "arm_d_stratified_lr15e5": {"order": "deterministic_stratified_v1", "peak_learning_rate": 0.00015},
}

# Optional 0.0002 is triggered only when a matched 0.00015 arm is stable, its
# 0.0003 counterpart is unstable, its best checkpoint is step 610, and both
# validation and eval improve monotonically over the final three evaluations.
MIDPOINT_LR_TRIGGER = {
    "lower_lr": 0.00015,
    "higher_lr": 0.0003,
    "candidate_lr": 0.0002,
    "requires_lower_stability": "stable",
    "requires_higher_stability": "unstable",
    "requires_best_step": 610,
    "requires_final_three_monotonic_improvement": True,
}


class OrderedTokenDataset:
    def __init__(self, tokenized_dir: Path, order_manifest: Path) -> None:
        self.base = TokenShardDataset(tokenized_dir, "train")
        payload = json.loads(order_manifest.read_text(encoding="utf-8"))
        core = {key: value for key, value in payload.items() if key != "deterministic_content_hash"}
        if canonical_json_hash(core) != payload["deterministic_content_hash"]:
            raise ValueError("sequence-order manifest hash mismatch")
        self.order = [int(value) for value in payload["indices"]]
        if len(self.order) != len(set(self.order)) or sorted(self.order) != list(range(len(self.order))):
            raise ValueError("sequence order contains duplicate or missing indices")
        self.order_hash = payload["deterministic_content_hash"]
        self.order_name = payload["name"]
        self.total_tokens = len(self.order) * SEQUENCE_LENGTH

    def source_indices(self, offset: int, count: int) -> list[int]:
        if offset % SEQUENCE_LENGTH or count % SEQUENCE_LENGTH:
            raise ValueError("ordered reads must align to complete sequences")
        start = offset // SEQUENCE_LENGTH
        end = start + count // SEQUENCE_LENGTH
        if start < 0 or end > len(self.order):
            raise ValueError("ordered read exceeds complete sequence order")
        return self.order[start:end]

    def read(self, offset: int, count: int) -> np.ndarray:
        indices = self.source_indices(offset, count)
        pieces = [self.base.read(index * SEQUENCE_LENGTH, SEQUENCE_LENGTH) for index in indices]
        return pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)

    def batch(
        self,
        *,
        offset: int,
        micro_batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if sequence_length != SEQUENCE_LENGTH:
            raise ValueError("Phase 4B sequence length changed")
        values = self.read(offset, micro_batch_size * sequence_length)
        return torch.from_numpy(values).to(device=device, dtype=torch.long).view(micro_batch_size, sequence_length)


def build_arm_config(arm_name: str, order_name: str, peak_learning_rate: float, run_dir: Path) -> dict[str, Any]:
    run_dir = ensure_runtime_path(run_dir)
    return {
        "schema_version": "darkmind-v2-phase4b-factorial-arm-v1",
        "arm_name": arm_name,
        "model_name": "darkmind-v2-base-v1",
        "model_config": str(MODEL_INPUT),
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_dir": str(TOKENIZER_INPUT),
        "corpus": {
            "tokenized_dir": str(TOKENIZED_INPUT),
            "corpus_hash": EXPECTED_CORPUS_HASH,
            "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        },
        "run_dir": str(run_dir),
        "sequence_order_manifest": str(ORDER_ROOT / f"{order_name}.json"),
        "initialization_seed": INITIALIZATION_SEED,
        "data": {
            "data_order_seed": DATA_ORDER_SEED,
            "sequence_order": order_name,
            "sequence_length": SEQUENCE_LENGTH,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_tokens_per_optimizer_step": TOKENS_PER_STEP,
            "no_replacement": True,
            "no_wrap": True,
        },
        "optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "gradient_clipping": 1.0,
        },
        "schedule": {
            "name": "warmup_cosine",
            "peak_learning_rate": peak_learning_rate,
            "minimum_learning_rate": 0.00003,
            "warmup_optimizer_steps": 100,
            "scheduler_horizon_optimizer_steps": 12_207,
            "scheduler_horizon_tokens": 99_999_744,
            "scheduler_restart": False,
        },
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "gradient_checkpointing": False,
        "training_compile": False,
        "fused_optimizer": False,
        "authorization": {
            "maximum_optimizer_steps": STAGE1_STEPS,
            "maximum_training_tokens": STAGE1_TOKENS,
            "phase_25m_authorized": False,
            "phase_100m_authorized": False,
        },
        "evaluation_steps": list(MILESTONES),
    }


def factorial_contract(configs: list[dict[str, Any]]) -> dict[str, Any]:
    ignored = {"arm_name", "run_dir", "sequence_order_manifest"}
    baseline = configs[0]
    differences: dict[str, list[str]] = {}
    for config in configs[1:]:
        changed = []
        for key in sorted(set(baseline) | set(config)):
            if key in ignored:
                continue
            if key == "data":
                left = {name: value for name, value in baseline[key].items() if name != "sequence_order"}
                right = {name: value for name, value in config[key].items() if name != "sequence_order"}
                if left != right:
                    changed.append("data")
                continue
            if key == "schedule":
                left = {name: value for name, value in baseline[key].items() if name != "peak_learning_rate"}
                right = {name: value for name, value in config[key].items() if name != "peak_learning_rate"}
                if left != right:
                    changed.append("schedule")
                continue
            if baseline.get(key) != config.get(key):
                changed.append(key)
        differences[config["arm_name"]] = changed
    if any(differences.values()):
        raise ValueError(f"factorial configs differ outside declared factors: {differences}")
    return {
        "result": "PASS",
        "allowed_factors": ["data.sequence_order", "schedule.peak_learning_rate", "arm_name", "run_dir"],
        "unexpected_differences": differences,
    }


def prepare_configs() -> dict[str, Any]:
    configs = []
    for arm_name, spec in ARM_SPECS.items():
        run_dir = RUNTIME_ROOT / "runs" / arm_name
        config = build_arm_config(arm_name, spec["order"], spec["peak_learning_rate"], run_dir)
        config_path = INPUT_ROOT / "configs" / f"{arm_name}.json"
        atomic_write_json(config_path, config)
        configs.append(config)
    contract = factorial_contract(configs)
    payload = {
        "schema_version": "darkmind-v2-phase4b-factorial-design-v1",
        "result": "PASS",
        "initialization_seed": INITIALIZATION_SEED,
        "stage1_steps": STAGE1_STEPS,
        "stage1_tokens": STAGE1_TOKENS,
        "milestones": list(MILESTONES),
        "arms": ARM_SPECS,
        "contract": contract,
        "optional_midpoint_lr_trigger": MIDPOINT_LR_TRIGGER,
    }
    atomic_write_json(INPUT_ROOT / "factorial_design.json", payload)
    return payload


def validate_arm_config(config: dict[str, Any]) -> None:
    ensure_runtime_path(Path(config["run_dir"]))
    ensure_runtime_path(Path(config["corpus"]["tokenized_dir"]))
    ensure_runtime_path(Path(config["tokenizer_dir"]))
    ensure_runtime_path(Path(config["sequence_order_manifest"]))
    if config["initialization_seed"] != INITIALIZATION_SEED or config["data"]["data_order_seed"] != DATA_ORDER_SEED:
        raise ValueError("Phase 4B seed changed")
    if config["authorization"] != {
        "maximum_optimizer_steps": STAGE1_STEPS,
        "maximum_training_tokens": STAGE1_TOKENS,
        "phase_25m_authorized": False,
        "phase_100m_authorized": False,
    }:
        raise ValueError("Phase 4B authorization contract changed")
    fixed = {
        "sequence_length": SEQUENCE_LENGTH,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_tokens_per_optimizer_step": TOKENS_PER_STEP,
        "no_replacement": True,
        "no_wrap": True,
    }
    for key, value in fixed.items():
        if config["data"].get(key) != value:
            raise ValueError(f"Phase 4B data contract changed: {key}")
    if config["optimizer"] != {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
    }:
        raise ValueError("Phase 4B optimizer contract changed")
    schedule = config["schedule"]
    if schedule["peak_learning_rate"] not in {0.00015, 0.0002, 0.0003}:
        raise ValueError("unapproved Phase 4B peak learning rate")
    if {key: value for key, value in schedule.items() if key != "peak_learning_rate"} != {
        "name": "warmup_cosine",
        "minimum_learning_rate": 0.00003,
        "warmup_optimizer_steps": 100,
        "scheduler_horizon_optimizer_steps": 12_207,
        "scheduler_horizon_tokens": 99_999_744,
        "scheduler_restart": False,
    }:
        raise ValueError("Phase 4B scheduler contract changed")
    if config["precision"] != "bf16" or config["attention_implementation"] != "sdpa":
        raise ValueError("Phase 4B precision/attention changed")
    if config["gradient_checkpointing"] or config["training_compile"] or config["fused_optimizer"]:
        raise ValueError("unapproved Phase 4B backend feature enabled")
    if config["evaluation_steps"] != list(MILESTONES):
        raise ValueError("Phase 4B evaluation schedule changed")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = ensure_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_model_checkpoint(
    checkpoint: Path,
    model: DarkMindV2ForCausalLM,
    *,
    step: int,
    tokens: int,
    order_hash: str,
    data_hash: str,
) -> dict[str, Any]:
    checkpoint = ensure_runtime_path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite immutable checkpoint: {checkpoint}")
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete checkpoint requires inspection: {temporary}")
    temporary.mkdir(parents=True)
    model_hashes = save_model_package(model, temporary / "model")
    metadata = {
        "schema_version": "darkmind-v2-phase4b-model-checkpoint-v1",
        "result": "PASS",
        "optimizer_step": step,
        "consumed_tokens": tokens,
        "model_sha256": model_hashes["model_sha256"],
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_hashes": EXPECTED_HASHES,
        "data_manifest_file_sha256": data_hash,
        "sequence_order_hash": order_hash,
        "resume_capable": False,
    }
    atomic_write_json(temporary / "checkpoint_metadata.json", metadata)
    _rename_with_retry(temporary, checkpoint)
    return metadata


def optimizer_step(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    dataset: OrderedTokenDataset,
    *,
    data_position: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    data_wait = 0.0
    transfer = 0.0
    source_indices: list[int] = []
    started = time.perf_counter()
    for micro_step in range(8):
        offset = data_position + micro_step * 2 * SEQUENCE_LENGTH
        read_started = time.perf_counter()
        values = dataset.read(offset, 2 * SEQUENCE_LENGTH)
        data_wait += time.perf_counter() - read_started
        source_indices.extend(dataset.source_indices(offset, 2 * SEQUENCE_LENGTH))
        transfer_started = time.perf_counter()
        batch = torch.from_numpy(values).to(device=device, dtype=torch.long).view(2, SEQUENCE_LENGTH)
        torch.cuda.synchronize(device)
        transfer += time.perf_counter() - transfer_started
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("non-finite Phase 4B training loss")
            (output.loss / 8).backward()
        losses.append(float(output.loss.detach()))
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite Phase 4B gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    return {
        "raw_train_loss": statistics.fmean(losses),
        "gradient_norm": float(gradient_norm.detach()),
        "optimizer_step_duration_seconds": duration,
        "data_loader_wait_seconds": data_wait,
        "host_to_device_seconds": transfer,
        "active_tokens_per_second": TOKENS_PER_STEP / duration,
        "source_sequence_indices": source_indices,
    }


def _checkpoint_name(step: int) -> str:
    return f"step_{step:06d}_tokens_{step * TOKENS_PER_STEP:09d}"


def _linear_slope(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator if denominator else 0.0


def _max_consecutive_worsening(values: list[float]) -> int:
    current = 0
    best = 0
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value > previous else 0
        best = max(best, current)
    return best


def rebound_percent(best: float, final: float) -> float:
    if best <= 0:
        raise ValueError("best loss must be positive")
    return (final - best) * 100.0 / best


def require_initialization_hash_identity(hashes: dict[str, str]) -> str:
    if not hashes or len(set(hashes.values())) != 1:
        raise ValueError(f"factorial initialization hashes differ: {hashes}")
    return next(iter(hashes.values()))


def training_stability(summary: dict[str, Any], severe_generation_regression: bool = False) -> str:
    val_improvement = summary["validation_improvement_percent"]
    eval_improvement = summary["eval_improvement_percent"]
    val_rebound = summary["validation_rebound_percent"]
    eval_rebound = summary["eval_rebound_percent"]
    unstable = (
        not summary["integrity_pass"]
        or val_improvement < 5.0
        or eval_improvement < 5.0
        or val_rebound > 5.0
        or eval_rebound > 5.0
        or summary["sustained_divergence"]
        or severe_generation_regression
    )
    if unstable:
        return "unstable"
    if (
        val_improvement >= 15.0
        and eval_improvement >= 15.0
        and val_rebound <= 2.0
        and eval_rebound <= 2.0
    ):
        return "stable"
    return "partial stability"


def run_arm(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_arm_config(config)
    run_dir = ensure_runtime_path(Path(config["run_dir"]))
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4B arm: {run_dir}")
    run_dir.mkdir(parents=True)
    validation = json.loads((INPUT_ROOT / "validation_pass2.json").read_text(encoding="utf-8"))
    if validation.get("result") != "PASS" or validation.get("cross_pass_asset_identity") is not True:
        raise ValueError("two-pass relocated input validation is incomplete")
    verify_frozen_tokenizer(TOKENIZER_INPUT)
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Base V1 config changed")
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, Path(config["sequence_order_manifest"]))
    if order_data.total_tokens < STAGE1_TOKENS:
        raise ValueError("sequence order cannot satisfy the Stage-1 budget")
    data_hash = tokenized_manifest_hash(TOKENIZED_INPUT)
    atomic_write_json(run_dir / "resolved_config.json", config)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "darkmind-v2-phase4b-arm-run-v1",
            "status": "initializing",
            "arm_name": config["arm_name"],
            "runtime_root": str(RUNTIME_ROOT),
            "mutable_runtime_outside_onedrive": True,
            "sequence_order": order_data.order_name,
            "sequence_order_hash": order_data.order_hash,
            "peak_learning_rate": config["schedule"]["peak_learning_rate"],
            "initialization_seed": INITIALIZATION_SEED,
            "source_git_commit": os.popen("git rev-parse HEAD").read().strip(),
        },
    )
    set_deterministic_seed(INITIALIZATION_SEED)
    device = torch.device("cuda")
    model_config = DarkMindV2Config.from_json_file(MODEL_INPUT)
    validate_training_environment(model_config, micro_batch_size=2, precision="bf16")
    model = DarkMindV2ForCausalLM(model_config).to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    initial_train = evaluate_loss(model, order_data, sequence_length=512, micro_batch_size=2, device=device, maximum_tokens=TOKENS_PER_STEP)
    initial_validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
    initial_eval = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
    initial_checkpoint = run_dir / "checkpoints" / _checkpoint_name(0)
    initial_metadata = save_model_checkpoint(
        initial_checkpoint,
        model,
        step=0,
        tokens=0,
        order_hash=order_data.order_hash,
        data_hash=data_hash,
    )
    initialization_hash = initial_metadata["model_sha256"]
    evaluations: dict[str, Any] = {
        "0": {"train": initial_train, "validation": initial_validation, "eval": initial_eval}
    }
    checkpoints = {"0": str(initial_checkpoint)}
    checkpoint_hashes = {"0": initialization_hash}
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    smoothed_loss = None
    metrics: list[dict[str, Any]] = []
    data_position = 0
    try:
        for step in range(1, STAGE1_STEPS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            metric = optimizer_step(
                model,
                optimizer,
                order_data,
                data_position=data_position,
                config=config,
                device=device,
            )
            scheduler.step()
            data_position += TOKENS_PER_STEP
            smoothed_loss = metric["raw_train_loss"] if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * metric["raw_train_loss"]
            metric.update(
                {
                    "optimizer_step": step,
                    "tokens_consumed": data_position,
                    "ordered_sequence_position_start": step * 16 - 16,
                    "ordered_sequence_position_end_exclusive": step * 16,
                    "smoothed_train_loss": smoothed_loss,
                    "learning_rate": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "gpu": monitor.latest(),
                    "non_finite_event_count": 0,
                }
            )
            if step in MILESTONES:
                validation_result = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                eval_result = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                checkpoint = run_dir / "checkpoints" / _checkpoint_name(step)
                metadata = save_model_checkpoint(
                    checkpoint,
                    model,
                    step=step,
                    tokens=data_position,
                    order_hash=order_data.order_hash,
                    data_hash=data_hash,
                )
                checkpoints[str(step)] = str(checkpoint)
                checkpoint_hashes[str(step)] = metadata["model_sha256"]
                evaluations[str(step)] = {
                    "train": {"loss": metric["raw_train_loss"], "smoothed_loss": smoothed_loss},
                    "validation": validation_result,
                    "eval": eval_result,
                }
                metric.update(
                    {
                        "validation_loss": validation_result["loss"],
                        "eval_loss": eval_result["loss"],
                        "validation_perplexity": validation_result["perplexity"],
                        "eval_perplexity": eval_result["perplexity"],
                        "checkpoint": str(checkpoint),
                    }
                )
            _append_jsonl(run_dir / "metrics.jsonl", metric)
            metrics.append(metric)
            if step % 5 == 0 or step in MILESTONES:
                atomic_write_json(
                    run_dir / "progress.json",
                    {
                        "status": "training",
                        "optimizer_step": step,
                        "tokens_consumed": data_position,
                        "next_ordered_sequence_position": data_position // SEQUENCE_LENGTH,
                    },
                )
            if step % 10 == 0 or step in MILESTONES:
                print(
                    f"arm={config['arm_name']} step={step} loss={metric['raw_train_loss']:.6f} "
                    f"lr={learning_rate:.8f} tok_s={metric['active_tokens_per_second']:.1f}",
                    flush=True,
                )
    finally:
        monitor.close()
    if data_position != STAGE1_TOKENS or len(metrics) != STAGE1_STEPS:
        raise ValueError("Phase 4B arm did not stop at exact Stage-1 authorization")
    validation_losses = [evaluations[str(step)]["validation"]["loss"] for step in MILESTONES]
    eval_losses = [evaluations[str(step)]["eval"]["loss"] for step in MILESTONES]
    best_validation = min(validation_losses)
    best_eval = min(eval_losses)
    best_step = min(MILESTONES, key=lambda step: (evaluations[str(step)]["validation"]["loss"] + evaluations[str(step)]["eval"]["loss"]) / 2)
    initial_val = validation_losses[0]
    initial_eval_loss = eval_losses[0]
    final_val = validation_losses[-1]
    final_eval = eval_losses[-1]
    gradients = [item["gradient_norm"] for item in metrics]
    summary = {
        "schema_version": "darkmind-v2-phase4b-arm-summary-v1",
        "result": "PASS",
        "integrity_pass": True,
        "arm_name": config["arm_name"],
        "sequence_order": order_data.order_name,
        "sequence_order_hash": order_data.order_hash,
        "peak_learning_rate": config["schedule"]["peak_learning_rate"],
        "initialization_hash": initialization_hash,
        "optimizer_steps": STAGE1_STEPS,
        "training_tokens": STAGE1_TOKENS,
        "initial_validation_loss": initial_val,
        "initial_eval_loss": initial_eval_loss,
        "best_validation_loss": best_validation,
        "best_eval_loss": best_eval,
        "best_combined_step": best_step,
        "final_validation_loss": final_val,
        "final_eval_loss": final_eval,
        "validation_improvement_percent": (initial_val - final_val) * 100.0 / initial_val,
        "eval_improvement_percent": (initial_eval_loss - final_eval) * 100.0 / initial_eval_loss,
        "validation_best_improvement_percent": (initial_val - best_validation) * 100.0 / initial_val,
        "eval_best_improvement_percent": (initial_eval_loss - best_eval) * 100.0 / initial_eval_loss,
        "validation_rebound_percent": rebound_percent(best_validation, final_val),
        "eval_rebound_percent": rebound_percent(best_eval, final_eval),
        "post_warmup_train_loss_slope_per_step": _linear_slope(
            [(item["optimizer_step"], item["raw_train_loss"]) for item in metrics if item["optimizer_step"] > 100]
        ),
        "consecutive_worsening_validation_evaluations": _max_consecutive_worsening(validation_losses),
        "consecutive_worsening_eval_evaluations": _max_consecutive_worsening(eval_losses),
        "gradient_norm_p50": percentile(gradients, 0.50),
        "gradient_norm_p95": percentile(gradients, 0.95),
        "gradient_norm_max": max(gradients),
        "non_finite_events": 0,
        "sustained_divergence": _max_consecutive_worsening(validation_losses) >= 3 and _max_consecutive_worsening(eval_losses) >= 3,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "checkpoint_hashes": checkpoint_hashes,
        "active_tokens_per_second": STAGE1_TOKENS / sum(item["optimizer_step_duration_seconds"] for item in metrics),
        "wall_tokens_per_second": STAGE1_TOKENS / (time.perf_counter() - wall_started),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "temperature_c_min_max": [
            min(item["temperature_c"] for item in monitor.samples if item.get("available")),
            max(item["temperature_c"] for item in monitor.samples if item.get("available")),
        ],
        "power_w_min_max": [
            min(item["power_w"] for item in monitor.samples if item.get("available")),
            max(item["power_w"] for item in monitor.samples if item.get("available")),
        ],
        "runtime_stable": True,
    }
    summary["preliminary_stability"] = training_stability(summary)
    atomic_write_json(run_dir / "training_summary.json", summary)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "darkmind-v2-phase4b-arm-run-v1",
            "status": "training_complete",
            "arm_name": config["arm_name"],
            "initialization_hash": initialization_hash,
            "optimizer_steps": STAGE1_STEPS,
            "training_tokens": STAGE1_TOKENS,
            "checkpoints": checkpoints,
            "checkpoint_hashes": checkpoint_hashes,
            "runtime_stable": True,
            "mutable_runtime_outside_onedrive": True,
        },
    )
    atomic_write_json(run_dir / "progress.json", {"status": "training_complete", "optimizer_step": 610, "tokens_consumed": STAGE1_TOKENS})
    return summary


def verify_initialization_identity() -> dict[str, Any]:
    summaries = {
        name: json.loads((RUNTIME_ROOT / "runs" / name / "training_summary.json").read_text(encoding="utf-8"))
        for name in ARM_SPECS
    }
    hashes = {name: value["initialization_hash"] for name, value in summaries.items()}
    initialization_hash = require_initialization_hash_identity(hashes)
    return {"result": "PASS", "initialization_hash": initialization_hash, "arm_hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run-arm")
    run_parser.add_argument("--config", type=Path, required=True)
    subparsers.add_parser("verify-initialization")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_configs()
    elif args.command == "run-arm":
        payload = run_arm(args.config)
    else:
        payload = verify_initialization_identity()
    printable = {key: value for key, value in payload.items() if key not in {"evaluations", "checkpoints", "checkpoint_hashes", "arms"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
