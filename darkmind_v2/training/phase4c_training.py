"""Run controlled Phase 4C Base V1 optimizer and scheduler experiments."""

from __future__ import annotations

import argparse
import hashlib
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
from darkmind_v2.modeling.model_io import save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.phase3b_finalist_pilots import evaluate_loss
from darkmind_v2.training.phase4b_factorial import OrderedTokenDataset, percentile, rebound_percent
from darkmind_v2.training.phase4b_runtime import load_document_spans, sequence_labels
from darkmind_v2.training.phase4c_diagnostics import (
    DIAGNOSTIC_ROOT,
    INITIALIZATION_SEED,
    MINIMUM_LR,
    MODEL_INPUT,
    ORDER_INPUT,
    PHASE4B_INPUT_ROOT,
    RUNTIME_ROOT,
    STAGE1_STEPS,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    _activation_run,
    atomic_write_json,
    build_optimizer_groups,
    build_scheduler,
    ensure_runtime_path,
    learning_rate_for_policy,
    sha256_file,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_base_v1_stage1 import GpuMonitor
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
)


STAGE1_TOKENS = 4_997_120
MILESTONES = (0, 64, 128, 192, 256, 384, 512, 610)
RUN_ROOT = RUNTIME_ROOT / "runs"
CONFIG_ROOT = RUNTIME_ROOT / "inputs" / "arm_configs"
PROBE_MANIFEST = RUNTIME_ROOT / "inputs" / "fixed_probe_manifest.json"
ARM_SPECS = {
    "arm1_global_lr1e4_current_groups": {
        "peak_learning_rate": 0.0001,
        "optimizer_grouping": "current_decay_all",
        "scheduler": "global",
        "initialization_policy": "base_v1_standard_v1",
    },
    "arm2_global_lr75e6_current_groups": {
        "peak_learning_rate": 0.000075,
        "optimizer_grouping": "current_decay_all",
        "scheduler": "global",
        "initialization_policy": "base_v1_standard_v1",
    },
    "arm3_global_lr1e4_corrected_groups": {
        "peak_learning_rate": 0.0001,
        "optimizer_grouping": "corrected_adamw_v1",
        "scheduler": "global",
        "initialization_policy": "base_v1_standard_v1",
    },
}
FOLLOWUP_ARMS = (
    "arm4_staged_decay_corrected_groups",
    "arm5_depth_scaled_init_staged",
)
DIAGNOSTIC_THRESHOLDS = {
    "matrix_parameter_update_to_weight_ratio": 0.01,
    "residual_ratio_growth_multiple_from_step0": 4.0,
    "logit_standard_deviation": 20.0,
    "gradient_norm": 1000.0,
}


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


def schedule_payload(kind: str, peak: float) -> dict[str, Any]:
    if kind == "global":
        return {
            "name": "warmup_cosine_global",
            "peak_learning_rate": peak,
            "minimum_learning_rate": MINIMUM_LR,
            "warmup_optimizer_steps": 100,
            "scheduler_horizon_optimizer_steps": TOTAL_STEPS,
            "scheduler_horizon_tokens": 99_999_744,
            "scheduler_restart": False,
        }
    if kind == "staged":
        return {
            "name": "warmup_cosine_staged_continuation",
            "peak_learning_rate": peak,
            "stage1_end_learning_rate": peak * 0.5,
            "stage1_end_optimizer_step": 610,
            "minimum_learning_rate": MINIMUM_LR,
            "warmup_optimizer_steps": 100,
            "scheduler_horizon_optimizer_steps": TOTAL_STEPS,
            "scheduler_horizon_tokens": 99_999_744,
            "scheduler_restart": False,
        }
    raise ValueError(f"unsupported schedule kind: {kind}")


def arm_config(arm_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = ensure_runtime_path(RUN_ROOT / arm_name)
    payload = {
        "schema_version": "darkmind-v2-phase4c-diagnostic-arm-v1",
        "arm_name": arm_name,
        "model_name": "darkmind-v2-base-v1",
        "model_config": str(MODEL_INPUT),
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_dir": str(TOKENIZER_INPUT),
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "corpus": {
            "tokenized_dir": str(TOKENIZED_INPUT),
            "corpus_hash": EXPECTED_CORPUS_HASH,
            "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        },
        "run_dir": str(run_dir),
        "sequence_order_manifest": str(ORDER_INPUT),
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_policy": spec["initialization_policy"],
        "data": {
            "data_order_seed": INITIALIZATION_SEED,
            "sequence_order": "deterministic_stratified_v1",
            "sequence_length": 512,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_tokens_per_optimizer_step": TOKENS_PER_STEP,
            "no_replacement": True,
            "no_wrap": True,
        },
        "optimizer": {
            "name": "AdamW",
            "grouping": spec["optimizer_grouping"],
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "gradient_clipping": 1.0,
        },
        "schedule": schedule_payload(spec["scheduler"], float(spec["peak_learning_rate"])),
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "gradient_checkpointing": False,
        "training_compile": False,
        "fused_optimizer": False,
        "evaluation_steps": list(MILESTONES),
        "diagnostic_thresholds": dict(DIAGNOSTIC_THRESHOLDS),
        "authorization": {
            "maximum_optimizer_steps": 610,
            "maximum_training_tokens": STAGE1_TOKENS,
            "phase_25m_authorized": False,
            "phase_100m_authorized": False,
        },
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    return payload


def validate_arm_config(config: dict[str, Any]) -> None:
    core = {key: value for key, value in config.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != config.get("deterministic_content_hash"):
        raise ValueError("Phase 4C arm config hash mismatch")
    if config.get("schema_version") != "darkmind-v2-phase4c-diagnostic-arm-v1":
        raise ValueError("unsupported Phase 4C arm config")
    if config["model_config_sha256"] != EXPECTED_CONFIG_SHA256 or config["architecture_hash"] != EXPECTED_ARCHITECTURE_HASH:
        raise ValueError("frozen Base V1 identity changed")
    if config["tokenizer_hashes"] != EXPECTED_HASHES:
        raise ValueError("frozen tokenizer identity changed")
    if config["corpus"]["corpus_hash"] != EXPECTED_CORPUS_HASH or config["corpus"]["tokenized_manifest_hash"] != EXPECTED_TOKENIZED_HASH:
        raise ValueError("Corpus V3 identity changed")
    if config["initialization_seed"] != INITIALIZATION_SEED:
        raise ValueError("initialization seed changed")
    if config["initialization_policy"] not in {"base_v1_standard_v1", "base_v1_depth_scaled_residual_v2"}:
        raise ValueError("unapproved initialization policy")
    if config["data"] != {
        "data_order_seed": INITIALIZATION_SEED,
        "sequence_order": "deterministic_stratified_v1",
        "sequence_length": 512,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_tokens_per_optimizer_step": 8192,
        "no_replacement": True,
        "no_wrap": True,
    }:
        raise ValueError("Phase 4C data contract changed")
    optimizer = config["optimizer"]
    if {key: value for key, value in optimizer.items() if key != "grouping"} != {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
    }:
        raise ValueError("Phase 4C optimizer contract changed")
    if optimizer["grouping"] not in {"current_decay_all", "corrected_adamw_v1"}:
        raise ValueError("unapproved optimizer grouping")
    schedule = config["schedule"]
    if schedule["peak_learning_rate"] not in {0.000075, 0.0001}:
        raise ValueError("unapproved Phase 4C peak LR")
    if schedule["scheduler_horizon_optimizer_steps"] != TOTAL_STEPS or schedule["scheduler_horizon_tokens"] != 99_999_744:
        raise ValueError("100M scheduler horizon changed")
    if schedule["warmup_optimizer_steps"] != 100 or schedule["minimum_learning_rate"] != MINIMUM_LR or schedule["scheduler_restart"]:
        raise ValueError("Phase 4C scheduler bounds changed")
    if schedule["name"] == "warmup_cosine_staged_continuation":
        if schedule["stage1_end_optimizer_step"] != 610 or schedule["stage1_end_learning_rate"] != schedule["peak_learning_rate"] * 0.5:
            raise ValueError("staged continuation contract changed")
    elif schedule["name"] != "warmup_cosine_global":
        raise ValueError("unapproved scheduler")
    if config["authorization"] != {
        "maximum_optimizer_steps": 610,
        "maximum_training_tokens": 4_997_120,
        "phase_25m_authorized": False,
        "phase_100m_authorized": False,
    }:
        raise ValueError("Phase 4C arm exceeds the 5M authorization")
    if config["precision"] != "bf16" or config["attention_implementation"] != "sdpa":
        raise ValueError("Phase 4C numerical profile changed")
    if config["gradient_checkpointing"] or config["training_compile"] or config["fused_optimizer"]:
        raise ValueError("unapproved backend optimization enabled")
    if config["evaluation_steps"] != list(MILESTONES):
        raise ValueError("Phase 4C evaluation schedule changed")


def prepare_core_configs() -> dict[str, Any]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    configs = {}
    for name, spec in ARM_SPECS.items():
        config = arm_config(name, spec)
        validate_arm_config(config)
        atomic_write_json(CONFIG_ROOT / f"{name}.json", config)
        configs[name] = config
    payload = {
        "schema_version": "darkmind-v2-phase4c-core-arm-design-v1",
        "result": "PASS",
        "arms": {name: str(CONFIG_ROOT / f"{name}.json") for name in configs},
        "controlled_differences": ["optimizer.grouping", "schedule.peak_learning_rate"],
        "maximum_optimizer_steps": 610,
        "maximum_training_tokens": STAGE1_TOKENS,
        "phase_25m_authorized": False,
    }
    atomic_write_json(RUNTIME_ROOT / "inputs" / "core_arm_design.json", payload)
    return payload


def prepare_probe_manifest() -> dict[str, Any]:
    if PROBE_MANIFEST.is_file():
        return json.loads(PROBE_MANIFEST.read_text(encoding="utf-8"))
    tokenized = json.loads((TOKENIZED_INPUT / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    complete_sequences = int(tokenized["statistics"]["split_tokens"]["train"]) // 512
    labels = sequence_labels(load_document_spans()["train"], complete_sequences)
    order = json.loads(ORDER_INPUT.read_text(encoding="utf-8"))["indices"]

    def first_matching(language: str | None, category: str | None) -> int:
        for index, (current_language, current_category, _) in enumerate(labels):
            if (language is None or current_language == language) and (category is None or current_category == category):
                return index
        raise ValueError(f"no probe sequence for {language}/{category}")

    probes = {
        "turkish_prose": {"split": "train", "sequence_indices": [first_matching("tr", "prose")]},
        "english_prose": {"split": "train", "sequence_indices": [first_matching("en", "prose")]},
        "technical": {"split": "train", "sequence_indices": [first_matching(None, "technical")]},
        "training_distribution": {"split": "train", "sequence_indices": [int(value) for value in order[:16]]},
        "validation": {"split": "validation", "sequence_indices": list(range(16))},
        "eval": {"split": "eval", "sequence_indices": list(range(16))},
    }
    datasets = {split: TokenShardDataset(TOKENIZED_INPUT, split) for split in ("train", "validation", "eval")}
    for probe in probes.values():
        digest = hashlib.sha256()
        for index in probe["sequence_indices"]:
            digest.update(datasets[probe["split"]].read(index * 512, 512).tobytes())
        probe["token_sha256"] = digest.hexdigest()
        probe["tokens"] = len(probe["sequence_indices"]) * 512
    payload = {
        "schema_version": "darkmind-v2-phase4c-fixed-probes-v1",
        "result": "PASS",
        "sequence_length": 512,
        "probes": probes,
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    atomic_write_json(PROBE_MANIFEST, payload)
    return payload


@torch.no_grad()
def evaluate_probe(model: DarkMindV2ForCausalLM, probe: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dataset = TokenShardDataset(TOKENIZED_INPUT, probe["split"])
    model.eval()
    losses = []
    for start in range(0, len(probe["sequence_indices"]), 2):
        indices = probe["sequence_indices"][start : start + 2]
        values = np.concatenate([dataset.read(int(index) * 512, 512) for index in indices])
        batch = torch.from_numpy(values.astype(np.int64, copy=False)).to(device=device, dtype=torch.long).view(len(indices), 512)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch, labels=batch).loss
        losses.extend([float(loss)] * len(indices))
    model.train()
    value = statistics.fmean(losses)
    return {"loss": value, "perplexity": math.exp(min(value, 80.0)), "sequences": len(probe["sequence_indices"]), "token_sha256": probe["token_sha256"]}


def apply_initialization_policy(model: DarkMindV2ForCausalLM, policy: str) -> dict[str, Any]:
    if policy == "base_v1_standard_v1":
        return {"name": policy, "residual_projection_scale": 1.0, "modified_tensors": 0}
    if policy != "base_v1_depth_scaled_residual_v2":
        raise ValueError("unsupported initialization policy")
    scale = 1.0 / math.sqrt(2.0 * len(model.blocks))
    modified = 0
    with torch.no_grad():
        for block in model.blocks:
            block.attn.proj.weight.mul_(scale)
            block.mlp.proj.weight.mul_(scale)
            modified += 2
    return {"name": policy, "residual_projection_scale": scale, "modified_tensors": modified}


def build_optimizer(model: DarkMindV2ForCausalLM, config: dict[str, Any]) -> torch.optim.AdamW:
    corrected = config["optimizer"]["grouping"] == "corrected_adamw_v1"
    groups = build_optimizer_groups(model, corrected=corrected, weight_decay=config["optimizer"]["weight_decay"])
    return torch.optim.AdamW(
        groups,
        lr=float(config["schedule"]["peak_learning_rate"]),
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
        fused=False,
    )


def save_model_checkpoint(
    checkpoint: Path,
    model: DarkMindV2ForCausalLM,
    *,
    config: dict[str, Any],
    step: int,
    data_hash: str,
    order_hash: str,
) -> dict[str, Any]:
    checkpoint = ensure_runtime_path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4C checkpoint: {checkpoint}")
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale Phase 4C checkpoint requires inspection: {temporary}")
    temporary.mkdir(parents=True)
    model_hashes = save_model_package(model, temporary / "model")
    metadata = {
        "schema_version": "darkmind-v2-phase4c-model-checkpoint-v1",
        "result": "PASS",
        "optimizer_step": step,
        "consumed_tokens": step * TOKENS_PER_STEP,
        "model_sha256": model_hashes["model_sha256"],
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_hashes": EXPECTED_HASHES,
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "data_manifest_file_sha256": data_hash,
        "sequence_order_hash": order_hash,
        "initialization_policy": config["initialization_policy"],
        "resume_capable": False,
    }
    atomic_write_json(temporary / "checkpoint_metadata.json", metadata)
    _rename_with_retry(temporary, checkpoint)
    return metadata


def _parameter_diagnostics(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    before: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    records = []
    for name, parameter in model.named_parameters():
        value = parameter.detach().float()
        gradient = parameter.grad.detach().float() if parameter.grad is not None else torch.zeros((), device=parameter.device)
        delta = (parameter.detach() - before[name]).float()
        state = optimizer.state.get(parameter, {})
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        parameter_rms = float(value.square().mean().sqrt())
        update_rms = float(delta.square().mean().sqrt())
        records.append(
            {
                "parameter": name,
                "module": module_name,
                "gradient_rms": float(gradient.square().mean().sqrt()),
                "gradient_norm": float(gradient.norm()),
                "parameter_rms": parameter_rms,
                "update_rms": update_rms,
                "update_to_weight_ratio": update_rms / max(parameter_rms, 1e-12),
                "adam_first_moment_rms": float(first.detach().float().square().mean().sqrt()) if isinstance(first, torch.Tensor) else 0.0,
                "adam_second_moment_rms": float(second.detach().float().square().mean().sqrt()) if isinstance(second, torch.Tensor) else 0.0,
            }
        )
    return records


def optimizer_step(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    dataset: OrderedTokenDataset,
    *,
    data_position: int,
    diagnostic: bool,
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    source_indices = []
    started = time.perf_counter()
    for micro_step in range(8):
        offset = data_position + micro_step * 1024
        values = dataset.read(offset, 1024)
        source_indices.extend(dataset.source_indices(offset, 1024))
        batch = torch.from_numpy(values.astype(np.int64, copy=False)).to(device=device, dtype=torch.long).view(2, 512)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("non-finite Phase 4C training loss")
        (output.loss / 8).backward()
        losses.append(float(output.loss.detach()))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()} if diagnostic else {}
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite Phase 4C gradient norm")
    clipped = float(gradient_norm) > 1.0
    optimizer.step()
    torch.cuda.synchronize(device)
    parameter_diagnostics = _parameter_diagnostics(model, optimizer, before) if diagnostic else None
    return {
        "raw_train_loss": statistics.fmean(losses),
        "gradient_norm": float(gradient_norm),
        "clipped": clipped,
        "optimizer_step_duration_seconds": time.perf_counter() - started,
        "active_tokens_per_second": TOKENS_PER_STEP / (time.perf_counter() - started),
        "source_sequence_indices": source_indices,
        "parameter_diagnostics": parameter_diagnostics,
    }


def _checkpoint_name(step: int) -> str:
    return f"step_{step:06d}_tokens_{step * TOKENS_PER_STEP:09d}"


def _max_consecutive_worsening(values: list[float]) -> int:
    current = 0
    maximum = 0
    for left, right in zip(values, values[1:]):
        current = current + 1 if right > left else 0
        maximum = max(maximum, current)
    return maximum


def classify_stability(summary: dict[str, Any]) -> str:
    stable = (
        summary["integrity_pass"]
        and summary["validation_improvement_percent"] >= 15.0
        and summary["eval_improvement_percent"] >= 15.0
        and summary["validation_rebound_percent"] <= 2.0
        and summary["eval_rebound_percent"] <= 2.0
        and not summary["last_three_validation_sustained_worsening"]
        and not summary["last_three_eval_sustained_worsening"]
        and not summary["abnormal_diagnostic_growth"]
        and summary["runtime_stable"]
    )
    if stable:
        return "stable"
    if (
        summary["integrity_pass"]
        and summary["validation_improvement_percent"] >= 5.0
        and summary["eval_improvement_percent"] >= 5.0
        and summary["validation_rebound_percent"] <= 5.0
        and summary["eval_rebound_percent"] <= 5.0
    ):
        return "partial stability"
    return "unstable"


def normalize_diagnostic_summary(
    summary: dict[str, Any],
    diagnostic_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reclassify an arm from its immutable snapshots using normalized thresholds."""
    normalized = dict(summary)
    parameter_records = [
        record
        for snapshot in diagnostic_snapshots.values()
        for record in snapshot["parameter_diagnostics"]
    ]
    matrix_records = [record for record in parameter_records if record["parameter"].endswith(".weight")]
    maximum_matrix_ratio = max((record["update_to_weight_ratio"] for record in matrix_records), default=0.0)
    residual_ratios = {
        step: snapshot["activation"]["final_to_early_residual_ratio"]
        for step, snapshot in diagnostic_snapshots.items()
    }
    initial_residual_ratio = residual_ratios["0"]
    maximum_residual_ratio = max(residual_ratios.values())
    maximum_residual_growth = maximum_residual_ratio / initial_residual_ratio
    maximum_logit_std = max(snapshot["activation"]["logits"]["std"] for snapshot in diagnostic_snapshots.values())
    maximum_gradient = float(summary["gradient_norm_max"])
    normalized.update(
        {
            "maximum_matrix_parameter_update_to_weight_ratio": maximum_matrix_ratio,
            "maximum_residual_final_to_early_ratio": maximum_residual_ratio,
            "initial_residual_final_to_early_ratio": initial_residual_ratio,
            "maximum_residual_ratio_growth_multiple_from_step0": maximum_residual_growth,
            "maximum_logit_standard_deviation": maximum_logit_std,
            "abnormal_diagnostic_growth": (
                maximum_matrix_ratio > DIAGNOSTIC_THRESHOLDS["matrix_parameter_update_to_weight_ratio"]
                or maximum_residual_growth > DIAGNOSTIC_THRESHOLDS["residual_ratio_growth_multiple_from_step0"]
                or maximum_logit_std > DIAGNOSTIC_THRESHOLDS["logit_standard_deviation"]
                or maximum_gradient > DIAGNOSTIC_THRESHOLDS["gradient_norm"]
            ),
        }
    )
    normalized["stability"] = classify_stability(normalized)
    return normalized


def validate_training_diagnostic_schema(snapshot: dict[str, Any]) -> None:
    required = {
        "optimizer_step",
        "parameter_diagnostics",
        "clipped_step_fraction",
        "embedding_norm",
        "residual_projection_norms",
        "layernorm_weight_statistics",
        "activation",
        "probe_losses",
    }
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"training diagnostic schema is missing: {sorted(missing)}")
    activation = snapshot["activation"]
    if not activation.get("layers") or "logits" not in activation:
        raise ValueError("activation diagnostic schema is incomplete")


def _milestone_diagnostics(
    model: DarkMindV2ForCausalLM,
    *,
    step: int,
    parameter_diagnostics: list[dict[str, Any]],
    clipped_fraction: float,
    probe_losses: dict[str, Any],
    activation_tokens: torch.Tensor,
) -> dict[str, Any]:
    embedding_norm = model.token_embedding.weight.detach().float().norm(dim=1)
    residual_projection_norms = {
        f"blocks.{index}.attn.proj.weight": float(block.attn.proj.weight.detach().float().norm())
        for index, block in enumerate(model.blocks)
    }
    residual_projection_norms.update(
        {
            f"blocks.{index}.mlp.proj.weight": float(block.mlp.proj.weight.detach().float().norm())
            for index, block in enumerate(model.blocks)
        }
    )
    layernorm_values = torch.cat(
        [parameter.detach().float().flatten() for name, parameter in model.named_parameters() if ".ln_" in name or name.startswith("final_norm")]
    )
    activation = _activation_run(model, activation_tokens, f"training_step_{step}")
    snapshot = {
        "schema_version": "darkmind-v2-phase4c-training-diagnostic-v1",
        "optimizer_step": step,
        "parameter_diagnostics": parameter_diagnostics,
        "clipped_step_fraction": clipped_fraction,
        "embedding_norm": {
            "mean": float(embedding_norm.mean()),
            "p95": float(torch.quantile(embedding_norm, 0.95)),
            "maximum": float(embedding_norm.max()),
        },
        "residual_projection_norms": residual_projection_norms,
        "layernorm_weight_statistics": {
            "mean": float(layernorm_values.mean()),
            "std": float(layernorm_values.std(unbiased=False)),
            "minimum": float(layernorm_values.min()),
            "maximum": float(layernorm_values.max()),
        },
        "activation": activation,
        "probe_losses": probe_losses,
    }
    validate_training_diagnostic_schema(snapshot)
    return snapshot


def run_arm(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_arm_config(config)
    run_dir = ensure_runtime_path(Path(config["run_dir"]))
    if (run_dir / "training_summary.json").is_file():
        existing = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        if existing["deterministic_content_hash"] != config["deterministic_content_hash"]:
            raise ValueError("completed Phase 4C arm config mismatch")
        return json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    if run_dir.exists():
        raise FileExistsError(f"incomplete Phase 4C arm requires inspection: {run_dir}")
    run_dir.mkdir(parents=True)
    validation = json.loads((RUNTIME_ROOT / "inputs" / "shared_input_validation.json").read_text(encoding="utf-8"))
    if validation.get("result") != "PASS" or validation.get("cross_pass_asset_identity") is not True:
        raise ValueError("Phase 4C shared input validation is incomplete")
    verify_frozen_tokenizer(TOKENIZER_INPUT)
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Base V1 config hash changed")
    probe_manifest = prepare_probe_manifest()
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.total_tokens < STAGE1_TOKENS:
        raise ValueError("stratified order cannot satisfy the Stage-1 budget")
    data_hash = tokenized_manifest_hash(TOKENIZED_INPUT)
    atomic_write_json(run_dir / "resolved_config.json", config)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "darkmind-v2-phase4c-arm-run-v1",
            "status": "initializing",
            "arm_name": config["arm_name"],
            "runtime_root": str(RUNTIME_ROOT),
            "mutable_runtime_outside_onedrive": True,
            "initialization_seed": INITIALIZATION_SEED,
            "initialization_policy": config["initialization_policy"],
            "optimizer_grouping": config["optimizer"]["grouping"],
            "schedule": config["schedule"],
        },
    )

    set_deterministic_seed(INITIALIZATION_SEED)
    device = torch.device("cuda")
    model_config = DarkMindV2Config.from_json_file(MODEL_INPUT)
    validate_training_environment(model_config, micro_batch_size=2, precision="bf16")
    model = DarkMindV2ForCausalLM(model_config)
    initialization = apply_initialization_policy(model, config["initialization_policy"])
    model = model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config["schedule"])
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_base = TokenShardDataset(TOKENIZED_INPUT, "train")

    initial_train = evaluate_loss(model, order_data, sequence_length=512, micro_batch_size=2, device=device, maximum_tokens=TOKENS_PER_STEP)
    initial_validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
    initial_eval = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
    initial_checkpoint = run_dir / "checkpoints" / _checkpoint_name(0)
    initial_metadata = save_model_checkpoint(initial_checkpoint, model, config=config, step=0, data_hash=data_hash, order_hash=order_data.order_hash)
    evaluations = {"0": {"train": initial_train, "validation": initial_validation, "eval": initial_eval}}
    checkpoints = {"0": str(initial_checkpoint)}
    checkpoint_hashes = {"0": initial_metadata["model_sha256"]}

    activation_index = int(probe_manifest["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_base.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    initial_probe_losses = {name: evaluate_probe(model, probe, device) for name, probe in probe_manifest["probes"].items()}
    initial_diagnostics = _milestone_diagnostics(
        model,
        step=0,
        parameter_diagnostics=[],
        clipped_fraction=0.0,
        probe_losses=initial_probe_losses,
        activation_tokens=activation_tokens,
    )
    atomic_write_json(run_dir / "diagnostics" / "step_000000.json", initial_diagnostics)

    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    metrics = []
    diagnostic_snapshots = {"0": initial_diagnostics}
    clipped_steps = 0
    smoothed_loss = None
    data_position = 0
    try:
        for step in range(1, STAGE1_STEPS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            metric = optimizer_step(
                model,
                optimizer,
                order_data,
                data_position=data_position,
                diagnostic=step in MILESTONES,
                device=device,
            )
            scheduler.step()
            data_position += TOKENS_PER_STEP
            clipped_steps += int(metric["clipped"])
            smoothed_loss = metric["raw_train_loss"] if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * metric["raw_train_loss"]
            parameter_diagnostics = metric.pop("parameter_diagnostics")
            metric.update(
                {
                    "optimizer_step": step,
                    "tokens_consumed": data_position,
                    "learning_rate_applied": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "smoothed_train_loss": smoothed_loss,
                    "clipped_step_fraction": clipped_steps / step,
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "gpu": monitor.latest(),
                    "non_finite_event_count": 0,
                }
            )
            if step in MILESTONES:
                validation_result = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                eval_result = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                probe_losses = {name: evaluate_probe(model, probe, device) for name, probe in probe_manifest["probes"].items()}
                diagnostics = _milestone_diagnostics(
                    model,
                    step=step,
                    parameter_diagnostics=parameter_diagnostics or [],
                    clipped_fraction=clipped_steps / step,
                    probe_losses=probe_losses,
                    activation_tokens=activation_tokens,
                )
                atomic_write_json(run_dir / "diagnostics" / f"step_{step:06d}.json", diagnostics)
                diagnostic_snapshots[str(step)] = diagnostics
                checkpoint = run_dir / "checkpoints" / _checkpoint_name(step)
                metadata = save_model_checkpoint(checkpoint, model, config=config, step=step, data_hash=data_hash, order_hash=order_data.order_hash)
                checkpoints[str(step)] = str(checkpoint)
                checkpoint_hashes[str(step)] = metadata["model_sha256"]
                evaluations[str(step)] = {
                    "train": {"loss": metric["raw_train_loss"], "smoothed_loss": smoothed_loss},
                    "validation": validation_result,
                    "eval": eval_result,
                    "probes": probe_losses,
                }
                metric.update(
                    {
                        "validation_loss": validation_result["loss"],
                        "eval_loss": eval_result["loss"],
                        "checkpoint": str(checkpoint),
                    }
                )
            _append_jsonl(run_dir / "metrics.jsonl", metric)
            metrics.append(metric)
            if step % 5 == 0 or step in MILESTONES:
                atomic_write_json(run_dir / "progress.json", {"status": "training", "optimizer_step": step, "tokens_consumed": data_position})
            if step % 10 == 0 or step in MILESTONES:
                print(
                    f"arm={config['arm_name']} step={step} loss={metric['raw_train_loss']:.6f} "
                    f"lr={learning_rate:.9f} tok_s={metric['active_tokens_per_second']:.1f}",
                    flush=True,
                )
    finally:
        monitor.close()

    if data_position != STAGE1_TOKENS or len(metrics) != STAGE1_STEPS:
        raise ValueError("Phase 4C arm did not stop at the exact 5M authorization")
    validation_losses = [evaluations[str(step)]["validation"]["loss"] for step in MILESTONES]
    eval_losses = [evaluations[str(step)]["eval"]["loss"] for step in MILESTONES]
    best_validation = min(validation_losses)
    best_eval = min(eval_losses)
    initial_validation_loss = validation_losses[0]
    initial_eval_loss = eval_losses[0]
    final_validation = validation_losses[-1]
    final_eval = eval_losses[-1]
    all_parameter_diagnostics = [
        record
        for snapshot in diagnostic_snapshots.values()
        for record in snapshot["parameter_diagnostics"]
    ]
    maximum_update_ratio = max((record["update_to_weight_ratio"] for record in all_parameter_diagnostics), default=0.0)
    matrix_diagnostics = [record for record in all_parameter_diagnostics if record["parameter"].endswith(".weight")]
    maximum_matrix_update_ratio = max((record["update_to_weight_ratio"] for record in matrix_diagnostics), default=0.0)
    maximum_residual_ratio = max(snapshot["activation"]["final_to_early_residual_ratio"] for snapshot in diagnostic_snapshots.values())
    initial_residual_ratio = diagnostic_snapshots["0"]["activation"]["final_to_early_residual_ratio"]
    maximum_residual_growth_multiple = maximum_residual_ratio / initial_residual_ratio
    maximum_logit_std = max(snapshot["activation"]["logits"]["std"] for snapshot in diagnostic_snapshots.values())
    maximum_gradient = max(item["gradient_norm"] for item in metrics)
    abnormal = (
        maximum_matrix_update_ratio > DIAGNOSTIC_THRESHOLDS["matrix_parameter_update_to_weight_ratio"]
        or maximum_residual_growth_multiple > DIAGNOSTIC_THRESHOLDS["residual_ratio_growth_multiple_from_step0"]
        or maximum_logit_std > DIAGNOSTIC_THRESHOLDS["logit_standard_deviation"]
        or maximum_gradient > DIAGNOSTIC_THRESHOLDS["gradient_norm"]
    )
    summary = {
        "schema_version": "darkmind-v2-phase4c-arm-summary-v1",
        "result": "PASS",
        "integrity_pass": True,
        "arm_name": config["arm_name"],
        "sequence_order": "deterministic_stratified_v1",
        "sequence_order_hash": order_data.order_hash,
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_policy": initialization,
        "initialization_hash": initial_metadata["model_sha256"],
        "optimizer_grouping": config["optimizer"]["grouping"],
        "schedule": config["schedule"],
        "optimizer_steps": STAGE1_STEPS,
        "training_tokens": STAGE1_TOKENS,
        "initial_validation_loss": initial_validation_loss,
        "initial_eval_loss": initial_eval_loss,
        "best_validation_loss": best_validation,
        "best_eval_loss": best_eval,
        "best_combined_step": min(MILESTONES, key=lambda step: evaluations[str(step)]["validation"]["loss"] + evaluations[str(step)]["eval"]["loss"]),
        "final_validation_loss": final_validation,
        "final_eval_loss": final_eval,
        "validation_improvement_percent": (initial_validation_loss - final_validation) * 100.0 / initial_validation_loss,
        "eval_improvement_percent": (initial_eval_loss - final_eval) * 100.0 / initial_eval_loss,
        "validation_rebound_percent": rebound_percent(best_validation, final_validation),
        "eval_rebound_percent": rebound_percent(best_eval, final_eval),
        "consecutive_worsening_validation_evaluations": _max_consecutive_worsening(validation_losses),
        "consecutive_worsening_eval_evaluations": _max_consecutive_worsening(eval_losses),
        "last_three_validation_sustained_worsening": all(right > left for left, right in zip(validation_losses[-3:], validation_losses[-2:])),
        "last_three_eval_sustained_worsening": all(right > left for left, right in zip(eval_losses[-3:], eval_losses[-2:])),
        "gradient_norm_p50": percentile([item["gradient_norm"] for item in metrics], 0.50),
        "gradient_norm_p95": percentile([item["gradient_norm"] for item in metrics], 0.95),
        "gradient_norm_max": maximum_gradient,
        "clipped_step_fraction": clipped_steps / STAGE1_STEPS,
        "maximum_parameter_update_to_weight_ratio": maximum_update_ratio,
        "maximum_matrix_parameter_update_to_weight_ratio": maximum_matrix_update_ratio,
        "maximum_residual_final_to_early_ratio": maximum_residual_ratio,
        "initial_residual_final_to_early_ratio": initial_residual_ratio,
        "maximum_residual_ratio_growth_multiple_from_step0": maximum_residual_growth_multiple,
        "maximum_logit_standard_deviation": maximum_logit_std,
        "abnormal_diagnostic_growth": abnormal,
        "non_finite_events": 0,
        "evaluations": evaluations,
        "diagnostic_snapshots": {step: str(run_dir / "diagnostics" / f"step_{int(step):06d}.json") for step in diagnostic_snapshots},
        "checkpoints": checkpoints,
        "checkpoint_hashes": checkpoint_hashes,
        "active_tokens_per_second": STAGE1_TOKENS / sum(item["optimizer_step_duration_seconds"] for item in metrics),
        "wall_tokens_per_second": STAGE1_TOKENS / (time.perf_counter() - wall_started),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_stable": True,
    }
    summary["stability"] = classify_stability(summary)
    atomic_write_json(run_dir / "training_summary.json", summary)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "darkmind-v2-phase4c-arm-run-v1",
            "status": "training_complete",
            "arm_name": config["arm_name"],
            "initialization_hash": summary["initialization_hash"],
            "optimizer_steps": STAGE1_STEPS,
            "training_tokens": STAGE1_TOKENS,
            "stability": summary["stability"],
            "checkpoints": checkpoints,
            "runtime_stable": True,
            "mutable_runtime_outside_onedrive": True,
        },
    )
    atomic_write_json(run_dir / "progress.json", {"status": "training_complete", "optimizer_step": 610, "tokens_consumed": STAGE1_TOKENS})
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return summary


def choose_staged_peak() -> dict[str, Any]:
    raw_summaries = {
        name: json.loads((RUN_ROOT / name / "training_summary.json").read_text(encoding="utf-8"))
        for name in ARM_SPECS
    }
    summaries = {}
    reclassifications = {}
    for name, summary in raw_summaries.items():
        snapshots = {
            step: json.loads(Path(path).read_text(encoding="utf-8"))
            for step, path in summary["diagnostic_snapshots"].items()
        }
        normalized = normalize_diagnostic_summary(summary, snapshots)
        summaries[name] = normalized
        reclassifications[name] = {
            "original_stability": summary["stability"],
            "normalized_stability": normalized["stability"],
            "maximum_matrix_parameter_update_to_weight_ratio": normalized[
                "maximum_matrix_parameter_update_to_weight_ratio"
            ],
            "maximum_residual_ratio_growth_multiple_from_step0": normalized[
                "maximum_residual_ratio_growth_multiple_from_step0"
            ],
            "maximum_logit_standard_deviation": normalized["maximum_logit_standard_deviation"],
            "abnormal_diagnostic_growth": normalized["abnormal_diagnostic_growth"],
        }
    atomic_write_json(
        DIAGNOSTIC_ROOT / "core_arm_diagnostic_reclassification.json",
        {
            "schema_version": "darkmind-v2-phase4c-diagnostic-reclassification-v1",
            "result": "PASS",
            "method": (
                "immutable snapshots; matrix weights only for update ratio; residual ratio normalized to step 0"
            ),
            "arms": reclassifications,
        },
    )
    stable = [name for name, item in summaries.items() if item["stability"] == "stable"]
    pool = stable or list(summaries)
    selected = min(
        pool,
        key=lambda name: (
            (summaries[name]["final_validation_loss"] + summaries[name]["final_eval_loss"]) / 2.0,
            summaries[name]["schedule"]["peak_learning_rate"],
        ),
    )
    peak = float(summaries[selected]["schedule"]["peak_learning_rate"])
    payload = {
        "schema_version": "darkmind-v2-phase4c-staged-peak-selection-v1",
        "result": "PASS",
        "selected_reference_arm": selected,
        "selected_peak_learning_rate": peak,
        "stable_core_arm_available": bool(stable),
        "stable_core_arms": stable,
        "rule": "best final mean validation/eval among stable arms, otherwise all core arms; lower LR breaks exact ties",
    }
    atomic_write_json(DIAGNOSTIC_ROOT / "staged_peak_selection.json", payload)
    return payload


def prepare_followup_configs() -> dict[str, Any]:
    selection = choose_staged_peak()
    peak = selection["selected_peak_learning_rate"]
    if peak not in {0.000075, 0.0001}:
        raise ValueError("staged peak selection escaped approved values")
    arm4 = arm_config(
        "arm4_staged_decay_corrected_groups",
        {
            "peak_learning_rate": peak,
            "optimizer_grouping": "corrected_adamw_v1",
            "scheduler": "staged",
            "initialization_policy": "base_v1_standard_v1",
        },
    )
    validate_arm_config(arm4)
    atomic_write_json(CONFIG_ROOT / "arm4_staged_decay_corrected_groups.json", arm4)
    activation = json.loads((DIAGNOSTIC_ROOT / "initialization_activation_audit.json").read_text(encoding="utf-8"))
    optional_authorized = activation["findings"]["optional_versioned_initialization_arm_justified"] is True
    arm5_path = None
    if optional_authorized:
        arm5 = arm_config(
            "arm5_depth_scaled_init_staged",
            {
                "peak_learning_rate": peak,
                "optimizer_grouping": "corrected_adamw_v1",
                "scheduler": "staged",
                "initialization_policy": "base_v1_depth_scaled_residual_v2",
            },
        )
        validate_arm_config(arm5)
        arm5_path = CONFIG_ROOT / "arm5_depth_scaled_init_staged.json"
        atomic_write_json(arm5_path, arm5)
    payload = {
        "schema_version": "darkmind-v2-phase4c-followup-arm-design-v1",
        "result": "PASS",
        "arm4": str(CONFIG_ROOT / "arm4_staged_decay_corrected_groups.json"),
        "arm5": str(arm5_path) if arm5_path else None,
        "optional_initialization_arm_authorized": optional_authorized,
        "selected_peak": peak,
        "phase_25m_authorized": False,
    }
    atomic_write_json(RUNTIME_ROOT / "inputs" / "followup_arm_design.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-core")
    subparsers.add_parser("prepare-probes")
    subparsers.add_parser("prepare-followups")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-core":
        payload = prepare_core_configs()
    elif args.command == "prepare-probes":
        payload = prepare_probe_manifest()
    elif args.command == "prepare-followups":
        payload = prepare_followup_configs()
    else:
        payload = run_arm(args.config)
    printable = {key: value for key, value in payload.items() if key not in {"evaluations", "checkpoints", "checkpoint_hashes", "diagnostic_snapshots", "probes"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
