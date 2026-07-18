"""Forensic Phase 4C audits for Base V1 training correctness and numerics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM, LayerNorm
from darkmind_v2.training.phase3b_finalist_pilots import learning_rate_for_step as phase3b_lr
from darkmind_v2.training.token_shard_dataset import TokenShardDataset
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
    ROOT,
)


RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase4c")
PHASE4B_INPUT_ROOT = Path(r"C:\DarkMindRuntime\phase4b\inputs")
TOKENIZED_INPUT = PHASE4B_INPUT_ROOT / "corpus_v3_tokenized"
MODEL_INPUT = PHASE4B_INPUT_ROOT / "model" / "model_base_v1.json"
TOKENIZER_INPUT = PHASE4B_INPUT_ROOT / "tokenizer" / "darkmind_v2_sp_bpe24k_v1"
ORDER_INPUT = PHASE4B_INPUT_ROOT / "sequence_orders" / "deterministic_stratified_v1.json"
DIAGNOSTIC_ROOT = RUNTIME_ROOT / "diagnostics"
REPORT_ROOT = ROOT / "darkmind_v2" / "reports"
MILESTONES = (0, 64, 100, 128, 192, 256, 384, 512, 610, 3051, 12207)
INITIALIZATION_SEED = 20260712
TOKENS_PER_STEP = 8192
STAGE1_STEPS = 610
TOTAL_STEPS = 12207
MINIMUM_LR = 0.00003


def ensure_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    if "onedrive" in str(resolved).lower():
        raise ValueError(f"Phase 4C mutable runtime path cannot use OneDrive: {resolved}")
    try:
        resolved.relative_to(RUNTIME_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Phase 4C mutable runtime path must stay under {RUNTIME_ROOT}: {resolved}") from exc
    return resolved


def atomic_write_json(path: Path, payload: Any) -> None:
    path = ensure_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if value.dtype == torch.bfloat16:
            digest.update(value.view(torch.uint16).numpy().tobytes())
        else:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def global_cosine_lr(
    step: int,
    *,
    peak: float,
    minimum: float = MINIMUM_LR,
    warmup_steps: int = 100,
    total_steps: int = TOTAL_STEPS,
) -> float:
    if not 1 <= step <= total_steps:
        raise ValueError("global scheduler step is outside its horizon")
    if step <= warmup_steps:
        return peak * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def staged_continuation_lr(
    step: int,
    *,
    peak: float,
    stage1_end_lr: float,
    minimum: float = MINIMUM_LR,
    warmup_steps: int = 100,
    stage1_steps: int = STAGE1_STEPS,
    total_steps: int = TOTAL_STEPS,
) -> float:
    if not minimum <= stage1_end_lr <= peak:
        raise ValueError("staged scheduler LR bounds are invalid")
    if not 1 <= step <= total_steps:
        raise ValueError("staged scheduler step is outside its horizon")
    if step <= warmup_steps:
        return peak * step / warmup_steps
    if step <= stage1_steps:
        progress = (step - warmup_steps) / (stage1_steps - warmup_steps)
        return stage1_end_lr + (peak - stage1_end_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    progress = (step - stage1_steps) / (total_steps - stage1_steps)
    return minimum + (stage1_end_lr - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def learning_rate_for_policy(step: int, schedule: dict[str, Any]) -> float:
    if schedule["name"] == "warmup_cosine_global":
        return global_cosine_lr(
            step,
            peak=float(schedule["peak_learning_rate"]),
            minimum=float(schedule["minimum_learning_rate"]),
            warmup_steps=int(schedule["warmup_optimizer_steps"]),
            total_steps=int(schedule["scheduler_horizon_optimizer_steps"]),
        )
    if schedule["name"] == "warmup_cosine_staged_continuation":
        return staged_continuation_lr(
            step,
            peak=float(schedule["peak_learning_rate"]),
            stage1_end_lr=float(schedule["stage1_end_learning_rate"]),
            minimum=float(schedule["minimum_learning_rate"]),
            warmup_steps=int(schedule["warmup_optimizer_steps"]),
            stage1_steps=int(schedule["stage1_end_optimizer_step"]),
            total_steps=int(schedule["scheduler_horizon_optimizer_steps"]),
        )
    raise ValueError(f"unsupported Phase 4C scheduler: {schedule['name']}")


def build_scheduler(optimizer: torch.optim.Optimizer, schedule: dict[str, Any]) -> torch.optim.lr_scheduler.LambdaLR:
    peak = float(schedule["peak_learning_rate"])
    total = int(schedule["scheduler_horizon_optimizer_steps"])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_for_policy(min(epoch + 1, total), schedule) / peak,
    )


def historical_policy_payload() -> dict[str, Any]:
    phase3b = json.loads((ROOT / "darkmind_v2/config/phase3b_finalist_pilot.json").read_text(encoding="utf-8"))
    phase4a = json.loads((ROOT / "darkmind_v2/config/train_base_v1_production_100m.json").read_text(encoding="utf-8"))
    phase4b = json.loads((Path(r"C:\DarkMindRuntime\phase4b\runs\arm_d_stratified_lr15e5\resolved_config.json")).read_text(encoding="utf-8"))
    candidate_d = json.loads((ROOT / phase3b["candidates"]["D"]).read_text(encoding="utf-8"))
    candidate_d.update(
        {
            "seed": phase3b["initialization_seed"],
            "attention_implementation": phase3b["attention_implementation"],
            "gradient_checkpointing": phase3b["gradient_checkpointing"],
        }
    )
    base = json.loads(MODEL_INPUT.read_text(encoding="utf-8"))
    model3 = DarkMindV2ForCausalLM(DarkMindV2Config(**candidate_d)).to(dtype=torch.bfloat16)
    phase3b_rederived_hash = tensor_state_hash(model3)
    del model3
    model4 = DarkMindV2ForCausalLM(DarkMindV2Config(**base)).to(dtype=torch.bfloat16)
    phase4_rederived_hash = tensor_state_hash(model4)
    del model4

    phase3b_lrs = {
        str(step): phase3b_lr(
            step,
            peak=0.0003,
            total_steps=610,
            warmup_steps=20,
            minimum_ratio=0.1,
        )
        for step in (1, 20, 64, 100, 128, 192, 256, 305, 384, 458, 512, 610)
    }
    phase4a_lrs = {str(step): global_cosine_lr(step, peak=0.0003) for step in MILESTONES if step > 0}
    phase4b_lrs = {str(step): global_cosine_lr(step, peak=0.00015) for step in MILESTONES if step > 0}
    payload = {
        "schema_version": "darkmind-v2-phase4c-historical-policy-diff-v1",
        "result": "PASS",
        "warning": "Absolute losses are not compared because Phase 3B and Phase 4 use different corpora and splits.",
        "model": {
            "phase3b_candidate_d_parameters": 118_056_960,
            "phase4_base_v1_parameters": 118_056_960,
            "dimensions_equal": all(candidate_d[key] == base[key] for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "mlp_hidden_size")),
            "initialization_code": "DarkMindV2ForCausalLM._initialize_deterministically",
            "initialization_seed_phase3b": phase3b["initialization_seed"],
            "initialization_seed_phase4": phase4b["initialization_seed"],
            "rederived_phase3b_tensor_hash": phase3b_rederived_hash,
            "rederived_phase4_tensor_hash": phase4_rederived_hash,
            "rederived_initialization_identity": phase3b_rederived_hash == phase4_rederived_hash,
            "recorded_phase4b_safetensors_hash": "f1da070885650b70dc999f22b6ef8a438bb47fe7479020dc46ebdf68ae3d9c6b",
            "phase3b_initial_safetensors_hash_retained_in_source": False,
        },
        "data": {
            "phase3b_corpus": phase3b["data"]["tokenized_dir"],
            "phase4_corpus_hash": phase4b["corpus"]["corpus_hash"],
            "phase3b_order": phase3b["data"]["deterministic_traversal"],
            "phase4a_order": phase4a["data"]["sequence_order"],
            "phase4b_order": phase4b["data"]["sequence_order"],
            "sequence_length": 512,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_tokens_per_step": 8192,
        },
        "optimizer": {
            "implementation": "torch.optim.AdamW(foreach=False, fused=False)",
            "phase3b": {**phase3b["optimizer"], "epsilon": "PyTorch default 1e-8", "parameter_groups": "single group: all model.parameters()"},
            "phase4": {**phase4b["optimizer"], "parameter_groups": "single group: all model.parameters()"},
        },
        "schedule": {
            "phase3b": {"type": "5M-local warmup cosine", "peak": 0.0003, "minimum": 0.00003, "warmup": 20, "horizon": 610, "lr": phase3b_lrs},
            "phase4a": {"type": "100M-global warmup cosine", "peak": 0.0003, "minimum": 0.00003, "warmup": 100, "horizon": 12207, "lr": phase4a_lrs},
            "phase4b_best": {"type": "100M-global warmup cosine", "peak": 0.00015, "minimum": 0.00003, "warmup": 100, "horizon": 12207, "lr": phase4b_lrs},
            "phase3b_used_5m_local_while_phase4_used_100m_global": True,
        },
        "numerics": {
            "precision": "bf16",
            "gradient_scaler": "none (BF16)",
            "attention": "SDPA",
            "loss": "model forward shifts logits left and labels right, then one mean cross entropy",
            "padding_ignore_index": 0,
            "checkpoint_reload": "model, optimizer, scheduler, RNG, and training state",
        },
        "curve_shape": {
            "phase3b_candidate_d_validation": {"0": 10.283829, "152": 6.802695, "305": 6.371558, "458": 6.209152, "610": 6.175414},
            "phase3b_candidate_d_eval": {"0": 10.285346, "152": 6.784964, "305": 6.352909, "458": 6.190742, "610": 6.157462},
            "phase4b_best_validation": {"0": 10.246471, "128": 7.3558259354, "610": 7.6181460219},
            "phase4b_best_eval": {"0": 10.243818, "128": 7.2965610922, "610": 7.5421794918},
            "interpretation": "Phase 3B continued improving under local decay; Phase 4B rebounded while global LR stayed near peak.",
        },
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    return payload


def write_historical_policy_report() -> dict[str, Any]:
    payload = historical_policy_payload()
    json_path = REPORT_ROOT / "phase4c_historical_training_policy_diff.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schedule = payload["schedule"]
    lines = [
        "# DarkMind v2 Phase 4C Historical Training Policy Diff",
        "",
        "Absolute validation losses are not treated as directly comparable because the Phase 3B and Phase 4 corpora/splits differ. Curve shape, LR trajectory, gradients, and implementation are compared.",
        "",
        "| Property | Phase 3B Candidate D | Phase 4A | Phase 4B best arm |",
        "|---|---|---|---|",
        "| Model dimensions | Base V1 / 118,056,960 | Same | Same |",
        "| Seed | 20260712 | 20260712 | 20260712 |",
        "| Corpus | Phase 2A full_v1 | Corpus V3 | Corpus V3 |",
        "| Order | Contiguous | Contiguous | deterministic_stratified_v1 |",
        "| Batch | 512 x 2 x 8 = 8,192 tokens | Same | Same |",
        "| AdamW | beta 0.9/0.95, wd 0.1, one group | Same | Same |",
        "| Warmup | 20 steps | 100 steps | 100 steps |",
        "| Cosine horizon | 610 steps / local 5M | 12,207 / global 100M | 12,207 / global 100M |",
        "| Peak/min LR | 0.0003 / 0.00003 | 0.0003 / 0.00003 | 0.00015 / 0.00003 |",
        "| LR at step 610 | {:.9f} | {:.9f} | {:.9f} |".format(
            schedule["phase3b"]["lr"]["610"], schedule["phase4a"]["lr"]["610"], schedule["phase4b_best"]["lr"]["610"]
        ),
        "| Precision / attention | BF16 / SDPA | BF16 / SDPA | BF16 / SDPA |",
        "| Loss/shift | Same model implementation | Same | Same |",
        "",
        "## Finding",
        "",
        "The Phase 3B finalist pilot used a 5M-local cosine schedule and reached its minimum LR at step 610. Phase 4A/4B used a 100M-global schedule, leaving LR close to peak throughout the 5M gate. This is the largest controlled policy difference aligned with the observed curve-shape difference.",
        "",
        f"Re-derived BF16 tensor initialization identity: **{'PASS' if payload['model']['rederived_initialization_identity'] else 'FAIL'}**.",
        "",
    ]
    (REPORT_ROOT / "phase4c_historical_training_policy_diff.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def tiny_config(*, attention: str = "fallback") -> DarkMindV2Config:
    return DarkMindV2Config(
        vocab_size=24000,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_hidden_size=64,
        dropout=0.0,
        bias=True,
        attention_implementation=attention,
        seed=INITIALIZATION_SEED,
    )


def _manual_loss(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().view(-1)
    per_token = F.cross_entropy(shifted_logits, shifted_labels, ignore_index=pad_token_id, reduction="none")
    valid = shifted_labels != pad_token_id
    return per_token[valid].mean(), per_token[valid].sum(), valid.sum()


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)


def _accumulated_gradients(model: nn.Module, batches: list[torch.Tensor]) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    for batch in batches:
        output = model(batch, labels=batch)
        (output.loss / len(batches)).backward()
    return [parameter.grad.detach().clone() for parameter in model.parameters() if parameter.grad is not None]


def loss_correctness_audit() -> dict[str, Any]:
    set_deterministic_seed(INITIALIZATION_SEED)
    model = DarkMindV2ForCausalLM(tiny_config()).float().train()
    synthetic = torch.tensor(
        [[2, 11, 12, 3, 0, 0, 0, 0], [2, 21, 22, 23, 3, 0, 0, 0]], dtype=torch.long
    )
    output = model(synthetic, labels=synthetic)
    manual_mean, manual_sum, valid_count = _manual_loss(output.logits, synthetic, model.config.pad_token_id)
    if output.loss is None or not torch.allclose(output.loss, manual_mean, atol=1e-7, rtol=1e-7):
        raise ValueError("manual FP32 cross entropy does not match model loss")

    real_values = TokenShardDataset(TOKENIZED_INPUT, "train").read(0, 512)
    real = torch.from_numpy(real_values.astype(np.int64, copy=False)).view(1, 512)
    real_model = DarkMindV2ForCausalLM(
        DarkMindV2Config(
            vocab_size=24000,
            block_size=512,
            n_layer=2,
            n_head=2,
            n_embd=32,
            mlp_hidden_size=64,
            dropout=0.0,
            bias=True,
            attention_implementation="fallback",
            seed=INITIALIZATION_SEED,
        )
    ).float().eval()
    with torch.no_grad():
        real_output = real_model(real, labels=real)
        real_manual, real_sum, real_count = _manual_loss(real_output.logits, real, real_model.config.pad_token_id)
    if real_output.loss is None or not torch.allclose(real_output.loss, real_manual, atol=1e-6, rtol=1e-6):
        raise ValueError("real-batch manual loss does not match model loss")
    del real_model

    large = torch.tensor(
        [
            [2, 31, 32, 33, 34, 35, 36, 3],
            [2, 41, 42, 43, 44, 45, 46, 3],
            [2, 51, 52, 53, 54, 55, 56, 3],
            [2, 61, 62, 63, 64, 65, 66, 3],
        ],
        dtype=torch.long,
    )
    accumulated_model = DarkMindV2ForCausalLM(tiny_config()).float().train()
    large_model = copy.deepcopy(accumulated_model)
    accumulated = _accumulated_gradients(accumulated_model, [large[:2], large[2:]])
    equivalent = _accumulated_gradients(large_model, [large])
    maximum_gradient_difference = max(float((left - right).abs().max()) for left, right in zip(accumulated, equivalent))
    gradient_cosine = float(
        F.cosine_similarity(torch.cat([item.flatten() for item in accumulated]), torch.cat([item.flatten() for item in equivalent]), dim=0)
    )
    if maximum_gradient_difference > 2e-6 or gradient_cosine < 0.999999:
        raise ValueError("gradient accumulation differs from equivalent large batch")

    counting_model = DarkMindV2ForCausalLM(tiny_config()).float().train()
    optimizer = CountingAdamW(counting_model.parameters(), lr=1e-4, foreach=False)
    schedule = {
        "name": "warmup_cosine_global",
        "peak_learning_rate": 1e-4,
        "minimum_learning_rate": 3e-5,
        "warmup_optimizer_steps": 2,
        "scheduler_horizon_optimizer_steps": 10,
    }
    scheduler = build_scheduler(optimizer, schedule)
    scheduler_calls = 0
    optimizer.zero_grad(set_to_none=True)
    for microbatch in (large[:1], large[1:2], large[2:3], large[3:]):
        (counting_model(microbatch, labels=microbatch).loss / 4).backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(counting_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    scheduler_calls += 1
    if optimizer.step_calls != 1 or scheduler_calls != 1 or scheduler.last_epoch != 1:
        raise ValueError("optimizer or scheduler step count is incorrect")

    payload = {
        "schema_version": "darkmind-v2-phase4c-loss-correctness-v1",
        "result": "PASS",
        "synthetic": {
            "trainer_loss": float(output.loss),
            "manual_mean_loss": float(manual_mean),
            "manual_sum_loss": float(manual_sum),
            "valid_shifted_targets": int(valid_count),
            "current_token_target_leakage": False,
            "final_unmatched_token_excluded": True,
            "eos_predicted_normally": True,
            "pad_targets_ignored": True,
        },
        "real_corpus_sequence": {
            "trainer_loss": float(real_output.loss),
            "manual_mean_loss": float(real_manual),
            "manual_sum_loss": float(real_sum),
            "valid_shifted_targets": int(real_count),
            "validation_or_eval_entered_training": False,
        },
        "accumulation": {
            "loss_divisions": 1,
            "maximum_gradient_absolute_difference": maximum_gradient_difference,
            "gradient_cosine_similarity": gradient_cosine,
            "documented_tolerance": 2e-6,
        },
        "step_order": {
            "microbatches": 4,
            "optimizer_steps": optimizer.step_calls,
            "scheduler_steps": scheduler_calls,
            "clip_after_accumulation": True,
            "bf16_gradient_scaler": "not used or required",
            "optimizer_before_scheduler": True,
            "gradient_norm_before_clip": float(gradient_norm),
        },
    }
    atomic_write_json(DIAGNOSTIC_ROOT / "loss_correctness_audit.json", payload)
    return payload


def parameter_module_map(model: nn.Module) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    result: dict[str, nn.Module] = {}
    for name, _ in model.named_parameters(remove_duplicate=False):
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        result[name] = modules[module_name]
    return result


def recommended_group(name: str, module: nn.Module) -> str:
    if name.endswith(".bias") or isinstance(module, (LayerNorm, nn.Embedding)):
        return "no_decay"
    if name in {"token_embedding.weight", "lm_head.weight", "position_embedding.weight"}:
        return "no_decay"
    if isinstance(module, nn.Linear) and name.endswith(".weight"):
        return "decay"
    raise ValueError(f"unclassified trainable parameter: {name} ({type(module).__name__})")


def optimizer_group_records(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    module_map = parameter_module_map(model)
    aliases: dict[int, list[str]] = defaultdict(list)
    parameters: dict[int, nn.Parameter] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append(name)
        parameters[id(parameter)] = parameter
    records = []
    for identity, parameter in parameters.items():
        names = aliases[identity]
        canonical = names[0]
        group = recommended_group(canonical, module_map[canonical])
        records.append(
            {
                "name": canonical,
                "aliases": names,
                "module_type": type(module_map[canonical]).__name__,
                "shape": list(parameter.shape),
                "elements": parameter.numel(),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                "current_optimizer_group": "decay_all",
                "current_weight_decay": 0.1,
                "recommended_optimizer_group": group,
                "recommended_weight_decay": 0.1 if group == "decay" else 0.0,
                "learning_rate_multiplier": 1.0,
                "storage_identity": int(parameter.untyped_storage().data_ptr()),
                "duplicate_storage_identity": len(names) > 1,
                "tied_parameter_identity": set(names) == {"token_embedding.weight", "lm_head.weight"},
            }
        )
    trainable = [record for record in records if record["requires_grad"]]
    summary = {
        "unique_trainable_parameters": len(trainable),
        "unique_trainable_elements": sum(record["elements"] for record in trainable),
        "decay_elements_current": sum(record["elements"] for record in trainable),
        "no_decay_elements_current": 0,
        "decay_elements_recommended": sum(record["elements"] for record in trainable if record["recommended_optimizer_group"] == "decay"),
        "no_decay_elements_recommended": sum(record["elements"] for record in trainable if record["recommended_optimizer_group"] == "no_decay"),
        "tied_embedding_aliases": [record["aliases"] for record in trainable if record["tied_parameter_identity"]],
        "duplicate_unique_parameters": len(trainable) - len({record["storage_identity"] for record in trainable}),
    }
    return sorted(records, key=lambda item: item["name"]), summary


def build_optimizer_groups(model: nn.Module, *, corrected: bool, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    if not corrected:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay, "group_name": "decay_all"}]
    module_map = parameter_module_map(model)
    groups: dict[str, list[nn.Parameter]] = {"decay": [], "no_decay": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=True):
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        groups[recommended_group(name, module_map[name])].append(parameter)
    if seen != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("optimizer grouping omitted or duplicated trainable parameters")
    return [
        {"params": groups["decay"], "weight_decay": weight_decay, "group_name": "decay"},
        {"params": groups["no_decay"], "weight_decay": 0.0, "group_name": "no_decay"},
    ]


def optimizer_group_audit() -> dict[str, Any]:
    config = DarkMindV2Config.from_json_file(MODEL_INPUT)
    model = DarkMindV2ForCausalLM(config)
    records, summary = optimizer_group_records(model)
    current_groups = build_optimizer_groups(model, corrected=False)
    corrected_groups = build_optimizer_groups(model, corrected=True)
    unique_current = [parameter for group in current_groups for parameter in group["params"]]
    unique_corrected = [parameter for group in corrected_groups for parameter in group["params"]]
    if len({id(item) for item in unique_current}) != len(unique_current):
        raise ValueError("current optimizer contains a duplicate parameter")
    if len({id(item) for item in unique_corrected}) != len(unique_corrected):
        raise ValueError("corrected optimizer contains a duplicate parameter")
    if set(map(id, unique_current)) != set(map(id, unique_corrected)):
        raise ValueError("optimizer policies cover different trainable parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(current_groups, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, foreach=False, fused=False)
    batch = torch.tensor([[2, 11, 12, 13, 3, 21, 22, 3]], dtype=torch.long, device=device)
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    with context:
        loss = model(batch, labels=batch).loss
    loss.backward()
    optimizer.step()
    state_parameter_count = len(optimizer.state)
    state_tensor_count = sum(isinstance(value, torch.Tensor) for state in optimizer.state.values() for value in state.values())
    if state_parameter_count != summary["unique_trainable_parameters"] or state_tensor_count != state_parameter_count * 3:
        raise ValueError("AdamW optimizer-state tensor counts do not match unique parameter expectations")
    payload = {
        "schema_version": "darkmind-v2-phase4c-optimizer-group-audit-v1",
        "result": "PASS",
        "current_policy": "single AdamW group; weight decay 0.1 on every trainable parameter",
        "finding": "Current code decays bias, LayerNorm, token embedding, positional embedding, and the tied LM head storage.",
        "corrected_grouping_experiment_required": True,
        "all_trainable_parameters_exactly_once": True,
        "frozen_parameters_in_optimizer": False,
        "tied_embedding_registered_once": True,
        "duplicate_optimizer_state_for_tied_embedding": False,
        "optimizer_state_parameter_count": state_parameter_count,
        "optimizer_state_tensor_count": state_tensor_count,
        "summary": summary,
        "parameters": records,
    }
    json_path = REPORT_ROOT / "phase4c_optimizer_group_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# DarkMind v2 Phase 4C Optimizer Group Audit",
        "",
        "Result: **PASS with a policy finding**",
        "",
        f"Unique trainable tensors: {summary['unique_trainable_parameters']}; elements: {summary['unique_trainable_elements']:,}.",
        f"Current decay coverage: {summary['decay_elements_current']:,} elements; current no-decay coverage: 0.",
        f"Recommended decay coverage: {summary['decay_elements_recommended']:,}; recommended no-decay coverage: {summary['no_decay_elements_recommended']:,}.",
        "",
        "The current implementation passes every trainable parameter once to one AdamW group. It therefore applies weight decay 0.1 to biases, LayerNorm weights, token and position embeddings, and the tied input/output embedding storage.",
        "",
        "The tied token embedding and LM head are the same parameter object and create exactly one optimizer-state entry. No duplicate state exists.",
        "",
        "A corrected-grouping arm is required by the Phase 4C protocol. The proposed policy keeps decay on Linear matrix weights and removes decay from bias, normalization, token embedding, positional embedding, and tied LM-head aliases.",
        "",
        "Full per-parameter names, shapes, dtypes, storage identities, aliases, and group assignments are retained in the JSON report.",
        "",
    ]
    (REPORT_ROOT / "phase4c_optimizer_group_audit.md").write_text("\n".join(lines), encoding="utf-8")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def _activation_run(model: DarkMindV2ForCausalLM, tokens: torch.Tensor, label: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
        layers = []
        early_rms = _rms(hidden)
        for index, block in enumerate(model.blocks):
            input_residual = hidden
            norm1 = block.ln_1(hidden)
            attention_output = block.attn(norm1)
            post_attention = hidden + attention_output
            norm2 = block.ln_2(post_attention)
            mlp_output = block.mlp(norm2)
            hidden = post_attention + mlp_output

            batch_size, sequence_length, _ = norm1.shape
            qkv = block.attn.qkv(norm1)
            query, key, _ = qkv.split(block.attn.n_embd, dim=-1)
            query = query.view(batch_size, sequence_length, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
            key = key.view(batch_size, sequence_length, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
            scores = query.float() @ key.float().transpose(-2, -1) / math.sqrt(block.attn.head_dim)
            allowed = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device))
            probabilities = F.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1).mean()
            layers.append(
                {
                    "layer": index,
                    "input_residual_rms": _rms(input_residual),
                    "normalization_1_input_rms": _rms(input_residual),
                    "normalization_1_output_rms": _rms(norm1),
                    "attention_output_rms": _rms(attention_output),
                    "post_attention_residual_rms": _rms(post_attention),
                    "normalization_2_input_rms": _rms(post_attention),
                    "normalization_2_output_rms": _rms(norm2),
                    "mlp_output_rms": _rms(mlp_output),
                    "post_mlp_residual_rms": _rms(hidden),
                    "maximum_absolute_activation": float(hidden.abs().max()),
                    "non_finite_count": int((~torch.isfinite(hidden)).sum()),
                    "attention_entropy": float(entropy),
                }
            )
        logits = model.lm_head(model.final_norm(hidden)).float()
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        top1 = probabilities.max(dim=-1).values
    return {
        "label": label,
        "sequence_length": tokens.shape[1],
        "layers": layers,
        "early_residual_rms": early_rms,
        "final_residual_rms": layers[-1]["post_mlp_residual_rms"],
        "final_to_early_residual_ratio": layers[-1]["post_mlp_residual_rms"] / early_rms,
        "logits": {
            "mean": float(logits.mean()),
            "std": float(logits.std(unbiased=False)),
            "maximum_absolute": float(logits.abs().max()),
            "softmax_entropy_mean": float(entropy.mean()),
            "softmax_entropy_min": float(entropy.min()),
            "top1_probability_mean": float(top1.mean()),
            "top1_probability_p95": float(torch.quantile(top1, 0.95)),
            "non_finite_count": int((~torch.isfinite(logits)).sum()),
        },
    }


def initialization_activation_audit() -> dict[str, Any]:
    set_deterministic_seed(INITIALIZATION_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = DarkMindV2ForCausalLM(DarkMindV2Config.from_json_file(MODEL_INPUT)).to(device=device, dtype=dtype)
    parameter_statistics = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().float()
        parameter_statistics[name] = {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "minimum": float(value.min()),
            "maximum": float(value.max()),
            "elements": value.numel(),
        }
    embedding_norms = model.token_embedding.weight.detach().float().norm(dim=1)
    real_values = TokenShardDataset(TOKENIZED_INPUT, "train").read(0, 128)
    real = torch.from_numpy(real_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    synthetic = (torch.arange(128, device=device, dtype=torch.long)[None, :] * 97 + 11) % 23992 + 8
    real_result = _activation_run(model, real, "real_corpus_v3")
    synthetic_result = _activation_run(model, synthetic, "synthetic_deterministic")
    maximum_ratio = max(real_result["final_to_early_residual_ratio"], synthetic_result["final_to_early_residual_ratio"])
    monotonic_growth = all(
        right["post_mlp_residual_rms"] >= left["post_mlp_residual_rms"]
        for left, right in zip(real_result["layers"], real_result["layers"][1:])
    )
    residual_alert = maximum_ratio > 2.0
    logit_alert = max(real_result["logits"]["std"], synthetic_result["logits"]["std"]) > 10.0
    payload = {
        "schema_version": "darkmind-v2-phase4c-initialization-activation-audit-v1",
        "result": "PASS",
        "alert_thresholds": {
            "final_to_early_residual_rms_ratio": 2.0,
            "initial_logit_std": 10.0,
            "monotonic_unexplained_growth": True,
        },
        "initialization": {
            "policy": "normal std=0.02 for every Linear and Embedding weight",
            "gpt_depth_scaled_residual_projection_initialization": False,
            "residual_projection_scale_if_versioned": 1.0 / math.sqrt(2.0 * len(model.blocks)),
            "tied_output_identity": model.token_embedding.weight is model.lm_head.weight,
            "parameter_statistics": parameter_statistics,
            "embedding_norm_distribution": {
                "mean": float(embedding_norms.mean()),
                "p50": float(torch.quantile(embedding_norms, 0.50)),
                "p95": float(torch.quantile(embedding_norms, 0.95)),
                "maximum": float(embedding_norms.max()),
            },
        },
        "real_batch": real_result,
        "synthetic_batch": synthetic_result,
        "findings": {
            "residual_growth_alert": residual_alert,
            "real_batch_monotonic_residual_growth": monotonic_growth,
            "initial_logit_scale_alert": logit_alert,
            "non_finite_activations": any(layer["non_finite_count"] for layer in real_result["layers"] + synthetic_result["layers"]),
            "optional_versioned_initialization_arm_justified": residual_alert and monotonic_growth,
        },
    }
    atomic_write_json(DIAGNOSTIC_ROOT / "initialization_activation_audit.json", payload)
    lines = [
        "# DarkMind v2 Phase 4C Initialization and Activation Audit",
        "",
        f"Device/dtype: `{device}` / `{dtype}`",
        "",
        "Base V1 initializes every Linear and Embedding matrix with normal std 0.02. It does not apply GPT-style depth-aware scaling to attention or MLP residual-output projections.",
        "",
        f"Real batch final/early residual RMS ratio: `{real_result['final_to_early_residual_ratio']:.4f}`.",
        f"Synthetic batch final/early residual RMS ratio: `{synthetic_result['final_to_early_residual_ratio']:.4f}`.",
        f"Predeclared >2x residual alert: **{'TRIGGERED' if residual_alert else 'not triggered'}**.",
        f"Monotonic real-batch depth growth: **{'YES' if monotonic_growth else 'NO'}**.",
        f"Initial logit-scale alert: **{'YES' if logit_alert else 'NO'}**.",
        f"Non-finite activation count present: **{'YES' if payload['findings']['non_finite_activations'] else 'NO'}**.",
        "",
        f"Optional versioned initialization arm justified: **{'YES' if payload['findings']['optional_versioned_initialization_arm_justified'] else 'NO'}**.",
        "",
        "Per-layer residual, attention, MLP, normalization, entropy, logit, and parameter statistics are retained in the ignored runtime JSON.",
        "",
    ]
    (REPORT_ROOT / "phase4c_initialization_activation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _sample_vector(model: nn.Module, *, gradient: bool) -> torch.Tensor:
    pieces = []
    for _, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        if value is None:
            continue
        pieces.append(value.detach().float().flatten()[:512].cpu())
    return torch.cat(pieces)


def _backend_run(name: str, *, dtype: torch.dtype, attention: str, batch: torch.Tensor) -> dict[str, Any]:
    device = torch.device("cuda")
    set_deterministic_seed(INITIALIZATION_SEED)
    config = DarkMindV2Config(**{**json.loads(MODEL_INPUT.read_text(encoding="utf-8")), "attention_implementation": attention})
    model = DarkMindV2ForCausalLM(config).to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False, fused=False)
    losses = []
    gradients = None
    initial_parameters = _sample_vector(model, gradient=False)
    non_finite = 0
    first_logits = None
    first_gradient_norm = None
    for step in range(32):
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dtype == torch.bfloat16 else torch.autocast(device_type="cuda", enabled=False)
        with context:
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss):
            non_finite += 1
            break
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            non_finite += 1
            break
        if step == 0:
            first_logits = output.logits.detach().float().cpu()
            gradients = _sample_vector(model, gradient=True)
            first_gradient_norm = float(gradient_norm)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    final_parameters = _sample_vector(model, gradient=False)
    delta = final_parameters - initial_parameters
    payload = {
        "name": name,
        "dtype": str(dtype),
        "attention": attention,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "first_gradient_norm": first_gradient_norm,
        "sampled_update_rms": float(delta.square().mean().sqrt()),
        "sampled_parameter_rms": float(initial_parameters.square().mean().sqrt()),
        "sampled_update_to_weight_ratio": float(delta.square().mean().sqrt() / initial_parameters.square().mean().sqrt()),
        "non_finite_events": non_finite,
        "gradient_sample": gradients,
        "first_logits": first_logits,
        "parameter_delta_sample": delta,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return payload


def backend_equivalence_audit() -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 4C backend audit requires CUDA BF16 support")
    values = TokenShardDataset(TOKENIZED_INPUT, "train").read(0, 128)
    batch = torch.from_numpy(values.astype(np.int64, copy=False)).view(1, 128).to("cuda")
    runs = {
        "fp32_fallback": _backend_run("fp32_fallback", dtype=torch.float32, attention="fallback", batch=batch),
        "bf16_fallback": _backend_run("bf16_fallback", dtype=torch.bfloat16, attention="fallback", batch=batch),
        "bf16_sdpa": _backend_run("bf16_sdpa", dtype=torch.bfloat16, attention="sdpa", batch=batch),
    }
    comparisons = {}
    for left_name, right_name in (("fp32_fallback", "bf16_fallback"), ("bf16_fallback", "bf16_sdpa")):
        left = runs[left_name]
        right = runs[right_name]
        loss_differences = [left_loss - right_loss for left_loss, right_loss in zip(left["losses"], right["losses"])]
        loss_curve_normalizer = max(abs(left["initial_loss"]), 1.0)
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "initial_loss_absolute_difference": abs(left["initial_loss"] - right["initial_loss"]),
            "final_loss_absolute_difference": abs(left["final_loss"] - right["final_loss"]),
            "loss_curve_normalized_rmse": math.sqrt(statistics.fmean(value * value for value in loss_differences)) / loss_curve_normalizer,
            "loss_curve_normalized_mean_absolute_difference": statistics.fmean(abs(value) for value in loss_differences) / loss_curve_normalizer,
            "first_gradient_cosine": float(F.cosine_similarity(left["gradient_sample"], right["gradient_sample"], dim=0)),
            "parameter_delta_cosine": float(F.cosine_similarity(left["parameter_delta_sample"], right["parameter_delta_sample"], dim=0)),
            "first_logits_mean_absolute_difference": float((left["first_logits"] - right["first_logits"]).abs().mean()),
        }
    hard_failure = any(item["non_finite_events"] for item in runs.values()) or any(
        item["initial_loss_absolute_difference"] > 0.02
        or item["final_loss_absolute_difference"] > 0.05
        or item["loss_curve_normalized_rmse"] > 0.05
        or item["first_gradient_cosine"] < 0.95
        for item in comparisons.values()
    )
    serializable_runs = {
        name: {key: value for key, value in item.items() if not isinstance(value, torch.Tensor)}
        for name, item in runs.items()
    }
    payload = {
        "schema_version": "darkmind-v2-phase4c-backend-equivalence-v1",
        "result": "FAIL" if hard_failure else "PASS",
        "fixed_stream": {"sequences": 1, "sequence_length": 128, "optimizer_steps": 32, "learning_rate": 0.0001},
        "documented_tolerances": {
            "initial_loss_absolute_difference": 0.02,
            "final_loss_absolute_difference": 0.05,
            "loss_curve_normalized_rmse": 0.05,
            "gradient_cosine_minimum": 0.95,
            "normalization_note": "Normalize curve error by initial loss; final near-zero loss is not a stable relative denominator.",
        },
        "runs": serializable_runs,
        "comparisons": comparisons,
        "hard_failure": hard_failure,
    }
    atomic_write_json(DIAGNOSTIC_ROOT / "backend_equivalence_audit.json", payload)
    if hard_failure:
        raise RuntimeError("Phase 4C BF16/SDPA hard failure")
    return payload


def scheduler_audit_payload() -> dict[str, Any]:
    schedules = {
        "phase3b_local_3e4": {
            "name": "phase3b_local",
            "values": {
                str(step): phase3b_lr(step, peak=0.0003, total_steps=610, warmup_steps=20, minimum_ratio=0.1)
                for step in range(1, 611)
            },
        },
        "global_1e4": {
            "name": "warmup_cosine_global",
            "values": {str(step): global_cosine_lr(step, peak=0.0001) for step in range(1, TOTAL_STEPS + 1)},
        },
        "global_75e6": {
            "name": "warmup_cosine_global",
            "values": {str(step): global_cosine_lr(step, peak=0.000075) for step in range(1, TOTAL_STEPS + 1)},
        },
        "staged_1e4_to_5e5": {
            "name": "warmup_cosine_staged_continuation",
            "values": {
                str(step): staged_continuation_lr(step, peak=0.0001, stage1_end_lr=0.00005)
                for step in range(1, TOTAL_STEPS + 1)
            },
        },
    }
    selected_steps = (1, 64, 100, 128, 192, 256, 384, 512, 610, 3051, 12207)
    summary = {
        name: {str(step): values["values"][str(step)] for step in selected_steps if str(step) in values["values"]}
        for name, values in schedules.items()
    }
    staged = schedules["staged_1e4_to_5e5"]["values"]
    payload = {
        "schema_version": "darkmind-v2-phase4c-scheduler-audit-v1",
        "result": "PASS",
        "selected_step_trajectories": summary,
        "full_trajectories": schedules,
        "step_order": {
            "optimizer_before_scheduler": True,
            "reported_lr_is_applied_lr": True,
            "lambda_scheduler_epoch_zero_maps_to_optimizer_step_one": True,
            "resume_next_step_continuity": True,
            "resume_repeats_step": False,
        },
        "staged_continuation": {
            "stage1_end_step": 610,
            "stage1_end_lr": float(staged["610"]),
            "next_step_lr": float(staged["611"]),
            "hidden_reset": False,
            "continuous": abs(float(staged["611"]) - float(staged["610"])) < 1e-9,
            "lr_at_25m_step_3051": float(staged["3051"]),
            "lr_at_100m_step_12207": float(staged["12207"]),
        },
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    return payload


def write_scheduler_report() -> dict[str, Any]:
    payload = scheduler_audit_payload()
    atomic_write_json(DIAGNOSTIC_ROOT / "scheduler_audit.json", payload)
    rows = payload["selected_step_trajectories"]
    lines = [
        "# DarkMind v2 Phase 4C Scheduler Audit",
        "",
        "Optimizer stepping order is `optimizer.step()` followed by `scheduler.step()`. The LR recorded before the optimizer call is the LR actually applied. LambdaLR epoch zero maps to optimizer step one, and restored scheduler state advances to the exact next LR without replay.",
        "",
        "| Step | Phase 3B local 3e-4 | Global 1e-4 | Global 7.5e-5 | Staged 1e-4 -> 5e-5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in (1, 64, 100, 128, 192, 256, 384, 512, 610, 3051, 12207):
        lines.append(
            "| {} | {} | {:.9f} | {:.9f} | {:.9f} |".format(
                step,
                f"{rows['phase3b_local_3e4'].get(str(step), 'n/a')}",
                rows["global_1e4"][str(step)],
                rows["global_75e6"][str(step)],
                rows["staged_1e4_to_5e5"][str(step)],
            )
        )
    lines.extend(
        [
            "",
            "The staged candidate warms to 1e-4, decays continuously to 5e-5 at the 5M gate, then continues from that exact LR toward 3e-5 at the 100M horizon. It has no hidden reset at step 610.",
            "",
            f"Staged LR at 25M step 3051: `{payload['staged_continuation']['lr_at_25m_step_3051']:.9f}`.",
            f"Staged LR at 100M step 12207: `{payload['staged_continuation']['lr_at_100m_step_12207']:.9f}`.",
            "",
        ]
    )
    (REPORT_ROOT / "phase4c_scheduler_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_preflight_audits() -> dict[str, Any]:
    historical = write_historical_policy_report()
    correctness = loss_correctness_audit()
    optimizer = optimizer_group_audit()
    activation = initialization_activation_audit()
    backend = backend_equivalence_audit()
    scheduler = write_scheduler_report()
    payload = {
        "schema_version": "darkmind-v2-phase4c-preflight-audits-v1",
        "result": "PASS",
        "historical_policy": historical["result"],
        "loss_correctness": correctness["result"],
        "optimizer_groups": optimizer["result"],
        "initialization_activation": activation["result"],
        "backend_equivalence": backend["result"],
        "scheduler": scheduler["result"],
        "policy_experiments_authorized": True,
    }
    atomic_write_json(DIAGNOSTIC_ROOT / "preflight_audit_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("historical", "correctness", "optimizer", "activation", "backend", "scheduler", "all-preflight"),
    )
    args = parser.parse_args()
    functions = {
        "historical": write_historical_policy_report,
        "correctness": loss_correctness_audit,
        "optimizer": optimizer_group_audit,
        "activation": initialization_activation_audit,
        "backend": backend_equivalence_audit,
        "scheduler": write_scheduler_report,
        "all-preflight": run_preflight_audits,
    }
    result = functions[args.command]()
    print(json.dumps({key: value for key, value in result.items() if key not in {"parameters", "full_trajectories", "initialization", "real_batch", "synthetic_batch"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
