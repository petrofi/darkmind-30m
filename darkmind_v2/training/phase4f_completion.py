"""Complete the first deterministic Corpus V3 pass from the exact 75M checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.modeling.model_io import save_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from darkmind_v2.training.phase3b_finalist_pilots import evaluate_loss
from darkmind_v2.training.phase4b_factorial import OrderedTokenDataset, percentile, rebound_percent
from darkmind_v2.training.phase4c_confirmation import rng_fingerprint
from darkmind_v2.training.phase4c_diagnostics import MODEL_INPUT, ORDER_INPUT, TOKENIZED_INPUT, TOKENIZER_INPUT, learning_rate_for_policy
from darkmind_v2.training.phase4c_training import _milestone_diagnostics, evaluate_probe
from darkmind_v2.training.phase4d_stage2 import (
    EXPECTED_V2_FILE_HASH,
    ROOT,
    V2_CONFIG,
    _create_model_stack,
    _validate_safetensors,
    _validate_shards,
    directory_size,
    load_json,
    sha256_file,
    state_hash,
    training_step,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_base_v1_stage1 import GpuMonitor
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
)


RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase4f")
RUN_DIR = RUNTIME_ROOT / "runs" / "base_v1_first_corpus_pass_completion_v2"
PHASE4E_RUN = Path(r"C:\DarkMindRuntime\phase4e\runs\base_v1_stage3_first_corpus_pass_v2")
SOURCE_CHECKPOINT = PHASE4E_RUN / "checkpoints" / "step_009155_tokens_074997760"
AUTHORIZATION_PATH = ROOT / "darkmind_v2" / "config" / "train_base_v1_first_pass_completion_authorization.json"
SOURCE_STEP = 9_155
GATE_85_STEP = 10_375
GATE_90_STEP = 10_986
GATE_95_STEP = 11_596
FINAL_STEP = 11_972
TOKENS_PER_STEP = 8_192
SEQUENCES_PER_STEP = 16
SOURCE_TOKENS = SOURCE_STEP * TOKENS_PER_STEP
FINAL_TOKENS = FINAL_STEP * TOKENS_PER_STEP
UNIQUE_SEQUENCE_CAPACITY = 191_566
USABLE_SEQUENCE_CAPACITY = 191_552
EXCLUDED_TAIL_SEQUENCES = 14
MILESTONES = (SOURCE_STEP, GATE_85_STEP, GATE_90_STEP, GATE_95_STEP, FINAL_STEP)
FULL_RESUME_STEPS = {GATE_85_STEP, GATE_95_STEP, FINAL_STEP}
MODEL_ONLY_STEPS = {GATE_90_STEP}
EXPECTED_SOURCE_MODEL_HASH = "d76b253fe0feced86e0a246c949649e95540f7f4bbeef3ba2491bcc7de3174f0"
EXPECTED_SOURCE_RESUME_HASH = "970f8c848487a383c63fb02e7049e28d22d0a282200c97f6c1f0376dce57efd0"
EXPECTED_ORDER_HASH = "4d31d6cf5532cd1729b35528a649c52b4552d8959a10b5813ce9851bb714ffd1"
BASELINE_VALIDATION = 5.091753994552637
BASELINE_EVAL = 5.048093634427402
PHASE4E_GRADIENT_P95 = 2.03125
MINIMUM_FREE_RESERVE_BYTES = 15_000_000_000


def ensure_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    if "onedrive" in str(resolved).lower():
        raise ValueError(f"Phase 4F mutable runtime cannot use OneDrive: {resolved}")
    try:
        resolved.relative_to(RUNTIME_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Phase 4F runtime escaped its root: {resolved}") from exc
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = ensure_runtime_path(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_name(step: int) -> str:
    return f"step_{step:06d}_tokens_{step * TOKENS_PER_STEP:09d}"


def gate_path(step: int) -> Path:
    return RUN_DIR / "gates" / f"step_{step:06d}.json"


def gate_passed(step: int) -> bool:
    path = gate_path(step)
    return path.is_file() and load_json(path).get("result") == "PASS" and load_json(path).get("continuation_authorized") is True


def validate_authorization(payload: dict[str, Any], *, requested_step: int, requested_tokens: int | None = None) -> None:
    required = {
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "source_checkpoint_resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH,
        "current_optimizer_step": SOURCE_STEP,
        "current_tokens": SOURCE_TOKENS,
        "current_sequence_index": SOURCE_STEP * SEQUENCES_PER_STEP,
        "maximum_optimizer_step": FINAL_STEP,
        "maximum_tokens": FINAL_TOKENS,
        "maximum_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "excluded_tail_sequences": EXCLUDED_TAIL_SEQUENCES,
        "sequences_per_optimizer_step": SEQUENCES_PER_STEP,
        "tokens_per_optimizer_step": TOKENS_PER_STEP,
        "no_wrap_enforced": True,
        "sequence_replacement_authorized": False,
        "partial_effective_batch_authorized": False,
        "second_epoch_authorized": False,
        "scheduler_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
        "data_order_reset": False,
        "sft_authorized": False,
        "qwen_teacher_generation_authorized": False,
        "upload_authorized": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Phase 4F authorization mismatch: {key}")
    if payload.get("tokenizer_hashes") != EXPECTED_HASHES:
        raise ValueError("Phase 4F tokenizer identity changed")
    corpus = payload.get("corpus_hashes", {})
    if corpus.get("corpus") != EXPECTED_CORPUS_HASH or corpus.get("tokenized_manifest_content") != EXPECTED_TOKENIZED_HASH or corpus.get("sequence_order") != EXPECTED_ORDER_HASH:
        raise ValueError("Phase 4F corpus identity changed")
    tokens = requested_step * TOKENS_PER_STEP if requested_tokens is None else int(requested_tokens)
    if requested_step < SOURCE_STEP or requested_step > FINAL_STEP:
        raise PermissionError("Phase 4F step exceeds exact no-wrap authorization")
    if tokens != requested_step * TOKENS_PER_STEP or tokens > FINAL_TOKENS:
        raise PermissionError("Phase 4F token limit or step/token boundary violation")
    if requested_step > GATE_85_STEP and not gate_passed(GATE_85_STEP):
        raise PermissionError("Phase 4F 85M gate has not authorized continuation")
    if requested_step > GATE_90_STEP and not gate_passed(GATE_90_STEP):
        raise PermissionError("Phase 4F 90M gate has not authorized continuation")
    if requested_step > GATE_95_STEP and not gate_passed(GATE_95_STEP):
        raise PermissionError("Phase 4F 95M gate has not authorized continuation")


def load_contract(*, requested_step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTHORIZATION_PATH)
    validate_authorization(authorization, requested_step=requested_step)
    config = load_json(V2_CONFIG)
    if sha256_file(V2_CONFIG) != EXPECTED_V2_FILE_HASH:
        raise ValueError("frozen V2 config file hash changed")
    return authorization, config


def storage_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(RUNTIME_ROOT.anchor)
    sizes = {name: directory_size(Path(fr"C:\DarkMindRuntime\{name}")) for name in ("phase4b", "phase4c", "phase4d", "phase4e")}
    full_resume = directory_size(SOURCE_CHECKPOINT)
    model_only = directory_size(SOURCE_CHECKPOINT / "model")
    export_estimate = model_only + 5_000_000
    audits_reports = 1_500_000_000
    contingency = 1_000_000_000
    projected = full_resume * len(FULL_RESUME_STEPS) + model_only + export_estimate + audits_reports + contingency
    payload = {
        "schema_version": "darkmind-v2-phase4f-storage-preflight-v1",
        "result": "PASS" if usage.free - projected >= MINIMUM_FREE_RESERVE_BYTES else "FAIL",
        "available_bytes": usage.free,
        **{f"{name}_bytes": value for name, value in sizes.items()},
        "phase4f_bytes_before_run": directory_size(RUNTIME_ROOT),
        "expected_full_resume_checkpoint_bytes": full_resume,
        "expected_model_only_checkpoint_bytes": model_only,
        "expected_audit_report_bytes": audits_reports,
        "expected_export_bytes": export_estimate,
        "contingency_bytes": contingency,
        "projected_phase4f_bytes": projected,
        "projected_free_after_completion_bytes": usage.free - projected,
        "required_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES,
        "immutable_input_bytes_duplicated": 0,
        "immutable_inputs_reused_read_only": str(TOKENIZED_INPUT),
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "storage_preflight.json", payload)
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4F disk reserve gate failed")
    return payload


def probe_manifest() -> dict[str, Any]:
    source = Path(r"C:\DarkMindRuntime\phase4e\manifests\fixed_probes.json")
    payload = load_json(source)
    core = {key: value for key, value in payload.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != payload.get("deterministic_content_hash"):
        raise ValueError("fixed probe manifest changed")
    target = RUNTIME_ROOT / "manifests" / "fixed_probes.json"
    if target.is_file() and load_json(target) != payload:
        raise ValueError("Phase 4F fixed probe copy changed")
    if not target.is_file():
        atomic_write_json(target, payload)
    return payload


def immutable_preflight() -> dict[str, Any]:
    authorization, config = load_contract(requested_step=GATE_85_STEP)
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Base V1 model config changed")
    verify_frozen_tokenizer(TOKENIZER_INPUT)
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != EXPECTED_ORDER_HASH or len(order_data.order) != UNIQUE_SEQUENCE_CAPACITY:
        raise ValueError("deterministic sequence order changed")
    corpus = _validate_shards()
    metadata = load_json(SOURCE_CHECKPOINT / "checkpoint_metadata.json")
    state = metadata["training_state"]
    if (state["step"], state["tokens_seen"], state["data_position"]) != (SOURCE_STEP, SOURCE_TOKENS, SOURCE_TOKENS):
        raise ValueError("Phase 4E source state changed")
    model_hash = sha256_file(SOURCE_CHECKPOINT / "model" / "model.safetensors")
    resume_hash = sha256_file(SOURCE_CHECKPOINT / "resume_state.pt")
    if model_hash != EXPECTED_SOURCE_MODEL_HASH or resume_hash != EXPECTED_SOURCE_RESUME_HASH:
        raise ValueError("Phase 4E source checkpoint hash changed")
    resume = torch.load(SOURCE_CHECKPOINT / "resume_state.pt", map_location="cpu", weights_only=False)
    if not resume.get("optimizer") or resume["scheduler"].get("last_epoch") != SOURCE_STEP:
        raise ValueError("Phase 4E optimizer/scheduler state changed")
    expected_next_lr = learning_rate_for_policy(SOURCE_STEP + 1, config["schedule"])
    if not math.isclose(float(resume["optimizer"]["param_groups"][0]["lr"]), expected_next_lr, abs_tol=1e-15):
        raise ValueError("Phase 4E next LR changed")
    payload = {
        "schema_version": "darkmind-v2-phase4f-immutable-resume-preflight-v1",
        "result": "PASS",
        "authorization_file_sha256": sha256_file(AUTHORIZATION_PATH),
        "v2_config_file_sha256": sha256_file(V2_CONFIG),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_metadata_sha256": sha256_file(SOURCE_CHECKPOINT / "checkpoint_metadata.json"),
        "model_weight_sha256": model_hash,
        "resume_state_file_sha256": resume_hash,
        "optimizer_state_sha256": state_hash(resume["optimizer"]),
        "scheduler_state_sha256": state_hash(resume["scheduler"]),
        "rng_state_sha256": rng_fingerprint(resume["rng"]),
        "training_state": state,
        "applied_learning_rate_at_source": learning_rate_for_policy(SOURCE_STEP, config["schedule"]),
        "next_learning_rate": expected_next_lr,
        "model_config_sha256": sha256_file(MODEL_INPUT),
        "architecture_hash": metadata["model_config_hash"],
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "corpus": corpus,
        "sequence_order_hash": order_data.order_hash,
        "unique_sequence_capacity": len(order_data.order),
        "maximum_complete_step": FINAL_STEP,
        "maximum_complete_tokens": FINAL_TOKENS,
        "maximum_complete_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "excluded_tail_sequences": EXCLUDED_TAIL_SEQUENCES,
        "source_checkpoint_modified": False,
        "second_epoch_authorized": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json", payload)
    del resume
    return payload


def disposable_resume_test(steps: int = 2) -> dict[str, Any]:
    if steps < 1 or steps > 4:
        raise ValueError("disposable diagnostic must use 1-4 steps")
    _, config = load_contract(requested_step=SOURCE_STEP)
    preflight = load_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json")
    if preflight.get("result") != "PASS":
        raise RuntimeError("immutable preflight must pass first")
    copy_dir = ensure_runtime_path(RUNTIME_ROOT / "temporary" / "disposable_step9155_checkpoint")
    if copy_dir.exists():
        raise FileExistsError(f"inspect stale disposable checkpoint: {copy_dir}")
    shutil.copytree(SOURCE_CHECKPOINT, copy_dir)
    for relative in ("checkpoint_metadata.json", "model/model.safetensors", "resume_state.pt"):
        if sha256_file(copy_dir / relative) != sha256_file(SOURCE_CHECKPOINT / relative):
            raise ValueError(f"disposable copy mismatch: {relative}")
    model, optimizer, scheduler, device = _create_model_stack(config)
    expected_rng = rng_fingerprint(torch.load(copy_dir / "resume_state.pt", map_location="cpu", weights_only=False)["rng"])
    state = load_checkpoint(copy_dir, model=model, optimizer=optimizer, scheduler=scheduler, expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"], expected_data_manifest_hash=tokenized_manifest_hash(TOKENIZED_INPUT))
    actual_rng = rng_fingerprint(capture_rng_state())
    validation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "validation"), sequence_length=512, micro_batch_size=2, device=device)
    evaluation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "eval"), sequence_length=512, micro_batch_size=2, device=device)
    if not math.isclose(validation["loss"], BASELINE_VALIDATION, abs_tol=1e-9) or not math.isclose(evaluation["loss"], BASELINE_EVAL, abs_tol=1e-9):
        raise ValueError("Phase 4F disposable loss reproduction failed")
    dataset = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    records = []
    for _ in range(steps):
        step = state.step + 1
        lr = optimizer.param_groups[0]["lr"]
        metric = training_step(model, optimizer, dataset, data_position=state.data_position, diagnostic=False, device=device)
        scheduler.step()
        state.step = step
        state.tokens_seen += TOKENS_PER_STEP
        state.data_position += TOKENS_PER_STEP
        records.append({"step": step, "loss": metric["raw_train_loss"], "learning_rate": lr, "sequence_index": state.data_position // 512})
    checks = {
        "validation_reproduced": True,
        "eval_reproduced": True,
        "optimizer_loaded": bool(optimizer.state),
        "rng_continuity": expected_rng == actual_rng,
        "scheduler_continuity": scheduler.last_epoch == SOURCE_STEP + steps,
        "data_position_continuity": state.data_position == SOURCE_TOKENS + steps * TOKENS_PER_STEP,
        "next_sequence_exact": state.data_position // 512 == SOURCE_STEP * SEQUENCES_PER_STEP + steps * SEQUENCES_PER_STEP,
        "source_model_unchanged": sha256_file(SOURCE_CHECKPOINT / "model" / "model.safetensors") == EXPECTED_SOURCE_MODEL_HASH,
        "source_resume_unchanged": sha256_file(SOURCE_CHECKPOINT / "resume_state.pt") == EXPECTED_SOURCE_RESUME_HASH,
    }
    payload = {
        "schema_version": "darkmind-v2-phase4f-disposable-resume-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "process_id": os.getpid(),
        "diagnostic_steps": steps,
        "validation": validation,
        "eval": evaluation,
        "records": records,
        "checks": checks,
        "copied_state_discarded": True,
        "official_checkpoint_modified": False,
    }
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    shutil.rmtree(copy_dir)
    atomic_write_json(RUNTIME_ROOT / "manifests" / "disposable_resume_test.json", payload)
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4F disposable resume failed")
    return payload


def prepare() -> dict[str, Any]:
    authorization, _ = load_contract(requested_step=GATE_85_STEP)
    storage = storage_preflight()
    if load_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json").get("result") != "PASS" or load_json(RUNTIME_ROOT / "manifests" / "disposable_resume_test.json").get("result") != "PASS":
        raise RuntimeError("Phase 4F preflights must pass")
    if RUN_DIR.exists() and (RUN_DIR / "run_manifest.json").is_file():
        existing = load_json(RUN_DIR / "run_manifest.json")
        if existing.get("authorization_file_sha256") != sha256_file(AUTHORIZATION_PATH):
            raise FileExistsError("existing Phase 4F run uses another authorization")
        return existing
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"incomplete Phase 4F run requires inspection: {RUN_DIR}")
    for directory in ("checkpoints", "diagnostics", "resume", "audits", "gates"):
        (RUN_DIR / directory).mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics.jsonl").write_text("", encoding="utf-8")
    probes = probe_manifest()
    source_evaluation = load_json(PHASE4E_RUN / "evaluations.json")[str(SOURCE_STEP)]
    atomic_write_json(RUN_DIR / "evaluations.json", {str(SOURCE_STEP): source_evaluation})
    manifest = {
        "schema_version": "darkmind-v2-phase4f-first-pass-run-v1",
        "status": "prepared",
        "authorization": authorization,
        "authorization_file_sha256": sha256_file(AUTHORIZATION_PATH),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "source_checkpoint_resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH,
        "checkpoints": {str(SOURCE_STEP): str(SOURCE_CHECKPOINT)},
        "checkpoint_hashes": {str(SOURCE_STEP): {"model_sha256": EXPECTED_SOURCE_MODEL_HASH, "resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH}},
        "checkpoint_policy": {"10375": "full_resume", "10986": "model_only", "11596": "full_resume", "11972": "full_resume"},
        "diagnostic_snapshots": {str(SOURCE_STEP): source_evaluation["diagnostics"]},
        "latest_resume_checkpoint": str(SOURCE_CHECKPOINT),
        "best_validation_step": SOURCE_STEP,
        "best_validation_checkpoint": str(SOURCE_CHECKPOINT),
        "sequence_order_hash": EXPECTED_ORDER_HASH,
        "data_manifest_hash": authorization["corpus_hashes"]["tokenized_manifest_file"],
        "fixed_probe_hash": probes["deterministic_content_hash"],
        "process_ids": [],
        "segments": [],
        "gate_status": {"10375": "AUTHORIZED", "10986": "CONDITIONAL", "11596": "CONDITIONAL", "11972": "CONDITIONAL"},
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "maximum_complete_step": FINAL_STEP,
        "maximum_complete_tokens": FINAL_TOKENS,
        "maximum_complete_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "excluded_tail_sequences": EXCLUDED_TAIL_SEQUENCES,
        "no_wrap": True,
        "second_epoch_authorized": False,
        "runtime_outside_onedrive": True,
        "immutable_input_bytes_duplicated": 0,
        "storage_preflight": storage,
    }
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": "prepared", "optimizer_step": SOURCE_STEP, "tokens_consumed": SOURCE_TOKENS, "sequence_index": SOURCE_STEP * SEQUENCES_PER_STEP})
    return manifest


def rename_checkpoint(temporary: Path, checkpoint: Path) -> None:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4F checkpoint: {checkpoint}")
    for attempt in range(20):
        try:
            os.replace(temporary, checkpoint)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def save_milestone(*, step: int, model: torch.nn.Module, optimizer: torch.optim.AdamW, scheduler: Any, state: TrainingState, data_hash: str, authorization_hash: str) -> tuple[Path, dict[str, Any]]:
    if step not in FULL_RESUME_STEPS | MODEL_ONLY_STEPS:
        raise ValueError(f"unsupported Phase 4F milestone: {step}")
    checkpoint = ensure_runtime_path(RUN_DIR / "checkpoints" / checkpoint_name(step))
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(f"Phase 4F milestone exists: {checkpoint}")
    if step in FULL_RESUME_STEPS:
        metadata = save_checkpoint(temporary, model=model, optimizer=optimizer, scheduler=scheduler, training_state=state, tokenizer_hash=EXPECTED_HASHES["tokenizer.model"], data_manifest_hash=data_hash)
        kind = "full_resume"
    else:
        temporary.mkdir(parents=True)
        metadata = {"model_files": save_model_package(model, temporary / "model"), "training_state": state.to_dict()}
        kind = "model_only"
    stage_metadata = {
        "schema_version": "darkmind-v2-phase4f-first-pass-checkpoint-v1",
        "result": "PASS",
        "checkpoint_kind": kind,
        "optimizer_step": step,
        "consumed_tokens": state.tokens_seen,
        "next_sequence_index": state.data_position // 512,
        "model_sha256": metadata["model_files"]["model_sha256"],
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_hashes": EXPECTED_HASHES,
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "data_manifest_file_sha256": data_hash,
        "sequence_order_hash": EXPECTED_ORDER_HASH,
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "authorization_file_sha256": authorization_hash,
        "resume_capable": kind == "full_resume",
        "scheduler_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
        "data_order_reset": False,
        "no_wrap": True,
        "second_epoch_authorized": False,
        "next_learning_rate": optimizer.param_groups[0]["lr"],
    }
    atomic_write_json(temporary / "phase4f_checkpoint_metadata.json", stage_metadata)
    _validate_safetensors(temporary / "model" / "model.safetensors")
    rename_checkpoint(temporary, checkpoint)
    hashes = {"model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"), "phase4f_metadata_sha256": sha256_file(checkpoint / "phase4f_checkpoint_metadata.json")}
    if kind == "full_resume":
        hashes["resume_state_sha256"] = sha256_file(checkpoint / "resume_state.pt")
        hashes["checkpoint_metadata_sha256"] = sha256_file(checkpoint / "checkpoint_metadata.json")
    return checkpoint, {**stage_metadata, "hashes": hashes}


def assert_resume_state(state: TrainingState, expected_step: int) -> None:
    expected_tokens = expected_step * TOKENS_PER_STEP
    if (state.step, state.tokens_seen, state.data_position) != (expected_step, expected_tokens, expected_tokens):
        raise ValueError("Phase 4F resume step/token/data-position mismatch")


def improvement(start: float, end: float) -> float:
    return (start - end) * 100.0 / start


def evaluate_gate(step: int, *, generation_required: bool = False) -> dict[str, Any]:
    if step not in (GATE_85_STEP, GATE_90_STEP, GATE_95_STEP):
        raise ValueError("Phase 4F gate must be 10375, 10986, or 11596")
    baselines = {GATE_85_STEP: SOURCE_STEP, GATE_90_STEP: GATE_85_STEP, GATE_95_STEP: GATE_90_STEP}
    baseline_step = baselines[step]
    evaluations = load_json(RUN_DIR / "evaluations.json")
    start = evaluations[str(baseline_step)]
    final = evaluations[str(step)]
    val_change = improvement(float(start["validation"]["loss"]), float(final["validation"]["loss"]))
    eval_change = improvement(float(start["eval"]["loss"]), float(final["eval"]["loss"]))
    considered = [value for value in MILESTONES if baseline_step <= value <= step and str(value) in evaluations]
    val_losses = [float(evaluations[str(value)]["validation"]["loss"]) for value in considered]
    eval_losses = [float(evaluations[str(value)]["eval"]["loss"]) for value in considered]
    val_rebound = rebound_percent(min(val_losses), val_losses[-1])
    eval_rebound = rebound_percent(min(eval_losses), eval_losses[-1])
    catastrophic = {name: float(final["probes"][name]["loss"]) > float(start["probes"][name]["loss"]) * 1.20 for name in start["probes"] if name in final["probes"]}
    manifest = load_json(RUN_DIR / "run_manifest.json")
    integrity = all(segment["result"] == "PASS" and segment["rng_continuity"] and segment["scheduler_continuity"] and segment["data_position_continuity"] and segment["no_repeated_or_skipped_sequence"] and segment["no_data_wrap"] for segment in manifest["segments"])
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    non_finite = sum(int(item["non_finite_event_count"]) for item in metrics)
    hard_failures = 0
    if generation_required:
        audit = RUN_DIR / "audits" / f"step_{step:06d}" / "subset" / "audit_summary.json"
        if not audit.is_file():
            raise RuntimeError("Phase 4F generation subset must run before this gate")
        generation = load_json(audit)
        hard_failures = int(generation["greedy"]["hard_failure_total"]) + int(generation["sampling"]["hard_failure_total"])
    failed = val_change < -0.5 or eval_change < -0.5 or val_rebound > 1.0 or eval_rebound > 1.0 or any(catastrophic.values()) or not integrity or non_finite or hard_failures
    result = "FAIL" if failed else "PASS"
    payload = {
        "schema_version": f"darkmind-v2-phase4f-{step}-gate-v1",
        "result": result,
        "continuation_authorized": result == "PASS",
        "baseline_step": baseline_step,
        "gate_step": step,
        "validation_improvement_percent": val_change,
        "eval_improvement_percent": eval_change,
        "validation_rebound_percent": val_rebound,
        "eval_rebound_percent": eval_rebound,
        "catastrophic_probe_regressions": catastrophic,
        "integrity_pass": integrity,
        "non_finite_events": non_finite,
        "generation_hard_failures": hard_failures,
        "flat_within_half_percent_is_acceptable": True,
        "second_epoch_authorized": False,
    }
    atomic_write_json(gate_path(step), payload)
    manifest["gate_status"][str(step)] = result
    next_step = {GATE_85_STEP: GATE_90_STEP, GATE_90_STEP: GATE_95_STEP, GATE_95_STEP: FINAL_STEP}[step]
    if result == "PASS":
        manifest["gate_status"][str(next_step)] = "AUTHORIZED"
        manifest["status"] = f"gate_{step}_passed"
    else:
        manifest["status"] = f"stopped_at_{step}_fail"
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": manifest["status"], "optimizer_step": step, "tokens_consumed": step * TOKENS_PER_STEP, "sequence_index": step * SEQUENCES_PER_STEP})
    return payload


def expected_segment(target_step: int, manifest: dict[str, Any]) -> int:
    expected = {GATE_85_STEP: SOURCE_STEP, GATE_95_STEP: GATE_85_STEP, FINAL_STEP: GATE_95_STEP}
    required = {GATE_85_STEP: 0, GATE_95_STEP: 1, FINAL_STEP: 2}
    if target_step not in expected or len(manifest["segments"]) != required[target_step]:
        raise ValueError("Phase 4F segment order changed")
    return expected[target_step]


def train_segment(target_step: int) -> dict[str, Any]:
    # Segment B establishes its 90M gate in-process, so its initial contract
    # check must stop at that boundary. Every subsequent step is revalidated.
    initial_contract_step = {
        GATE_85_STEP: GATE_85_STEP,
        GATE_95_STEP: GATE_90_STEP,
        FINAL_STEP: FINAL_STEP,
    }[target_step]
    authorization, config = load_contract(requested_step=initial_contract_step)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    expected_start = expected_segment(target_step, manifest)
    checkpoint = Path(manifest["latest_resume_checkpoint"])
    resume_payload = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_rng = rng_fingerprint(resume_payload["rng"])
    model, optimizer, scheduler, device = _create_model_stack(config)
    state = load_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"], expected_data_manifest_hash=manifest["data_manifest_hash"])
    assert_resume_state(state, expected_start)
    actual_rng = rng_fingerprint(capture_rng_state())
    if actual_rng != expected_rng or scheduler.last_epoch != expected_start:
        raise ValueError("Phase 4F RNG/scheduler continuity failed")
    if not math.isclose(optimizer.param_groups[0]["lr"], learning_rate_for_policy(expected_start + 1, config["schedule"]), abs_tol=1e-15):
        raise ValueError("Phase 4F next LR mismatch")
    if os.getpid() in manifest["process_ids"]:
        raise ValueError("Phase 4F segment did not use a fresh process")
    if expected_start != SOURCE_STEP:
        val = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "validation"), sequence_length=512, micro_batch_size=2, device=device)
        evl = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "eval"), sequence_length=512, micro_batch_size=2, device=device)
        prior = load_json(RUN_DIR / "evaluations.json")[str(expected_start)]
        if not math.isclose(val["loss"], prior["validation"]["loss"], abs_tol=1e-9) or not math.isclose(evl["loss"], prior["eval"]["loss"], abs_tol=1e-9):
            raise ValueError("Phase 4F restart loss reproduction failed")
    manifest["process_ids"].append(os.getpid())
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != EXPECTED_ORDER_HASH or len(order_data.order) != UNIQUE_SEQUENCE_CAPACITY or target_step * SEQUENCES_PER_STEP > USABLE_SEQUENCE_CAPACITY:
        raise PermissionError("Phase 4F target would wrap or use the excluded tail")
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_data = TokenShardDataset(TOKENIZED_INPUT, "train")
    probes = probe_manifest()
    activation_index = int(probes["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_data.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    evaluations = load_json(RUN_DIR / "evaluations.json")
    prior_metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gradients = [float(item["pre_clip_gradient_norm"]) for item in prior_metrics]
    coefficients = [float(item["clipping_coefficient"]) for item in prior_metrics]
    updates = [float(item["update_to_weight"]["maximum"]) for item in prior_metrics]
    clipped_steps = sum(int(item["clipped"]) for item in prior_metrics)
    high_update_run = 0
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    start_position = state.data_position
    expected_cursor = state.data_position // 512
    try:
        while state.step < target_step:
            step = state.step + 1
            validate_authorization(authorization, requested_step=step)
            lr = optimizer.param_groups[0]["lr"]
            diagnostic = step in MILESTONES
            metric = training_step(model, optimizer, order_data, data_position=state.data_position, diagnostic=diagnostic, device=device)
            expected_indices = order_data.order[expected_cursor : expected_cursor + SEQUENCES_PER_STEP]
            if len(expected_indices) != SEQUENCES_PER_STEP or metric["source_sequence_indices"] != expected_indices:
                raise ValueError("Phase 4F repeated, skipped, or partial sequence batch")
            expected_cursor += SEQUENCES_PER_STEP
            scheduler.step()
            state.step = step
            state.tokens_seen += TOKENS_PER_STEP
            state.data_position += TOKENS_PER_STEP
            assert_resume_state(state, step)
            pre_clip = float(metric.pop("gradient_norm"))
            coefficient = min(1.0, 1.0 / (pre_clip + 1e-6))
            post_clip = pre_clip * coefficient
            gradients.append(pre_clip)
            coefficients.append(coefficient)
            clipped_steps += int(metric["clipped"])
            update_max = float(metric["update_to_weight"]["maximum"])
            updates.append(update_max)
            high_update_run = high_update_run + 1 if update_max > 0.005 else 0
            if high_update_run >= 3:
                raise FloatingPointError("Phase 4F repeated update ratio above 0.005")
            state.last_training_loss = metric["raw_train_loss"]
            state.smoothed_training_loss = metric["raw_train_loss"] if state.smoothed_training_loss is None else 0.9 * state.smoothed_training_loss + 0.1 * metric["raw_train_loss"]
            parameter_diagnostics = metric.pop("parameter_diagnostics")
            metric.update({
                "optimizer_step": step,
                "tokens_consumed": state.tokens_seen,
                "sequence_index": state.data_position // 512,
                "pre_clip_gradient_norm": pre_clip,
                "clipping_coefficient": coefficient,
                "post_clip_gradient_norm": post_clip,
                "learning_rate_applied": lr,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "smoothed_train_loss": state.smoothed_training_loss,
                "clipped_step_fraction": clipped_steps / max(step - SOURCE_STEP, 1),
                "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                "gpu": monitor.latest(),
                "non_finite_event_count": 0,
            })
            if diagnostic:
                validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                evaluation = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                probe_losses = {name: evaluate_probe(model, probe, device) for name, probe in probes["probes"].items()}
                diagnostics = _milestone_diagnostics(model, step=step, parameter_diagnostics=parameter_diagnostics, clipped_fraction=clipped_steps / max(step - SOURCE_STEP, 1), probe_losses=probe_losses, activation_tokens=activation_tokens)
                diagnostics["training_health"] = {
                    "gradient_p50": percentile(gradients, 0.50),
                    "gradient_p95": percentile(gradients, 0.95),
                    "gradient_max": max(gradients),
                    "clipping_coefficient_p50": percentile(coefficients, 0.50),
                    "clipping_coefficient_p95": percentile(coefficients, 0.95),
                    "clipping_coefficient_min": min(coefficients),
                    "clipped_step_fraction": clipped_steps / max(step - SOURCE_STEP, 1),
                    "update_to_weight_p50": percentile(updates, 0.50),
                    "update_to_weight_p95": percentile(updates, 0.95),
                    "update_to_weight_max": max(updates),
                    "phase4e_gradient_p95": PHASE4E_GRADIENT_P95,
                    "non_finite_events": 0,
                }
                if diagnostics["training_health"]["gradient_p95"] > 2.0 * PHASE4E_GRADIENT_P95:
                    raise FloatingPointError("Phase 4F gradient p95 doubled")
                diagnostic_path = RUN_DIR / "diagnostics" / f"step_{step:06d}.json"
                atomic_write_json(diagnostic_path, diagnostics)
                state.last_validation_loss = validation["loss"]
                prospective = RUN_DIR / "checkpoints" / checkpoint_name(step)
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = validation["loss"]
                    state.best_checkpoint = str(prospective)
                    manifest["best_validation_step"] = step
                    manifest["best_validation_checkpoint"] = str(prospective)
                saved, checkpoint_metadata = save_milestone(step=step, model=model, optimizer=optimizer, scheduler=scheduler, state=state, data_hash=manifest["data_manifest_hash"], authorization_hash=manifest["authorization_file_sha256"])
                evaluations[str(step)] = {"train": {"loss": metric["raw_train_loss"], "smoothed_loss": state.smoothed_training_loss}, "validation": validation, "eval": evaluation, "probes": probe_losses, "diagnostics": str(diagnostic_path)}
                manifest["checkpoints"][str(step)] = str(saved)
                manifest["checkpoint_hashes"][str(step)] = checkpoint_metadata["hashes"]
                manifest["diagnostic_snapshots"][str(step)] = str(diagnostic_path)
                if step in FULL_RESUME_STEPS:
                    manifest["latest_resume_checkpoint"] = str(saved)
                metric.update({"validation_loss": validation["loss"], "eval_loss": evaluation["loss"], "checkpoint": str(saved)})
                atomic_write_json(RUN_DIR / "evaluations.json", evaluations)
                atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
                if step == GATE_90_STEP:
                    append_jsonl(RUN_DIR / "metrics.jsonl", metric)
                    atomic_write_json(RUN_DIR / "progress.json", {"status": "evaluating_90m_gate", "optimizer_step": step, "tokens_consumed": state.tokens_seen, "sequence_index": state.data_position // 512})
                    gate = evaluate_gate(GATE_90_STEP)
                    if gate["result"] != "PASS":
                        raise RuntimeError("Phase 4F 90M gate stopped Segment B")
                    continue
            append_jsonl(RUN_DIR / "metrics.jsonl", metric)
            atomic_write_json(RUN_DIR / "progress.json", {"status": "training", "optimizer_step": step, "tokens_consumed": state.tokens_seen, "sequence_index": state.data_position // 512})
            if step % 10 == 0 or diagnostic:
                print(f"phase4f step={step} loss={metric['raw_train_loss']:.6f} lr={lr:.9f} tok_s={metric['active_tokens_per_second']:.1f}", flush=True)
    finally:
        monitor.close()
    if state.step != target_step or state.tokens_seen != target_step * TOKENS_PER_STEP:
        raise RuntimeError("Phase 4F segment missed exact stop")
    segment = {
        "schema_version": "darkmind-v2-phase4f-segment-v1",
        "result": "PASS",
        "process_id": os.getpid(),
        "segment_start_step": expected_start + 1,
        "segment_end_step": target_step,
        "optimizer_steps_added": target_step - expected_start,
        "segment_token_range": [start_position, state.data_position],
        "tokens_added": state.data_position - start_position,
        "sequences_added": (state.data_position - start_position) // 512,
        "rng_continuity": expected_rng == actual_rng,
        "scheduler_continuity": scheduler.last_epoch == target_step,
        "data_position_continuity": expected_cursor == state.data_position // 512,
        "optimizer_continuity": True,
        "no_repeated_or_skipped_sequence": True,
        "no_data_wrap": state.data_position // 512 <= USABLE_SEQUENCE_CAPACITY,
        "partial_effective_batch": False,
        "next_optimizer_step": target_step + 1 if target_step < FINAL_STEP else None,
        "next_sequence_index": state.data_position // 512,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_samples": monitor.samples,
        "latest_resume_checkpoint": manifest["latest_resume_checkpoint"],
    }
    manifest["segments"].append(segment)
    manifest["status"] = {GATE_85_STEP: "awaiting_85m_gate", GATE_95_STEP: "awaiting_95m_gate", FINAL_STEP: "first_corpus_pass_training_complete"}[target_step]
    atomic_write_json(RUN_DIR / "resume" / f"segment_to_step_{target_step:06d}.json", segment)
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": manifest["status"], "optimizer_step": target_step, "tokens_consumed": state.tokens_seen, "sequence_index": state.data_position // 512})
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return segment


def validate_final_resume() -> dict[str, Any]:
    _, config = load_contract(requested_step=FINAL_STEP)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    checkpoint = Path(manifest["latest_resume_checkpoint"])
    if checkpoint != RUN_DIR / "checkpoints" / checkpoint_name(FINAL_STEP):
        raise ValueError("Phase 4F final checkpoint changed")
    resume = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_rng = rng_fingerprint(resume["rng"])
    model, optimizer, scheduler, device = _create_model_stack(config)
    state = load_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"], expected_data_manifest_hash=manifest["data_manifest_hash"])
    assert_resume_state(state, FINAL_STEP)
    checks = {
        "model_hash": sha256_file(checkpoint / "model" / "model.safetensors") == manifest["checkpoint_hashes"][str(FINAL_STEP)]["model_sha256"],
        "optimizer_loaded": bool(optimizer.state),
        "scheduler_epoch": scheduler.last_epoch == FINAL_STEP,
        "rng_continuity": rng_fingerprint(capture_rng_state()) == expected_rng,
        "tokens": state.tokens_seen == FINAL_TOKENS,
        "sequence_index": state.data_position // 512 == USABLE_SEQUENCE_CAPACITY,
        "excluded_tail": UNIQUE_SEQUENCE_CAPACITY - state.data_position // 512 == EXCLUDED_TAIL_SEQUENCES,
        "step_11973_not_run": state.step == FINAL_STEP,
        "second_epoch_not_started": manifest["second_epoch_authorized"] is False,
    }
    payload = {
        "schema_version": "darkmind-v2-phase4f-final-resume-validation-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "resume_state_sha256": sha256_file(checkpoint / "resume_state.pt"),
        "optimizer_step": state.step,
        "tokens_consumed": state.tokens_seen,
        "sequence_index": state.data_position // 512,
        "excluded_tail_sequences": EXCLUDED_TAIL_SEQUENCES,
        "checks": checks,
    }
    atomic_write_json(RUN_DIR / "final_resume_validation.json", payload)
    del model, optimizer, scheduler, resume
    torch.cuda.empty_cache()
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4F final resume validation failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("storage")
    commands.add_parser("preflight")
    disposable = commands.add_parser("disposable-resume")
    disposable.add_argument("--steps", type=int, default=2)
    commands.add_parser("prepare")
    segment = commands.add_parser("segment")
    segment.add_argument("--target-step", type=int, choices=(GATE_85_STEP, GATE_95_STEP, FINAL_STEP), required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--step", type=int, choices=(GATE_85_STEP, GATE_90_STEP, GATE_95_STEP), required=True)
    gate.add_argument("--generation-required", action="store_true")
    commands.add_parser("validate-final-resume")
    args = parser.parse_args()
    if args.command == "storage":
        payload = storage_preflight()
    elif args.command == "preflight":
        payload = immutable_preflight()
    elif args.command == "disposable-resume":
        payload = disposable_resume_test(args.steps)
    elif args.command == "prepare":
        payload = prepare()
    elif args.command == "segment":
        payload = train_segment(args.target_step)
    elif args.command == "gate":
        payload = evaluate_gate(args.step, generation_required=args.generation_required)
    else:
        payload = validate_final_resume()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
