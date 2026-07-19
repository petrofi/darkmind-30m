"""Continue Base V1 from the exact 25M checkpoint through guarded no-wrap gates."""

from __future__ import annotations

import argparse
import hashlib
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
from darkmind_v2.training.phase4c_diagnostics import (
    MODEL_INPUT,
    ORDER_INPUT,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    learning_rate_for_policy,
)
from darkmind_v2.training.phase4c_training import _max_consecutive_worsening, _milestone_diagnostics, evaluate_probe
from darkmind_v2.training.phase4d_stage2 import (
    EXPECTED_V2_FILE_HASH,
    ROOT,
    V2_CONFIG,
    _create_model_stack,
    _parameter_diagnostics,
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


RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase4e")
RUN_DIR = RUNTIME_ROOT / "runs" / "base_v1_stage3_first_corpus_pass_v2"
PHASE4D_RUN = Path(r"C:\DarkMindRuntime\phase4d\runs\base_v1_stage2_25m_v2_retry1")
SOURCE_CHECKPOINT = PHASE4D_RUN / "checkpoints" / "step_003051_tokens_024993792"
AUTHORIZATION_PATH = ROOT / "darkmind_v2" / "config" / "train_base_v1_stage3_100m_authorization.json"
SOURCE_STEP = 3_051
SOURCE_TOKENS = 24_993_792
GATE_50_STEP = 6_103
GATE_75_STEP = 9_155
FINAL_STEP = 11_972
HARD_STEP_LIMIT = 12_207
HARD_TOKEN_LIMIT = 99_999_744
TOKENS_PER_STEP = 8_192
SEQUENCES_PER_STEP = 16
UNIQUE_SEQUENCE_CAPACITY = 191_566
USABLE_SEQUENCE_CAPACITY = 191_552
UNUSABLE_TAIL_SEQUENCES = 14
FINAL_TOKENS = FINAL_STEP * TOKENS_PER_STEP
MILESTONES = (3_051, 4_096, 5_120, 6_103, 7_168, 8_192, 9_155, 10_240, 11_264, 11_972)
MODEL_ONLY_STEPS = {4_096, 5_120, 7_168, 8_192, 10_240, 11_264}
FULL_RESUME_STEPS = {GATE_50_STEP, GATE_75_STEP, FINAL_STEP}
EXPECTED_SOURCE_MODEL_HASH = "1f27c5096b8f38a5b98e7323a3051ab29365f1b5217e978dc2042c0503c6e48a"
EXPECTED_SOURCE_RESUME_HASH = "f868ebf51ec9f42abcf8e9f90f6cbb8b7707bc893d452608fb04698e2488ac01"
EXPECTED_ORDER_HASH = "4d31d6cf5532cd1729b35528a649c52b4552d8959a10b5813ce9851bb714ffd1"
BASELINE_VALIDATION = 5.526495547844112
BASELINE_EVAL = 5.47353916148324
PHASE4D_GRADIENT_P95 = 1.5703125
MINIMUM_FREE_RESERVE_BYTES = 15_000_000_000


def ensure_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    if "onedrive" in str(resolved).lower():
        raise ValueError(f"Phase 4E mutable runtime cannot use OneDrive: {resolved}")
    try:
        resolved.relative_to(RUNTIME_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Phase 4E runtime escaped its root: {resolved}") from exc
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


def _gate_path(step: int) -> Path:
    return RUN_DIR / "gates" / f"step_{step:06d}.json"


def _passed_gate(step: int) -> bool:
    path = _gate_path(step)
    if not path.is_file():
        return False
    payload = load_json(path)
    return payload.get("result") == "PASS" and payload.get("continuation_authorized") is True


def validate_authorization(payload: dict[str, Any], *, requested_step: int) -> None:
    required = {
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "source_checkpoint_resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH,
        "current_optimizer_step": SOURCE_STEP,
        "current_tokens": SOURCE_TOKENS,
        "current_sequence_index": SOURCE_TOKENS // 512,
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "sequences_per_optimizer_step": SEQUENCES_PER_STEP,
        "maximum_no_wrap_optimizer_step": FINAL_STEP,
        "maximum_no_wrap_tokens": FINAL_TOKENS,
        "maximum_no_wrap_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "unusable_tail_sequences_under_frozen_batch_policy": UNUSABLE_TAIL_SEQUENCES,
        "hard_scheduler_optimizer_step_limit": HARD_STEP_LIMIT,
        "hard_scheduler_token_limit": HARD_TOKEN_LIMIT,
        "no_wrap_enforced": True,
        "sequence_repetition_authorized": False,
        "partial_effective_batch_authorized": False,
        "second_epoch_authorized": False,
        "sft_authorized": False,
        "qwen_teacher_generation_authorized": False,
        "upload_authorized": False,
        "scheduler_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
        "data_order_reset": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Phase 4E authorization mismatch: {key}")
    if payload.get("tokenizer_hashes") != EXPECTED_HASHES:
        raise ValueError("Phase 4E tokenizer identity changed")
    corpus = payload.get("corpus_hashes", {})
    if corpus.get("corpus") != EXPECTED_CORPUS_HASH or corpus.get("tokenized_manifest_content") != EXPECTED_TOKENIZED_HASH:
        raise ValueError("Phase 4E corpus identity changed")
    if corpus.get("sequence_order") != EXPECTED_ORDER_HASH:
        raise ValueError("Phase 4E sequence order changed")
    if payload.get("gate_50m", {}).get("optimizer_step") != GATE_50_STEP:
        raise ValueError("Phase 4E 50M gate changed")
    if payload.get("gate_75m", {}).get("optimizer_step") != GATE_75_STEP:
        raise ValueError("Phase 4E 75M gate changed")
    if requested_step < SOURCE_STEP or requested_step > FINAL_STEP:
        raise PermissionError("Phase 4E requested step exceeds the no-wrap authorization")
    if requested_step > GATE_50_STEP and not _passed_gate(GATE_50_STEP):
        raise PermissionError("Phase 4E 50M gate has not authorized continuation")
    if requested_step > GATE_75_STEP and not _passed_gate(GATE_75_STEP):
        raise PermissionError("Phase 4E 75M gate has not authorized continuation")
    if requested_step * TOKENS_PER_STEP > HARD_TOKEN_LIMIT:
        raise PermissionError("Phase 4E hard scheduler horizon exceeded")


def load_contract(*, requested_step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTHORIZATION_PATH)
    validate_authorization(authorization, requested_step=requested_step)
    config = load_json(V2_CONFIG)
    if sha256_file(V2_CONFIG) != EXPECTED_V2_FILE_HASH:
        raise ValueError("frozen V2 config file hash changed")
    return authorization, config


def storage_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(RUNTIME_ROOT.anchor)
    phase4b = directory_size(Path(r"C:\DarkMindRuntime\phase4b"))
    phase4c = directory_size(Path(r"C:\DarkMindRuntime\phase4c"))
    phase4d = directory_size(Path(r"C:\DarkMindRuntime\phase4d"))
    full_resume = directory_size(SOURCE_CHECKPOINT)
    model_only = directory_size(SOURCE_CHECKPOINT / "model")
    export_estimate = model_only + 5_000_000
    metrics_audits_reports = 1_500_000_000
    contingency = 1_000_000_000
    projected = model_only * len(MODEL_ONLY_STEPS) + full_resume * len(FULL_RESUME_STEPS) + export_estimate + metrics_audits_reports + contingency
    payload = {
        "schema_version": "darkmind-v2-phase4e-storage-preflight-v1",
        "result": "PASS" if usage.free - projected >= MINIMUM_FREE_RESERVE_BYTES else "FAIL",
        "available_bytes": usage.free,
        "phase4b_bytes": phase4b,
        "phase4c_bytes": phase4c,
        "phase4d_bytes": phase4d,
        "phase4e_bytes_before_run": directory_size(RUNTIME_ROOT),
        "expected_model_only_checkpoint_bytes": model_only,
        "expected_full_resume_checkpoint_bytes": full_resume,
        "expected_local_export_bytes": export_estimate,
        "expected_metrics_audits_reports_bytes": metrics_audits_reports,
        "contingency_bytes": contingency,
        "projected_phase4e_bytes": projected,
        "projected_free_after_completion_bytes": usage.free - projected,
        "required_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES,
        "immutable_input_bytes_duplicated": 0,
        "immutable_inputs_reused_read_only": str(TOKENIZED_INPUT),
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "storage_preflight.json", payload)
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4E disk reserve gate failed")
    return payload


def _probe_manifest() -> dict[str, Any]:
    source = Path(r"C:\DarkMindRuntime\phase4d\manifests\fixed_probes.json")
    payload = load_json(source)
    core = {key: value for key, value in payload.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != payload.get("deterministic_content_hash"):
        raise ValueError("Phase 4D fixed probe manifest changed")
    target = RUNTIME_ROOT / "manifests" / "fixed_probes.json"
    if target.is_file() and load_json(target) != payload:
        raise ValueError("Phase 4E fixed probe manifest changed")
    if not target.is_file():
        atomic_write_json(target, payload)
    return payload


def immutable_preflight() -> dict[str, Any]:
    authorization, config = load_contract(requested_step=GATE_50_STEP)
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Base V1 model config changed")
    verify_frozen_tokenizer(TOKENIZER_INPUT)
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != EXPECTED_ORDER_HASH or len(order_data.order) != UNIQUE_SEQUENCE_CAPACITY:
        raise ValueError("immutable sequence-order capacity changed")
    if len(order_data.order) // SEQUENCES_PER_STEP != FINAL_STEP:
        raise ValueError("maximum complete optimizer-step capacity changed")
    corpus = _validate_shards()
    metadata = load_json(SOURCE_CHECKPOINT / "checkpoint_metadata.json")
    state = metadata["training_state"]
    if (state["step"], state["tokens_seen"], state["data_position"]) != (SOURCE_STEP, SOURCE_TOKENS, SOURCE_TOKENS):
        raise ValueError("Phase 4D source training state changed")
    model_hash = sha256_file(SOURCE_CHECKPOINT / "model" / "model.safetensors")
    resume_hash = sha256_file(SOURCE_CHECKPOINT / "resume_state.pt")
    if model_hash != EXPECTED_SOURCE_MODEL_HASH or resume_hash != EXPECTED_SOURCE_RESUME_HASH:
        raise ValueError("Phase 4D source checkpoint hash changed")
    resume = torch.load(SOURCE_CHECKPOINT / "resume_state.pt", map_location="cpu", weights_only=False)
    if not resume.get("optimizer") or resume["scheduler"].get("last_epoch") != SOURCE_STEP:
        raise ValueError("Phase 4D optimizer or scheduler state changed")
    expected_next_lr = learning_rate_for_policy(SOURCE_STEP + 1, config["schedule"])
    if not math.isclose(float(resume["optimizer"]["param_groups"][0]["lr"]), expected_next_lr, abs_tol=1e-15):
        raise ValueError("Phase 4D next learning rate changed")
    payload = {
        "schema_version": "darkmind-v2-phase4e-immutable-resume-preflight-v1",
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
        "applied_learning_rate_at_source": 0.000090230387,
        "next_learning_rate": expected_next_lr,
        "model_config_sha256": sha256_file(MODEL_INPUT),
        "architecture_hash": metadata["model_config_hash"],
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "corpus": corpus,
        "sequence_order_hash": order_data.order_hash,
        "unique_sequence_capacity": len(order_data.order),
        "maximum_complete_optimizer_step": FINAL_STEP,
        "maximum_complete_optimizer_tokens": FINAL_TOKENS,
        "maximum_complete_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "unusable_tail_sequences": UNUSABLE_TAIL_SEQUENCES,
        "hard_scheduler_step_limit": HARD_STEP_LIMIT,
        "hard_scheduler_token_limit": HARD_TOKEN_LIMIT,
        "source_checkpoint_modified": False,
        "no_wrap": True,
        "second_epoch_authorized": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json", payload)
    del resume
    return payload


def disposable_resume_test(steps: int = 2) -> dict[str, Any]:
    if steps < 1 or steps > 4:
        raise ValueError("disposable resume diagnostic must use 1-4 steps")
    _, config = load_contract(requested_step=SOURCE_STEP)
    preflight = load_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json")
    if preflight.get("result") != "PASS":
        raise RuntimeError("immutable preflight must pass first")
    copy_dir = ensure_runtime_path(RUNTIME_ROOT / "temporary" / "disposable_step3051_checkpoint")
    if copy_dir.exists():
        raise FileExistsError(f"inspect stale disposable checkpoint before retry: {copy_dir}")
    shutil.copytree(SOURCE_CHECKPOINT, copy_dir)
    for relative in ("checkpoint_metadata.json", "model/model.safetensors", "resume_state.pt"):
        if sha256_file(copy_dir / relative) != sha256_file(SOURCE_CHECKPOINT / relative):
            raise ValueError(f"disposable checkpoint copy mismatch: {relative}")
    model, optimizer, scheduler, device = _create_model_stack(config)
    expected_rng = rng_fingerprint(torch.load(copy_dir / "resume_state.pt", map_location="cpu", weights_only=False)["rng"])
    state = load_checkpoint(
        copy_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=tokenized_manifest_hash(TOKENIZED_INPUT),
    )
    actual_rng = rng_fingerprint(capture_rng_state())
    validation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "validation"), sequence_length=512, micro_batch_size=2, device=device)
    evaluation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "eval"), sequence_length=512, micro_batch_size=2, device=device)
    if not math.isclose(validation["loss"], BASELINE_VALIDATION, abs_tol=1e-9):
        raise ValueError("Phase 4E disposable validation reproduction failed")
    if not math.isclose(evaluation["loss"], BASELINE_EVAL, abs_tol=1e-9):
        raise ValueError("Phase 4E disposable eval reproduction failed")
    dataset = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    records = []
    for _ in range(steps):
        step = state.step + 1
        learning_rate = optimizer.param_groups[0]["lr"]
        metric = training_step(model, optimizer, dataset, data_position=state.data_position, diagnostic=False, device=device)
        scheduler.step()
        state.step = step
        state.tokens_seen += TOKENS_PER_STEP
        state.data_position += TOKENS_PER_STEP
        records.append({"step": step, "loss": metric["raw_train_loss"], "learning_rate": learning_rate, "sequence_index": state.data_position // 512})
    checks = {
        "validation_reproduced": True,
        "eval_reproduced": True,
        "optimizer_loaded": bool(optimizer.state),
        "rng_continuity": expected_rng == actual_rng,
        "scheduler_continuity": scheduler.last_epoch == SOURCE_STEP + steps,
        "data_position_continuity": state.data_position == SOURCE_TOKENS + steps * TOKENS_PER_STEP,
        "next_sequence_exact": state.data_position // 512 == SOURCE_TOKENS // 512 + steps * SEQUENCES_PER_STEP,
        "source_model_unchanged": sha256_file(SOURCE_CHECKPOINT / "model" / "model.safetensors") == EXPECTED_SOURCE_MODEL_HASH,
        "source_resume_unchanged": sha256_file(SOURCE_CHECKPOINT / "resume_state.pt") == EXPECTED_SOURCE_RESUME_HASH,
    }
    payload = {
        "schema_version": "darkmind-v2-phase4e-disposable-resume-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "process_id": os.getpid(),
        "copied_checkpoint": str(copy_dir),
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
        raise RuntimeError("Phase 4E disposable resume test failed")
    return payload


def prepare() -> dict[str, Any]:
    authorization, _ = load_contract(requested_step=GATE_50_STEP)
    storage = storage_preflight()
    preflight = load_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json")
    disposable = load_json(RUNTIME_ROOT / "manifests" / "disposable_resume_test.json")
    if preflight.get("result") != "PASS" or disposable.get("result") != "PASS":
        raise RuntimeError("Phase 4E immutable and disposable preflights must pass")
    if RUN_DIR.exists() and (RUN_DIR / "run_manifest.json").is_file():
        existing = load_json(RUN_DIR / "run_manifest.json")
        if existing.get("authorization_file_sha256") != sha256_file(AUTHORIZATION_PATH):
            raise FileExistsError("existing Phase 4E run uses another authorization")
        return existing
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"incomplete Phase 4E run requires inspection: {RUN_DIR}")
    for directory in ("checkpoints", "diagnostics", "resume", "audits", "gates"):
        (RUN_DIR / directory).mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics.jsonl").write_text("", encoding="utf-8")
    probes = _probe_manifest()
    source_evaluation = load_json(PHASE4D_RUN / "evaluations.json")[str(SOURCE_STEP)]
    atomic_write_json(RUN_DIR / "evaluations.json", {str(SOURCE_STEP): source_evaluation})
    manifest = {
        "schema_version": "darkmind-v2-phase4e-stage3-run-v1",
        "status": "prepared",
        "authorization": authorization,
        "authorization_file_sha256": sha256_file(AUTHORIZATION_PATH),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "source_checkpoint_resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH,
        "checkpoints": {str(SOURCE_STEP): str(SOURCE_CHECKPOINT)},
        "checkpoint_hashes": {str(SOURCE_STEP): {"model_sha256": EXPECTED_SOURCE_MODEL_HASH, "resume_state_sha256": EXPECTED_SOURCE_RESUME_HASH}},
        "checkpoint_policy": {str(step): ("full_resume" if step in FULL_RESUME_STEPS else "model_only") for step in MILESTONES if step != SOURCE_STEP},
        "diagnostic_snapshots": {str(SOURCE_STEP): source_evaluation["diagnostics"]},
        "latest_resume_checkpoint": str(SOURCE_CHECKPOINT),
        "best_validation_step": SOURCE_STEP,
        "best_validation_checkpoint": str(SOURCE_CHECKPOINT),
        "sequence_order_hash": EXPECTED_ORDER_HASH,
        "data_manifest_hash": authorization["corpus_hashes"]["tokenized_manifest_file"],
        "fixed_probe_hash": probes["deterministic_content_hash"],
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "maximum_complete_optimizer_step": FINAL_STEP,
        "maximum_complete_tokens": FINAL_TOKENS,
        "maximum_complete_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "unusable_tail_sequences": UNUSABLE_TAIL_SEQUENCES,
        "process_ids": [],
        "segments": [],
        "gate_status": {"6103": "AUTHORIZED", "9155": "CONDITIONAL", "11972": "CONDITIONAL"},
        "runtime_outside_onedrive": True,
        "immutable_input_bytes_duplicated": 0,
        "no_wrap": True,
        "second_epoch_authorized": False,
        "storage_preflight": storage,
    }
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": "prepared", "optimizer_step": SOURCE_STEP, "tokens_consumed": SOURCE_TOKENS, "sequence_index": SOURCE_TOKENS // 512})
    return manifest


def _rename_checkpoint(temporary: Path, checkpoint: Path) -> None:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4E checkpoint: {checkpoint}")
    for attempt in range(20):
        try:
            os.replace(temporary, checkpoint)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def save_milestone(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: Any,
    state: TrainingState,
    data_hash: str,
    authorization_hash: str,
) -> tuple[Path, dict[str, Any]]:
    if step not in MODEL_ONLY_STEPS | FULL_RESUME_STEPS:
        raise ValueError(f"unsupported Phase 4E milestone: {step}")
    checkpoint = ensure_runtime_path(RUN_DIR / "checkpoints" / checkpoint_name(step))
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(f"Phase 4E milestone already exists: {checkpoint}")
    if step in FULL_RESUME_STEPS:
        metadata = save_checkpoint(
            temporary,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=state,
            tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
            data_manifest_hash=data_hash,
        )
        kind = "full_resume"
    else:
        temporary.mkdir(parents=True)
        metadata = {"model_files": save_model_package(model, temporary / "model"), "training_state": state.to_dict()}
        kind = "model_only"
    stage_metadata = {
        "schema_version": "darkmind-v2-phase4e-stage3-checkpoint-v1",
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
        "next_learning_rate": optimizer.param_groups[0]["lr"],
    }
    atomic_write_json(temporary / "stage3_checkpoint_metadata.json", stage_metadata)
    _validate_safetensors(temporary / "model" / "model.safetensors")
    _rename_checkpoint(temporary, checkpoint)
    hashes = {
        "model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "stage3_metadata_sha256": sha256_file(checkpoint / "stage3_checkpoint_metadata.json"),
    }
    if kind == "full_resume":
        hashes["resume_state_sha256"] = sha256_file(checkpoint / "resume_state.pt")
        hashes["checkpoint_metadata_sha256"] = sha256_file(checkpoint / "checkpoint_metadata.json")
    return checkpoint, {**stage_metadata, "hashes": hashes}


def _assert_resume_state(state: TrainingState, expected_step: int) -> None:
    expected_tokens = expected_step * TOKENS_PER_STEP
    if (state.step, state.tokens_seen, state.data_position) != (expected_step, expected_tokens, expected_tokens):
        raise ValueError("Phase 4E resume step/token/data-position mismatch")


def _expected_segment(target_step: int, manifest: dict[str, Any]) -> int:
    expected = {GATE_50_STEP: SOURCE_STEP, GATE_75_STEP: GATE_50_STEP, FINAL_STEP: GATE_75_STEP}
    if target_step not in expected:
        raise ValueError("Phase 4E target must be 6103, 9155, or 11972")
    required_segments = {GATE_50_STEP: 0, GATE_75_STEP: 1, FINAL_STEP: 2}[target_step]
    if len(manifest["segments"]) != required_segments:
        raise ValueError("Phase 4E segment order or process boundary changed")
    return expected[target_step]


def train_segment(target_step: int) -> dict[str, Any]:
    authorization, config = load_contract(requested_step=target_step)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    expected_start = _expected_segment(target_step, manifest)
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
    _assert_resume_state(state, expected_start)
    actual_rng = rng_fingerprint(capture_rng_state())
    if actual_rng != expected_rng or scheduler.last_epoch != expected_start:
        raise ValueError("Phase 4E RNG or scheduler continuity failed")
    expected_lr = learning_rate_for_policy(expected_start + 1, config["schedule"])
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, abs_tol=1e-15):
        raise ValueError("Phase 4E next applied LR mismatch")
    if os.getpid() in manifest["process_ids"]:
        raise ValueError("Phase 4E segment did not use a fresh process")
    if expected_start != SOURCE_STEP:
        validation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "validation"), sequence_length=512, micro_batch_size=2, device=device)
        evaluation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "eval"), sequence_length=512, micro_batch_size=2, device=device)
        prior = load_json(RUN_DIR / "evaluations.json")[str(expected_start)]
        if not math.isclose(validation["loss"], prior["validation"]["loss"], abs_tol=1e-9) or not math.isclose(evaluation["loss"], prior["eval"]["loss"], abs_tol=1e-9):
            raise ValueError("Phase 4E gate evaluation changed after process restart")
    manifest["process_ids"].append(os.getpid())
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)

    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != EXPECTED_ORDER_HASH or len(order_data.order) != UNIQUE_SEQUENCE_CAPACITY:
        raise ValueError("Phase 4E sequence order changed")
    if target_step * SEQUENCES_PER_STEP > USABLE_SEQUENCE_CAPACITY:
        raise PermissionError("Phase 4E target would wrap or use a partial optimizer batch")
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_base = TokenShardDataset(TOKENIZED_INPUT, "train")
    probes = _probe_manifest()
    activation_index = int(probes["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_base.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    evaluations = load_json(RUN_DIR / "evaluations.json")
    prior_metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gradient_values = [float(item["pre_clip_gradient_norm"]) for item in prior_metrics]
    clipped_steps = sum(int(item["clipped"]) for item in prior_metrics)
    repeated_high_update = 0
    for item in reversed(prior_metrics):
        if item["update_to_weight"]["maximum"] <= 0.005:
            break
        repeated_high_update += 1
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    start_position = state.data_position
    expected_sequence_cursor = state.data_position // 512
    try:
        while state.step < target_step:
            step = state.step + 1
            validate_authorization(authorization, requested_step=step)
            learning_rate = optimizer.param_groups[0]["lr"]
            diagnostic = step in MILESTONES
            metric = training_step(model, optimizer, order_data, data_position=state.data_position, diagnostic=diagnostic, device=device)
            expected_indices = order_data.order[expected_sequence_cursor : expected_sequence_cursor + SEQUENCES_PER_STEP]
            if metric["source_sequence_indices"] != expected_indices or len(expected_indices) != SEQUENCES_PER_STEP:
                raise ValueError("Phase 4E repeated, skipped, or partial sequence batch")
            expected_sequence_cursor += SEQUENCES_PER_STEP
            scheduler.step()
            state.step = step
            state.tokens_seen += TOKENS_PER_STEP
            state.data_position += TOKENS_PER_STEP
            _assert_resume_state(state, step)
            pre_clip = float(metric.pop("gradient_norm"))
            clipping_coefficient = min(1.0, 1.0 / (pre_clip + 1e-6))
            post_clip = pre_clip * clipping_coefficient
            clipped_steps += int(metric["clipped"])
            gradient_values.append(pre_clip)
            update_max = float(metric["update_to_weight"]["maximum"])
            repeated_high_update = repeated_high_update + 1 if update_max > 0.005 else 0
            if repeated_high_update >= 3:
                raise FloatingPointError("Phase 4E repeated update-to-weight ratio above 0.005")
            state.last_training_loss = metric["raw_train_loss"]
            state.smoothed_training_loss = metric["raw_train_loss"] if state.smoothed_training_loss is None else 0.9 * state.smoothed_training_loss + 0.1 * metric["raw_train_loss"]
            parameter_diagnostics = metric.pop("parameter_diagnostics")
            metric.update(
                {
                    "optimizer_step": step,
                    "tokens_consumed": state.tokens_seen,
                    "sequence_index": state.data_position // 512,
                    "pre_clip_gradient_norm": pre_clip,
                    "clipping_coefficient": clipping_coefficient,
                    "post_clip_gradient_norm": post_clip,
                    "learning_rate_applied": learning_rate,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "smoothed_train_loss": state.smoothed_training_loss,
                    "clipped_step_fraction": clipped_steps / max(step - SOURCE_STEP, 1),
                    "allocated_vram_bytes": torch.cuda.memory_allocated(device),
                    "reserved_vram_bytes": torch.cuda.memory_reserved(device),
                    "gpu": monitor.latest(),
                    "non_finite_event_count": 0,
                }
            )
            if diagnostic:
                validation = evaluate_loss(model, validation_data, sequence_length=512, micro_batch_size=2, device=device)
                evaluation = evaluate_loss(model, eval_data, sequence_length=512, micro_batch_size=2, device=device)
                probe_losses = {name: evaluate_probe(model, probe, device) for name, probe in probes["probes"].items()}
                diagnostics = _milestone_diagnostics(
                    model,
                    step=step,
                    parameter_diagnostics=parameter_diagnostics,
                    clipped_fraction=clipped_steps / max(step - SOURCE_STEP, 1),
                    probe_losses=probe_losses,
                    activation_tokens=activation_tokens,
                )
                diagnostics["training_health"] = {
                    "gradient_p50": percentile(gradient_values, 0.50),
                    "gradient_p95": percentile(gradient_values, 0.95),
                    "gradient_max": max(gradient_values),
                    "phase4d_gradient_p95": PHASE4D_GRADIENT_P95,
                    "clipped_step_fraction": clipped_steps / max(step - SOURCE_STEP, 1),
                    "update_to_weight_current": metric["update_to_weight"],
                    "non_finite_events": 0,
                }
                if diagnostics["training_health"]["gradient_p95"] > 2.0 * PHASE4D_GRADIENT_P95:
                    raise FloatingPointError("Phase 4E gradient p95 exceeded twice the Phase 4D baseline")
                prior_step = max((value for value in MILESTONES if value < step and str(value) in evaluations), default=None)
                if prior_step is not None and diagnostics["training_health"]["clipped_step_fraction"] >= 0.999 and validation["loss"] > evaluations[str(prior_step)]["validation"]["loss"]:
                    raise FloatingPointError("Phase 4E effectively 100% clipping accompanied worsening validation")
                diagnostic_path = RUN_DIR / "diagnostics" / f"step_{step:06d}.json"
                atomic_write_json(diagnostic_path, diagnostics)
                state.last_validation_loss = validation["loss"]
                prospective = RUN_DIR / "checkpoints" / checkpoint_name(step)
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = validation["loss"]
                    state.best_checkpoint = str(prospective)
                    manifest["best_validation_step"] = step
                    manifest["best_validation_checkpoint"] = str(prospective)
                saved, checkpoint_metadata = save_milestone(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    data_hash=manifest["data_manifest_hash"],
                    authorization_hash=manifest["authorization_file_sha256"],
                )
                evaluations[str(step)] = {
                    "train": {"loss": metric["raw_train_loss"], "smoothed_loss": state.smoothed_training_loss},
                    "validation": validation,
                    "eval": evaluation,
                    "probes": probe_losses,
                    "diagnostics": str(diagnostic_path),
                }
                manifest["checkpoints"][str(step)] = str(saved)
                manifest["checkpoint_hashes"][str(step)] = checkpoint_metadata["hashes"]
                manifest["diagnostic_snapshots"][str(step)] = str(diagnostic_path)
                if step in FULL_RESUME_STEPS:
                    manifest["latest_resume_checkpoint"] = str(saved)
                metric.update({"validation_loss": validation["loss"], "eval_loss": evaluation["loss"], "checkpoint": str(saved)})
                atomic_write_json(RUN_DIR / "evaluations.json", evaluations)
                atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
            append_jsonl(RUN_DIR / "metrics.jsonl", metric)
            atomic_write_json(RUN_DIR / "progress.json", {"status": "training", "optimizer_step": step, "tokens_consumed": state.tokens_seen, "sequence_index": state.data_position // 512})
            if step % 10 == 0 or diagnostic:
                print(f"phase4e step={step} loss={metric['raw_train_loss']:.6f} lr={learning_rate:.9f} tok_s={metric['active_tokens_per_second']:.1f}", flush=True)
    finally:
        monitor.close()
    if state.step != target_step or state.tokens_seen != target_step * TOKENS_PER_STEP:
        raise RuntimeError("Phase 4E segment missed its exact stop")
    segment = {
        "schema_version": "darkmind-v2-phase4e-stage3-segment-v1",
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
        "data_position_continuity": expected_sequence_cursor == state.data_position // 512,
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
    status = {GATE_50_STEP: "awaiting_50m_gate", GATE_75_STEP: "awaiting_75m_gate", FINAL_STEP: "first_corpus_pass_training_complete"}[target_step]
    manifest["status"] = status
    atomic_write_json(RUN_DIR / "resume" / f"segment_to_step_{target_step:06d}.json", segment)
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": status, "optimizer_step": target_step, "tokens_consumed": state.tokens_seen, "sequence_index": state.data_position // 512})
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return segment


def _improvement(start: float, end: float) -> float:
    return (start - end) * 100.0 / start


def evaluate_gate(step: int) -> dict[str, Any]:
    if step not in (GATE_50_STEP, GATE_75_STEP):
        raise ValueError("Phase 4E gate must be 6103 or 9155")
    manifest = load_json(RUN_DIR / "run_manifest.json")
    expected_segments = 1 if step == GATE_50_STEP else 2
    if len(manifest["segments"]) != expected_segments or manifest["segments"][-1]["segment_end_step"] != step:
        raise RuntimeError("Phase 4E gate cannot run before exact segment completion")
    audit_path = RUN_DIR / "audits" / f"step_{step:06d}" / "subset" / "audit_summary.json"
    if not audit_path.is_file():
        raise RuntimeError("Phase 4E controlled generation audit must run before the gate")
    generation = load_json(audit_path)
    evaluations = load_json(RUN_DIR / "evaluations.json")
    baseline_step = SOURCE_STEP if step == GATE_50_STEP else GATE_50_STEP
    steps = [value for value in MILESTONES if baseline_step <= value <= step]
    start = evaluations[str(baseline_step)]
    final = evaluations[str(step)]
    validation_losses = [float(evaluations[str(value)]["validation"]["loss"]) for value in steps]
    eval_losses = [float(evaluations[str(value)]["eval"]["loss"]) for value in steps]
    val_improvement = _improvement(float(start["validation"]["loss"]), float(final["validation"]["loss"]))
    eval_improvement = _improvement(float(start["eval"]["loss"]), float(final["eval"]["loss"]))
    val_rebound = rebound_percent(min(validation_losses), validation_losses[-1])
    eval_rebound = rebound_percent(min(eval_losses), eval_losses[-1])
    catastrophic = {
        name: float(final["probes"][name]["loss"]) > float(start["probes"][name]["loss"]) * 1.20
        for name in start["probes"]
        if name in final["probes"]
    }
    late = all(right > left for left, right in zip(validation_losses[-3:], validation_losses[-2:])) or all(right > left for left, right in zip(eval_losses[-3:], eval_losses[-2:]))
    hard_failures = int(generation["greedy"]["hard_failure_total"]) + int(generation["sampling"]["hard_failure_total"])
    integrity = all(
        segment["result"] == "PASS"
        and segment["rng_continuity"]
        and segment["scheduler_continuity"]
        and segment["data_position_continuity"]
        and segment["no_repeated_or_skipped_sequence"]
        and segment["no_data_wrap"]
        for segment in manifest["segments"]
    )
    fail_floor = 2.0 if step == GATE_50_STEP else 1.0
    pass_floor = 5.0 if step == GATE_50_STEP else 3.0
    failed = (
        val_improvement < fail_floor
        or eval_improvement < fail_floor
        or validation_losses[-1] > validation_losses[0]
        or eval_losses[-1] > eval_losses[0]
        or val_rebound > 2.0
        or eval_rebound > 2.0
        or late
        or any(catastrophic.values())
        or hard_failures != 0
        or not integrity
    )
    if failed:
        result = "FAIL"
    elif val_improvement < pass_floor or eval_improvement < pass_floor:
        result = "CONDITIONAL"
    else:
        result = "PASS"
    payload = {
        "schema_version": f"darkmind-v2-phase4e-{step}-continuation-gate-v1",
        "result": result,
        "continuation_authorized": result == "PASS",
        "baseline_step": baseline_step,
        "gate_step": step,
        "validation_improvement_percent": val_improvement,
        "eval_improvement_percent": eval_improvement,
        "validation_rebound_percent": val_rebound,
        "eval_rebound_percent": eval_rebound,
        "sustained_late_worsening": late,
        "catastrophic_probe_regressions": catastrophic,
        "generation_hard_failures": hard_failures,
        "integrity_pass": integrity,
        "next_target_step": (GATE_75_STEP if step == GATE_50_STEP else FINAL_STEP) if result == "PASS" else None,
        "second_epoch_authorized": False,
    }
    atomic_write_json(_gate_path(step), payload)
    manifest["gate_status"][str(step)] = result
    if result == "PASS":
        manifest["gate_status"][str(GATE_75_STEP if step == GATE_50_STEP else FINAL_STEP)] = "AUTHORIZED"
        manifest["status"] = "50m_gate_passed" if step == GATE_50_STEP else "75m_gate_passed"
    else:
        manifest["status"] = f"stopped_at_{step}_{result.lower()}"
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(
        RUN_DIR / "progress.json",
        {
            "status": manifest["status"],
            "optimizer_step": step,
            "tokens_consumed": step * TOKENS_PER_STEP,
            "sequence_index": step * SEQUENCES_PER_STEP,
        },
    )
    return payload


def _probe_comparison(start: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in final.items():
        if name not in start:
            continue
        baseline = float(start[name]["loss"])
        end = float(value["loss"])
        result[name] = {
            "baseline_loss": baseline,
            "final_loss": end,
            "improvement_percent": _improvement(baseline, end),
            "catastrophic_regression": end > baseline * 1.20,
        }
    return result


def finalize_training_summary() -> dict[str, Any]:
    manifest = load_json(RUN_DIR / "run_manifest.json")
    if len(manifest["segments"]) != 3 or manifest["segments"][-1]["segment_end_step"] != FINAL_STEP:
        raise RuntimeError("Phase 4E all three exact segments must complete first")
    if not _passed_gate(GATE_50_STEP) or not _passed_gate(GATE_75_STEP):
        raise RuntimeError("Phase 4E intermediate gates did not pass")
    evaluations = load_json(RUN_DIR / "evaluations.json")
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(metrics) != FINAL_STEP - SOURCE_STEP or metrics[-1]["optimizer_step"] != FINAL_STEP:
        raise ValueError("Phase 4E telemetry missed or exceeded the no-wrap stop")
    validation_losses = [float(evaluations[str(step)]["validation"]["loss"]) for step in MILESTONES]
    eval_losses = [float(evaluations[str(step)]["eval"]["loss"]) for step in MILESTONES]
    update_values = [float(item["update_to_weight"]["maximum"]) for item in metrics]
    gradients = [float(item["pre_clip_gradient_norm"]) for item in metrics]
    probes = _probe_comparison(evaluations[str(SOURCE_STEP)]["probes"], evaluations[str(FINAL_STEP)]["probes"])
    integrity = (
        len(set(manifest["process_ids"])) == 3
        and all(segment["result"] == "PASS" and segment["rng_continuity"] and segment["scheduler_continuity"] and segment["data_position_continuity"] and segment["no_repeated_or_skipped_sequence"] and segment["no_data_wrap"] for segment in manifest["segments"])
        and sum(int(item["non_finite_event_count"]) for item in metrics) == 0
    )
    val_improvement = _improvement(validation_losses[0], validation_losses[-1])
    eval_improvement = _improvement(eval_losses[0], eval_losses[-1])
    val_rebound = rebound_percent(min(validation_losses), validation_losses[-1])
    eval_rebound = rebound_percent(min(eval_losses), eval_losses[-1])
    late = all(right > left for left, right in zip(validation_losses[-3:], validation_losses[-2:])) or all(right > left for left, right in zip(eval_losses[-3:], eval_losses[-2:]))
    catastrophic = any(item["catastrophic_regression"] for item in probes.values())
    if not integrity or val_improvement < 3.0 or eval_improvement < 3.0 or val_rebound > 2.0 or eval_rebound > 2.0 or late or catastrophic:
        preliminary = "FAIL"
    elif val_improvement >= 10.0 and eval_improvement >= 10.0:
        preliminary = "STRONG PASS"
    else:
        preliminary = "CONDITIONAL PASS"
    summary = {
        "schema_version": "darkmind-v2-phase4e-stage3-training-summary-v1",
        "integrity_pass": integrity,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "starting_optimizer_step": SOURCE_STEP,
        "final_optimizer_step": FINAL_STEP,
        "starting_tokens": SOURCE_TOKENS,
        "final_tokens": FINAL_TOKENS,
        "final_sequence_index": USABLE_SEQUENCE_CAPACITY,
        "unique_sequence_capacity": UNIQUE_SEQUENCE_CAPACITY,
        "unusable_tail_sequences": UNUSABLE_TAIL_SEQUENCES,
        "first_deterministic_corpus_pass_complete_under_frozen_full_batch_policy": True,
        "second_epoch_started": False,
        "validation_improvement_percent": val_improvement,
        "eval_improvement_percent": eval_improvement,
        "validation_rebound_percent": val_rebound,
        "eval_rebound_percent": eval_rebound,
        "last_three_sustained_worsening": late,
        "validation_progression": {str(step): evaluations[str(step)]["validation"] for step in MILESTONES},
        "eval_progression": {str(step): evaluations[str(step)]["eval"] for step in MILESTONES},
        "train_loss_progression": {str(step): evaluations[str(step)].get("train", {}) for step in MILESTONES},
        "learning_rate_progression": {str(step): learning_rate_for_policy(step, load_json(V2_CONFIG)["schedule"]) for step in MILESTONES},
        "gradient_norm_p50": percentile(gradients, 0.50),
        "gradient_norm_p95": percentile(gradients, 0.95),
        "gradient_norm_max": max(gradients),
        "clipped_step_fraction": sum(int(item["clipped"]) for item in metrics) / len(metrics),
        "update_to_weight_p50": percentile(update_values, 0.50),
        "update_to_weight_p95": percentile(update_values, 0.95),
        "update_to_weight_max": max(update_values),
        "non_finite_events": sum(int(item["non_finite_event_count"]) for item in metrics),
        "probe_comparison": probes,
        "evaluations": evaluations,
        "diagnostic_snapshots": manifest["diagnostic_snapshots"],
        "checkpoints": manifest["checkpoints"],
        "checkpoint_hashes": manifest["checkpoint_hashes"],
        "best_checkpoint": manifest["best_validation_checkpoint"],
        "best_validation_step": manifest["best_validation_step"],
        "active_tokens_per_second": (FINAL_TOKENS - SOURCE_TOKENS) / sum(float(item["optimizer_step_duration_seconds"]) for item in metrics),
        "wall_tokens_per_second": (FINAL_TOKENS - SOURCE_TOKENS) / sum(float(segment["elapsed_seconds"]) for segment in manifest["segments"]),
        "peak_allocated_vram_bytes": max(segment["peak_allocated_bytes"] for segment in manifest["segments"]),
        "peak_reserved_vram_bytes": max(segment["peak_reserved_bytes"] for segment in manifest["segments"]),
        "process_restart": {
            "result": "PASS" if integrity else "FAIL",
            "fresh_processes": len(set(manifest["process_ids"])) == 3,
            "process_ids": manifest["process_ids"],
            "optimizer_continuity": all(segment["optimizer_continuity"] for segment in manifest["segments"]),
            "scheduler_continuity": all(segment["scheduler_continuity"] for segment in manifest["segments"]),
            "rng_continuity": all(segment["rng_continuity"] for segment in manifest["segments"]),
            "data_position_continuity": all(segment["data_position_continuity"] for segment in manifest["segments"]),
            "no_repeated_or_skipped_sequence": all(segment["no_repeated_or_skipped_sequence"] for segment in manifest["segments"]),
        },
        "preliminary_classification": preliminary,
        "final_classification_requires_generation_and_memorization_audits": True,
        "second_epoch_authorized": False,
    }
    atomic_write_json(RUN_DIR / "training_summary.json", summary)
    manifest["status"] = "awaiting_final_audits"
    manifest["preliminary_classification"] = preliminary
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    return summary


def validate_final_resume() -> dict[str, Any]:
    _, config = load_contract(requested_step=FINAL_STEP)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    checkpoint = Path(manifest["latest_resume_checkpoint"])
    expected = RUN_DIR / "checkpoints" / checkpoint_name(FINAL_STEP)
    if checkpoint != expected:
        raise ValueError("Phase 4E final resume checkpoint changed")
    resume = torch.load(checkpoint / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_rng = rng_fingerprint(resume["rng"])
    model, optimizer, scheduler, device = _create_model_stack(config)
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=manifest["data_manifest_hash"],
    )
    _assert_resume_state(state, FINAL_STEP)
    checks = {
        "model_hash": sha256_file(checkpoint / "model" / "model.safetensors") == manifest["checkpoint_hashes"][str(FINAL_STEP)]["model_sha256"],
        "optimizer_loaded": bool(optimizer.state),
        "scheduler_epoch": scheduler.last_epoch == FINAL_STEP,
        "rng_continuity": rng_fingerprint(capture_rng_state()) == expected_rng,
        "data_position": state.data_position == FINAL_TOKENS,
        "final_sequence_index": state.data_position // 512 == USABLE_SEQUENCE_CAPACITY,
        "no_step_11973": state.step == FINAL_STEP,
        "no_second_epoch": manifest["second_epoch_authorized"] is False,
    }
    payload = {
        "schema_version": "darkmind-v2-phase4e-future-resume-validation-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "resume_state_sha256": sha256_file(checkpoint / "resume_state.pt"),
        "optimizer_step": state.step,
        "tokens_consumed": state.tokens_seen,
        "final_sequence_index": state.data_position // 512,
        "remaining_unusable_tail_sequences": UNUSABLE_TAIL_SEQUENCES,
        "next_learning_rate_if_separately_authorized": learning_rate_for_policy(FINAL_STEP + 1, config["schedule"]),
        "checks": checks,
        "step_11973_executed": False,
        "second_epoch_started": False,
    }
    atomic_write_json(RUN_DIR / "future_resume_validation.json", payload)
    del model, optimizer, scheduler, resume
    torch.cuda.empty_cache()
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4E final resume validation failed")
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
    segment.add_argument("--target-step", type=int, choices=(GATE_50_STEP, GATE_75_STEP, FINAL_STEP), required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--step", type=int, choices=(GATE_50_STEP, GATE_75_STEP), required=True)
    commands.add_parser("finalize-training")
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
        payload = evaluate_gate(args.step)
    elif args.command == "finalize-training":
        payload = finalize_training_summary()
    else:
        payload = validate_final_resume()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
