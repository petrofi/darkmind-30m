"""Continue the immutable Base V1 V2 checkpoint to the exact 25M gate."""

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
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from darkmind_v2.training.phase3b_finalist_pilots import evaluate_loss
from darkmind_v2.training.phase4b_factorial import OrderedTokenDataset, percentile, rebound_percent
from darkmind_v2.training.phase4b_runtime import load_document_spans, sequence_labels
from darkmind_v2.training.phase4c_confirmation import rng_fingerprint
from darkmind_v2.training.phase4c_diagnostics import (
    INITIALIZATION_SEED,
    MODEL_INPUT,
    ORDER_INPUT,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    build_scheduler,
    learning_rate_for_policy,
)
from darkmind_v2.training.phase4c_policy import V2_CONFIG, validate_v2_config
from darkmind_v2.training.phase4c_training import (
    _max_consecutive_worsening,
    _milestone_diagnostics,
    _parameter_diagnostics,
    apply_initialization_policy,
    build_optimizer,
    evaluate_probe,
)
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_base_v1_stage1 import GpuMonitor
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_BOUNDARIES_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_SHARD_CHECKSUMS_HASH,
    EXPECTED_TOKENIZED_HASH,
    ROOT,
)


RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase4d")
RUN_DIR = RUNTIME_ROOT / "runs" / "base_v1_stage2_25m_v2_retry1"
SOURCE_CHECKPOINT = Path(
    r"C:\DarkMindRuntime\phase4c\runs\base_v1_stage1_5m_v2_confirmation"
    r"\checkpoints\step_000610_tokens_004997120"
)
SOURCE_RUN = SOURCE_CHECKPOINT.parents[1]
AUTHORIZATION_PATH = ROOT / "darkmind_v2" / "config" / "train_base_v1_stage2_25m_authorization.json"
MILESTONES = (610, 1024, 1536, 1831, 2048, 2560, 3051)
MODEL_ONLY_STEPS = {1024, 1536, 2048, 2560}
FULL_RESUME_STEPS = {1831, 3051}
TOKENS_PER_STEP = 8192
SEQUENCES_PER_STEP = 16
SOURCE_STEP = 610
SOURCE_TOKENS = 4_997_120
TARGET_STEP = 3_051
TARGET_TOKENS = 24_993_792
EXPECTED_SOURCE_MODEL_HASH = "9222e15c8f3dc3972f6df6a747a87ad54cb37e59e06bf6f63ad8a5c3ef52d3b4"
EXPECTED_V2_FILE_HASH = "9358b8b33a87729ef2f19cfad76acbe370ed44ada911fa21512e4085eccf52ea"
BASELINE_VALIDATION = 6.356268899158205
BASELINE_EVAL = 6.306322918196789


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def ensure_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    if "onedrive" in str(resolved).lower():
        raise ValueError(f"Phase 4D mutable runtime cannot use OneDrive: {resolved}")
    try:
        resolved.relative_to(RUNTIME_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Phase 4D mutable runtime escaped its root: {resolved}") from exc
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_name(step: int) -> str:
    return f"step_{step:06d}_tokens_{step * TOKENS_PER_STEP:09d}"


def validate_authorization(payload: dict[str, Any], *, requested_step: int | None = None, requested_tokens: int | None = None) -> None:
    required = {
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "current_optimizer_step": SOURCE_STEP,
        "target_optimizer_step": TARGET_STEP,
        "maximum_authorized_step": TARGET_STEP,
        "current_tokens": SOURCE_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "maximum_authorized_tokens": TARGET_TOKENS,
        "current_sequence_index": SOURCE_TOKENS // 512,
        "target_sequence_index": TARGET_TOKENS // 512,
        "continuation_authorized": True,
        "continuation_100m_authorized": False,
        "scheduler_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
        "data_order_reset": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Stage-2 authorization mismatch: {key}")
    if payload.get("tokenizer_hashes") != EXPECTED_HASHES:
        raise ValueError("Stage-2 authorization tokenizer identity changed")
    corpus = payload.get("corpus_hashes", {})
    expected_corpus = {
        "corpus": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_content": EXPECTED_TOKENIZED_HASH,
        "tokenized_manifest_file": "85532a6933f04f7983e0b98bae5ec0c33f3110709eb248c9116858a5bcc8ab37",
        "document_boundaries": EXPECTED_BOUNDARIES_HASH,
        "shard_checksums": EXPECTED_SHARD_CHECKSUMS_HASH,
        "sequence_order": "4d31d6cf5532cd1729b35528a649c52b4552d8959a10b5813ce9851bb714ffd1",
    }
    if corpus != expected_corpus:
        raise ValueError("Stage-2 authorization corpus identity changed")
    if payload.get("training_segments") != [
        {"start_step": 611, "end_step": 1831},
        {"start_step": 1832, "end_step": 3051},
    ]:
        raise ValueError("Stage-2 process boundary changed")
    step = TARGET_STEP if requested_step is None else int(requested_step)
    tokens = step * TOKENS_PER_STEP if requested_tokens is None else int(requested_tokens)
    if step > TARGET_STEP or tokens > TARGET_TOKENS:
        raise PermissionError("Stage-2 authorization limit violation")
    if tokens != step * TOKENS_PER_STEP:
        raise ValueError("Stage-2 step/token boundary mismatch")


def load_contract(*, requested_step: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = load_json(AUTHORIZATION_PATH)
    validate_authorization(authorization, requested_step=requested_step)
    config = load_json(V2_CONFIG)
    validate_v2_config(config)
    if sha256_file(V2_CONFIG) != EXPECTED_V2_FILE_HASH:
        raise ValueError("frozen V2 config file hash changed")
    return authorization, config


def storage_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(RUNTIME_ROOT.anchor)
    phase4b = directory_size(Path(r"C:\DarkMindRuntime\phase4b"))
    phase4c = directory_size(Path(r"C:\DarkMindRuntime\phase4c"))
    model_only = directory_size(SOURCE_CHECKPOINT / "model")
    full_resume = directory_size(SOURCE_CHECKPOINT)
    metrics_and_audits = 21_568_406
    export_estimate = model_only + 5_000_000
    contingency = 512 * 1024 * 1024
    projected = model_only * 4 + full_resume * 3 + metrics_and_audits + export_estimate + contingency
    reserve = 10_000_000_000
    payload = {
        "schema_version": "darkmind-v2-phase4d-storage-preflight-v1",
        "result": "PASS" if usage.free - projected >= reserve else "FAIL",
        "available_bytes": usage.free,
        "phase4b_bytes": phase4b,
        "phase4c_bytes": phase4c,
        "expected_model_only_checkpoint_bytes": model_only,
        "expected_full_resume_checkpoint_bytes": full_resume,
        "expected_metrics_and_audits_bytes": metrics_and_audits,
        "expected_local_export_bytes": export_estimate,
        "contingency_bytes": contingency,
        "projected_phase4d_bytes": projected,
        "projected_free_after_completion_bytes": usage.free - projected,
        "required_free_reserve_bytes": reserve,
        "immutable_input_bytes_duplicated": 0,
        "immutable_inputs_reused_read_only": str(TOKENIZED_INPUT),
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "storage_preflight.json", payload)
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4D disk reserve gate failed")
    return payload


def _state_digest(value: Any, digest: hashlib._Hash, label: str = "root") -> None:
    digest.update(label.encode("utf-8"))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _state_digest(value[key], digest, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _state_digest(item, digest, f"{label}[{index}]")
    else:
        digest.update(repr(value).encode("utf-8"))


def state_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _state_digest(value, digest)
    return digest.hexdigest()


def _validate_shards() -> dict[str, Any]:
    manifest_path = TOKENIZED_INPUT / "tokenized_corpus_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("deterministic_content_hash") != EXPECTED_TOKENIZED_HASH:
        raise ValueError("Corpus V3 tokenized content hash changed")
    boundaries = TOKENIZED_INPUT / manifest["document_boundaries"]["filename"]
    checksums_path = TOKENIZED_INPUT / "shard_checksums.json"
    if sha256_file(boundaries) != EXPECTED_BOUNDARIES_HASH:
        raise ValueError("Corpus V3 boundary hash changed")
    if sha256_file(checksums_path) != EXPECTED_SHARD_CHECKSUMS_HASH:
        raise ValueError("Corpus V3 shard checksum manifest changed")
    checksums = load_json(checksums_path)
    verified = {}
    for filename, expected in sorted(checksums.items()):
        actual = sha256_file(TOKENIZED_INPUT / filename)
        if actual != expected:
            raise ValueError(f"Corpus V3 shard changed: {filename}")
        verified[filename] = actual
    return {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_hash": manifest["deterministic_content_hash"],
        "boundary_sha256": sha256_file(boundaries),
        "shard_checksum_manifest_sha256": sha256_file(checksums_path),
        "verified_shards": verified,
    }


def immutable_preflight() -> dict[str, Any]:
    authorization, config = load_contract(requested_step=TARGET_STEP)
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Base V1 model config changed")
    tokenizer_manifest = verify_frozen_tokenizer(TOKENIZER_INPUT)
    if tokenizer_manifest.get("model_sha256") != EXPECTED_HASHES["tokenizer.model"]:
        raise ValueError("frozen tokenizer validation hash mismatch")
    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != authorization["corpus_hashes"]["sequence_order"]:
        raise ValueError("deterministic sequence-order hash changed")
    corpus = _validate_shards()
    if corpus["manifest_file_sha256"] != authorization["corpus_hashes"]["tokenized_manifest_file"]:
        raise ValueError("tokenized manifest file hash changed")
    metadata = load_json(SOURCE_CHECKPOINT / "checkpoint_metadata.json")
    state = metadata["training_state"]
    if state["step"] != SOURCE_STEP or state["tokens_seen"] != SOURCE_TOKENS or state["data_position"] != SOURCE_TOKENS:
        raise ValueError("source checkpoint training state changed")
    model_hash = sha256_file(SOURCE_CHECKPOINT / "model" / "model.safetensors")
    if model_hash != EXPECTED_SOURCE_MODEL_HASH or metadata["model_files"]["model_sha256"] != model_hash:
        raise ValueError("source checkpoint model hash changed")
    resume_path = SOURCE_CHECKPOINT / "resume_state.pt"
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    scheduler_state = resume["scheduler"]
    if scheduler_state.get("last_epoch") != SOURCE_STEP:
        raise ValueError("source scheduler state is not at step 610")
    source_last_metric = json.loads((SOURCE_RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    expected_applied_lr = learning_rate_for_policy(SOURCE_STEP, config["schedule"])
    expected_next_lr = learning_rate_for_policy(SOURCE_STEP + 1, config["schedule"])
    if not math.isclose(source_last_metric["learning_rate_applied"], expected_applied_lr, abs_tol=1e-15):
        raise ValueError("source applied LR changed")
    if not math.isclose(float(resume["optimizer"]["param_groups"][0]["lr"]), expected_next_lr, abs_tol=1e-15):
        raise ValueError("source next optimizer LR changed")
    payload = {
        "schema_version": "darkmind-v2-phase4d-immutable-resume-preflight-v1",
        "result": "PASS",
        "authorization_file_sha256": sha256_file(AUTHORIZATION_PATH),
        "v2_config_file_sha256": sha256_file(V2_CONFIG),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_metadata_sha256": sha256_file(SOURCE_CHECKPOINT / "checkpoint_metadata.json"),
        "model_weight_sha256": model_hash,
        "resume_state_file_sha256": sha256_file(resume_path),
        "optimizer_state_sha256": state_hash(resume["optimizer"]),
        "scheduler_state_sha256": state_hash(resume["scheduler"]),
        "rng_state_sha256": rng_fingerprint(resume["rng"]),
        "training_state": state,
        "current_optimizer_step": SOURCE_STEP,
        "consumed_tokens": SOURCE_TOKENS,
        "consumed_sequences": SOURCE_TOKENS // 512,
        "next_optimizer_step": SOURCE_STEP + 1,
        "next_sequence_index": SOURCE_TOKENS // 512,
        "applied_learning_rate_at_gate": source_last_metric["learning_rate_applied"],
        "next_learning_rate": expected_next_lr,
        "model_config_sha256": sha256_file(MODEL_INPUT),
        "architecture_hash": metadata["model_config_hash"],
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "corpus": corpus,
        "sequence_order_hash": order_data.order_hash,
        "sequence_order_complete_sequences": len(order_data.order),
        "source_checkpoint_modified": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json", payload)
    del resume
    return payload


def _create_model_stack(config: dict[str, Any]) -> tuple[DarkMindV2ForCausalLM, torch.optim.AdamW, Any, torch.device]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 4D requires CUDA with BF16 support")
    set_deterministic_seed(INITIALIZATION_SEED)
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(DarkMindV2Config.from_json_file(MODEL_INPUT))
    apply_initialization_policy(model, config["initialization_policy"])
    model = model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config["schedule"])
    return model, optimizer, scheduler, device


def _build_probe_manifest() -> dict[str, Any]:
    path = RUNTIME_ROOT / "manifests" / "fixed_probes.json"
    if path.is_file():
        payload = load_json(path)
        core = {key: value for key, value in payload.items() if key != "deterministic_content_hash"}
        if canonical_json_hash(core) != payload["deterministic_content_hash"]:
            raise ValueError("Phase 4D fixed-probe manifest changed")
        return payload
    tokenized = load_json(TOKENIZED_INPUT / "tokenized_corpus_manifest.json")
    complete = int(tokenized["statistics"]["split_tokens"]["train"]) // 512
    labels = sequence_labels(load_document_spans()["train"], complete)
    order = load_json(ORDER_INPUT)["indices"]

    def first(language: str | None = None, category: str | None = None, source: str | None = None) -> int:
        for index, (current_language, current_category, current_source) in enumerate(labels):
            if language is not None and current_language != language:
                continue
            if category is not None and current_category != category:
                continue
            if source is not None and current_source != source:
                continue
            return index
        raise ValueError(f"no fixed probe for {language}/{category}/{source}")

    probes: dict[str, dict[str, Any]] = {
        "turkish_prose": {"split": "train", "sequence_indices": [first("tr", "prose")]},
        "english_prose": {"split": "train", "sequence_indices": [first("en", "prose")]},
        "turkish_technical": {"split": "train", "sequence_indices": [first("tr", "technical")]},
        "english_technical": {"split": "train", "sequence_indices": [first("en", "technical")]},
        "training_distribution": {"split": "train", "sequence_indices": [int(value) for value in order[:16]]},
        "validation": {"split": "validation", "sequence_indices": list(range(16))},
        "eval": {"split": "eval", "sequence_indices": list(range(16))},
    }
    for source in sorted({item[2] for item in labels}):
        probes[f"source_{source}"] = {"split": "train", "sequence_indices": [first(source=source)]}
    datasets = {split: TokenShardDataset(TOKENIZED_INPUT, split) for split in ("train", "validation", "eval")}
    for probe in probes.values():
        digest = hashlib.sha256()
        for index in probe["sequence_indices"]:
            digest.update(datasets[probe["split"]].read(int(index) * 512, 512).tobytes())
        probe["token_sha256"] = digest.hexdigest()
        probe["tokens"] = len(probe["sequence_indices"]) * 512
    payload = {
        "schema_version": "darkmind-v2-phase4d-fixed-probes-v1",
        "sequence_length": 512,
        "probes": probes,
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    atomic_write_json(path, payload)
    return payload


def prepare() -> dict[str, Any]:
    authorization, _ = load_contract(requested_step=TARGET_STEP)
    storage = storage_preflight()
    failed_run = RUNTIME_ROOT / "runs" / "base_v1_stage2_25m_v2"
    if failed_run.is_dir() and (failed_run / "progress.json").is_file():
        progress = load_json(failed_run / "progress.json")
        atomic_write_json(
            RUNTIME_ROOT / "manifests" / "runtime_io_retry_evidence.json",
            {
                "schema_version": "darkmind-v2-phase4d-runtime-io-retry-v1",
                "result": "PRESERVED_FAILURE",
                "failed_run": str(failed_run),
                "failure": "transient Windows access denial replacing progress.json",
                "last_durable_progress": progress,
                "source_resume_checkpoint_modified": False,
                "failed_run_deleted": False,
                "failed_run_overwritten": False,
                "retry_run": str(RUN_DIR),
            },
        )
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        existing = load_json(RUN_DIR / "run_manifest.json")
        if existing.get("authorization_file_sha256") != sha256_file(AUTHORIZATION_PATH):
            raise FileExistsError("existing Phase 4D run uses another authorization")
        return existing
    for directory in ("checkpoints", "diagnostics", "resume", "audits"):
        (RUN_DIR / directory).mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics.jsonl").write_text("", encoding="utf-8")
    probes = _build_probe_manifest()
    source_summary = load_json(SOURCE_RUN / "training_summary.json")
    evaluations = {"610": source_summary["evaluations"]["610"]}
    atomic_write_json(RUN_DIR / "evaluations.json", evaluations)
    manifest = {
        "schema_version": "darkmind-v2-phase4d-stage2-run-v1",
        "status": "prepared",
        "authorization": authorization,
        "authorization_file_sha256": sha256_file(AUTHORIZATION_PATH),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "checkpoints": {"610": str(SOURCE_CHECKPOINT)},
        "checkpoint_hashes": {"610": EXPECTED_SOURCE_MODEL_HASH},
        "checkpoint_policy": {
            "610": "external_full_resume_preserved",
            "1024": "model_only",
            "1536": "model_only",
            "1831": "full_resume",
            "2048": "model_only",
            "2560": "model_only",
            "3051": "full_resume",
        },
        "diagnostic_snapshots": {"610": source_summary["diagnostic_snapshots"]["610"]},
        "latest_resume_checkpoint": str(SOURCE_CHECKPOINT),
        "best_validation_step": 610,
        "best_validation_checkpoint": str(SOURCE_CHECKPOINT),
        "sequence_order_hash": authorization["corpus_hashes"]["sequence_order"],
        "data_manifest_hash": authorization["corpus_hashes"]["tokenized_manifest_file"],
        "fixed_probe_hash": probes["deterministic_content_hash"],
        "process_ids": [],
        "segments": [],
        "runtime_outside_onedrive": True,
        "immutable_input_bytes_duplicated": 0,
        "phase_100m_authorized": False,
        "storage_preflight": storage,
    }
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": "prepared", "optimizer_step": 610, "tokens_consumed": SOURCE_TOKENS})
    return manifest


def disposable_resume_test(steps: int = 2) -> dict[str, Any]:
    if steps < 1 or steps > 4:
        raise ValueError("disposable resume diagnostic must use 1-4 steps")
    _, config = load_contract(requested_step=SOURCE_STEP + steps)
    preflight = load_json(RUNTIME_ROOT / "manifests" / "immutable_resume_preflight.json")
    if preflight.get("result") != "PASS":
        raise RuntimeError("immutable preflight must pass before disposable resume")
    copy_dir = RUNTIME_ROOT / "temporary" / "disposable_step610_checkpoint"
    if not copy_dir.exists():
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
        raise ValueError("disposable validation reproduction failed")
    if not math.isclose(evaluation["loss"], BASELINE_EVAL, abs_tol=1e-9):
        raise ValueError("disposable eval reproduction failed")
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
        records.append({"step": step, "applied_learning_rate": lr, "loss": metric["raw_train_loss"], "data_position": state.data_position})
    expected_position = SOURCE_TOKENS + steps * TOKENS_PER_STEP
    checks = {
        "validation_reproduced": True,
        "eval_reproduced": True,
        "rng_continuity": expected_rng == actual_rng,
        "scheduler_continuity": scheduler.last_epoch == SOURCE_STEP + steps,
        "data_position_continuity": state.data_position == expected_position,
        "next_sequence_index": state.data_position // 512 == SOURCE_TOKENS // 512 + steps * SEQUENCES_PER_STEP,
        "source_checkpoint_unchanged": sha256_file(SOURCE_CHECKPOINT / "resume_state.pt") == preflight["resume_state_file_sha256"],
    }
    payload = {
        "schema_version": "darkmind-v2-phase4d-disposable-resume-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "process_id": os.getpid(),
        "copied_checkpoint": str(copy_dir),
        "diagnostic_steps": steps,
        "validation": validation,
        "eval": evaluation,
        "records": records,
        "checks": checks,
        "official_checkpoint_modified": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "disposable_resume_test.json", payload)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    if payload["result"] != "PASS":
        raise RuntimeError("disposable Phase 4D resume test failed")
    return payload


def record_baseline_probes() -> dict[str, Any]:
    _, config = load_contract(requested_step=SOURCE_STEP)
    model, optimizer, scheduler, device = _create_model_stack(config)
    state = load_checkpoint(
        SOURCE_CHECKPOINT,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=tokenized_manifest_hash(TOKENIZED_INPUT),
        restore_rng=False,
    )
    _assert_resume_state(state, expected_step=SOURCE_STEP)
    probe_manifest = _build_probe_manifest()
    probes = {name: evaluate_probe(model, probe, device) for name, probe in probe_manifest["probes"].items()}
    evaluations = load_json(RUN_DIR / "evaluations.json")
    evaluations["610"]["probes"] = probes
    atomic_write_json(RUN_DIR / "evaluations.json", evaluations)
    payload = {
        "schema_version": "darkmind-v2-phase4d-step610-fixed-probe-baseline-v1",
        "result": "PASS",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_model_sha256": EXPECTED_SOURCE_MODEL_HASH,
        "fixed_probe_hash": probe_manifest["deterministic_content_hash"],
        "probes": probes,
        "source_checkpoint_modified": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "step610_fixed_probe_baseline.json", payload)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return payload


def validate_final_resume() -> dict[str, Any]:
    _, config = load_contract(requested_step=TARGET_STEP)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    checkpoint = Path(manifest["latest_resume_checkpoint"])
    if checkpoint != RUN_DIR / "checkpoints" / checkpoint_name(TARGET_STEP):
        raise ValueError("Phase 4D future resume checkpoint changed")
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
    _assert_resume_state(state, expected_step=TARGET_STEP)
    next_lr = learning_rate_for_policy(TARGET_STEP + 1, config["schedule"])
    checks = {
        "model_hash": sha256_file(checkpoint / "model" / "model.safetensors")
        == manifest["checkpoint_hashes"][str(TARGET_STEP)]["model_sha256"],
        "optimizer_loaded": bool(optimizer.state),
        "scheduler_epoch": scheduler.last_epoch == TARGET_STEP,
        "scheduler_next_lr": math.isclose(optimizer.param_groups[0]["lr"], next_lr, abs_tol=1e-15),
        "rng_continuity": rng_fingerprint(capture_rng_state()) == expected_rng,
        "data_position": state.data_position == TARGET_TOKENS,
        "next_sequence_index": state.data_position // 512 == 48_816,
        "step_3052_not_consumed": state.step == TARGET_STEP,
        "phase_100m_not_authorized": manifest["phase_100m_authorized"] is False,
    }
    payload = {
        "schema_version": "darkmind-v2-phase4d-future-resume-validation-v1",
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "resume_state_sha256": sha256_file(checkpoint / "resume_state.pt"),
        "optimizer_step": state.step,
        "tokens_consumed": state.tokens_seen,
        "next_sequence_index": state.data_position // 512,
        "next_learning_rate_if_separately_authorized": next_lr,
        "checks": checks,
        "step_3052_executed": False,
        "phase_100m_started": False,
    }
    atomic_write_json(RUN_DIR / "future_resume_validation.json", payload)
    del model, optimizer, scheduler, resume
    torch.cuda.empty_cache()
    if payload["result"] != "PASS":
        raise RuntimeError("Phase 4D future resume validation failed")
    return payload


def _sentinel_parameters(model: DarkMindV2ForCausalLM) -> dict[str, torch.nn.Parameter]:
    named = dict(model.named_parameters())
    preferred = (
        "token_embedding.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.mlp.proj.weight",
        f"blocks.{len(model.blocks) - 1}.attn.proj.weight",
        f"blocks.{len(model.blocks) - 1}.mlp.proj.weight",
        "final_norm.weight",
    )
    return {name: named[name] for name in preferred if name in named}


def training_step(
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
    indices = []
    data_wait = 0.0
    host_to_device = 0.0
    started = time.perf_counter()
    for micro_step in range(8):
        offset = data_position + micro_step * 1024
        wait_started = time.perf_counter()
        values = dataset.read(offset, 1024)
        indices.extend(dataset.source_indices(offset, 1024))
        data_wait += time.perf_counter() - wait_started
        transfer_started = time.perf_counter()
        batch = torch.from_numpy(values.astype(np.int64, copy=False)).to(device=device, dtype=torch.long).view(2, 512)
        host_to_device += time.perf_counter() - transfer_started
        if int(batch.min()) < 0 or int(batch.max()) >= 24_000:
            raise ValueError("token outside frozen vocabulary")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("non-finite Phase 4D training loss")
        (output.loss / 8).backward()
        losses.append(float(output.loss.detach()))
    sentinels = _sentinel_parameters(model)
    before_sentinel = {name: parameter.detach().clone() for name, parameter in sentinels.items()}
    before_all = {name: parameter.detach().clone() for name, parameter in model.named_parameters()} if diagnostic else {}
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite Phase 4D gradient norm")
    clipped = float(gradient_norm) > 1.0
    optimizer.step()
    torch.cuda.synchronize(device)
    update_ratios = []
    for name, parameter in sentinels.items():
        current = parameter.detach().float()
        delta = (parameter.detach() - before_sentinel[name]).float()
        update_ratios.append(float(delta.square().mean().sqrt() / max(float(current.square().mean().sqrt()), 1e-12)))
    if any(not math.isfinite(value) for value in update_ratios):
        raise FloatingPointError("non-finite Phase 4D update ratio")
    duration = time.perf_counter() - started
    return {
        "raw_train_loss": statistics.fmean(losses),
        "gradient_norm": float(gradient_norm),
        "clipped": clipped,
        "optimizer_step_duration_seconds": duration,
        "active_tokens_per_second": TOKENS_PER_STEP / duration,
        "data_loader_wait_seconds": data_wait,
        "host_to_device_seconds": host_to_device,
        "source_sequence_indices": indices,
        "update_to_weight": {
            "minimum": min(update_ratios),
            "mean": statistics.fmean(update_ratios),
            "maximum": max(update_ratios),
            "sentinel_parameters": list(sentinels),
        },
        "parameter_diagnostics": _parameter_diagnostics(model, optimizer, before_all) if diagnostic else [],
    }


def _rename_checkpoint(temporary: Path, checkpoint: Path) -> None:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite Phase 4D checkpoint: {checkpoint}")
    for attempt in range(20):
        try:
            os.replace(temporary, checkpoint)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def _validate_safetensors(path: Path) -> None:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as weights:
        if not weights.keys():
            raise ValueError("empty Phase 4D safetensors checkpoint")


def save_milestone(
    *,
    step: int,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    scheduler: Any,
    state: TrainingState,
    config: dict[str, Any],
    data_hash: str,
    order_hash: str,
    authorization_hash: str,
) -> tuple[Path, dict[str, Any]]:
    if step not in MODEL_ONLY_STEPS | FULL_RESUME_STEPS:
        raise ValueError(f"unsupported Phase 4D checkpoint step: {step}")
    checkpoint = ensure_runtime_path(RUN_DIR / "checkpoints" / checkpoint_name(step))
    temporary = checkpoint.with_name(f".{checkpoint.name}.incomplete")
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(f"Phase 4D milestone already exists: {checkpoint}")
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
        model_files = save_model_package(model, temporary / "model")
        metadata = {"model_files": model_files, "training_state": state.to_dict()}
        kind = "model_only"
    stage_metadata = {
        "schema_version": "darkmind-v2-phase4d-stage2-checkpoint-v1",
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
        "sequence_order_hash": order_hash,
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "authorization_file_sha256": authorization_hash,
        "resume_capable": kind == "full_resume",
        "scheduler_reset": False,
        "optimizer_reset": False,
        "rng_reset": False,
        "data_order_reset": False,
        "next_learning_rate": optimizer.param_groups[0]["lr"],
    }
    atomic_write_json(temporary / "stage2_checkpoint_metadata.json", stage_metadata)
    _validate_safetensors(temporary / "model" / "model.safetensors")
    _rename_checkpoint(temporary, checkpoint)
    hashes = {
        "model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "stage2_metadata_sha256": sha256_file(checkpoint / "stage2_checkpoint_metadata.json"),
    }
    if kind == "full_resume":
        hashes["resume_state_sha256"] = sha256_file(checkpoint / "resume_state.pt")
        hashes["checkpoint_metadata_sha256"] = sha256_file(checkpoint / "checkpoint_metadata.json")
    return checkpoint, {**stage_metadata, "hashes": hashes}


def _assert_resume_state(state: TrainingState, *, expected_step: int) -> None:
    expected_tokens = expected_step * TOKENS_PER_STEP
    if state.step != expected_step or state.tokens_seen != expected_tokens or state.data_position != expected_tokens:
        raise ValueError("Phase 4D resume step/token/data-position mismatch")
    validate_authorization(load_json(AUTHORIZATION_PATH), requested_step=expected_step, requested_tokens=expected_tokens)


def train_segment(target_step: int) -> dict[str, Any]:
    if target_step not in (1831, 3051):
        raise ValueError("Phase 4D segment target must be 1831 or 3051")
    authorization, config = load_contract(requested_step=target_step)
    manifest = load_json(RUN_DIR / "run_manifest.json")
    disposable = load_json(RUNTIME_ROOT / "manifests" / "disposable_resume_test.json")
    if disposable.get("result") != "PASS":
        raise RuntimeError("disposable resume must pass before official training")
    expected_start = SOURCE_STEP if target_step == 1831 else 1831
    if target_step == 1831 and manifest["segments"]:
        raise ValueError("Phase 4D Segment A already completed")
    if target_step == 3051 and len(manifest["segments"]) != 1:
        raise ValueError("Phase 4D Segment B requires exactly one completed segment")
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
    _assert_resume_state(state, expected_step=expected_start)
    actual_rng = rng_fingerprint(capture_rng_state())
    if actual_rng != expected_rng:
        raise ValueError("Phase 4D RNG continuity failed")
    if scheduler.last_epoch != expected_start:
        raise ValueError("Phase 4D scheduler epoch reset")
    expected_lr = learning_rate_for_policy(expected_start + 1, config["schedule"])
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, abs_tol=1e-15):
        raise ValueError("Phase 4D next applied LR mismatch")
    if os.getpid() in manifest["process_ids"]:
        raise ValueError("Phase 4D segment did not use a fresh process")
    if target_step == 3051:
        validation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "validation"), sequence_length=512, micro_batch_size=2, device=device)
        evaluation = evaluate_loss(model, TokenShardDataset(TOKENIZED_INPUT, "eval"), sequence_length=512, micro_batch_size=2, device=device)
        prior = load_json(RUN_DIR / "evaluations.json")["1831"]
        if not math.isclose(validation["loss"], prior["validation"]["loss"], abs_tol=1e-9):
            raise ValueError("step-1831 validation changed after process restart")
        if not math.isclose(evaluation["loss"], prior["eval"]["loss"], abs_tol=1e-9):
            raise ValueError("step-1831 eval changed after process restart")
    manifest["process_ids"].append(os.getpid())
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)

    order_data = OrderedTokenDataset(TOKENIZED_INPUT, ORDER_INPUT)
    if order_data.order_hash != manifest["sequence_order_hash"] or order_data.total_tokens <= TARGET_TOKENS:
        raise ValueError("Phase 4D sequence order cannot cover target without wrap")
    validation_data = TokenShardDataset(TOKENIZED_INPUT, "validation")
    eval_data = TokenShardDataset(TOKENIZED_INPUT, "eval")
    train_base = TokenShardDataset(TOKENIZED_INPUT, "train")
    probes = _build_probe_manifest()
    activation_index = int(probes["probes"]["training_distribution"]["sequence_indices"][0])
    activation_values = train_base.read(activation_index * 512, 128)
    activation_tokens = torch.from_numpy(activation_values.astype(np.int64, copy=False)).view(1, 128).to(device)
    evaluations = load_json(RUN_DIR / "evaluations.json")
    prior_metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    clipped_steps = sum(int(item["clipped"]) for item in prior_metrics)
    monitor = GpuMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    start_position = state.data_position
    expected_sequence_cursor = state.data_position // 512
    try:
        while state.step < target_step:
            step = state.step + 1
            validate_authorization(authorization, requested_step=step, requested_tokens=step * TOKENS_PER_STEP)
            if step > TARGET_STEP:
                raise PermissionError("refusing to consume optimizer step 3052")
            learning_rate = optimizer.param_groups[0]["lr"]
            diagnostic = step in MILESTONES
            metric = training_step(
                model,
                optimizer,
                order_data,
                data_position=state.data_position,
                diagnostic=diagnostic,
                device=device,
            )
            expected_indices = order_data.order[expected_sequence_cursor : expected_sequence_cursor + SEQUENCES_PER_STEP]
            if metric["source_sequence_indices"] != expected_indices:
                raise ValueError("Phase 4D repeated or skipped sequence")
            expected_sequence_cursor += SEQUENCES_PER_STEP
            scheduler.step()
            state.step = step
            state.tokens_seen += TOKENS_PER_STEP
            state.data_position += TOKENS_PER_STEP
            _assert_resume_state(state, expected_step=step)
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
                    "sequence_index": state.data_position // 512,
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
                diagnostic_path = RUN_DIR / "diagnostics" / f"step_{step:06d}.json"
                atomic_write_json(diagnostic_path, diagnostics)
                state.last_validation_loss = validation["loss"]
                prospective_checkpoint = RUN_DIR / "checkpoints" / checkpoint_name(step)
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = validation["loss"]
                    state.best_checkpoint = str(prospective_checkpoint)
                    manifest["best_validation_step"] = step
                    manifest["best_validation_checkpoint"] = str(prospective_checkpoint)
                saved, checkpoint_metadata = save_milestone(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    config=config,
                    data_hash=manifest["data_manifest_hash"],
                    order_hash=manifest["sequence_order_hash"],
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
            atomic_write_json(RUN_DIR / "progress.json", {"status": "training", "optimizer_step": step, "tokens_consumed": state.tokens_seen})
            if step % 10 == 0 or diagnostic:
                print(
                    f"phase4d step={step} loss={metric['raw_train_loss']:.6f} "
                    f"lr={learning_rate:.9f} tok_s={metric['active_tokens_per_second']:.1f}",
                    flush=True,
                )
    finally:
        monitor.close()
    if state.step != target_step or state.tokens_seen != target_step * TOKENS_PER_STEP:
        raise RuntimeError("Phase 4D segment did not stop at its exact gate")
    segment = {
        "schema_version": "darkmind-v2-phase4d-stage2-segment-v1",
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
        "no_data_wrap": state.data_position < order_data.total_tokens,
        "next_optimizer_step": target_step + 1 if target_step < TARGET_STEP else None,
        "next_sequence_index": state.data_position // 512,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_samples": monitor.samples,
        "latest_resume_checkpoint": manifest["latest_resume_checkpoint"],
    }
    manifest["segments"].append(segment)
    manifest["status"] = "segment_a_complete" if target_step == 1831 else "training_complete"
    atomic_write_json(RUN_DIR / "resume" / f"segment_to_step_{target_step:06d}.json", segment)
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "progress.json", {"status": manifest["status"], "optimizer_step": target_step, "tokens_consumed": state.tokens_seen})
    if target_step == TARGET_STEP:
        finalize_training_summary(manifest, config)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return segment


def _probe_regression(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in final.items():
        if name not in baseline:
            continue
        start = float(baseline[name]["loss"])
        end = float(value["loss"])
        result[name] = {
            "baseline_loss": start,
            "final_loss": end,
            "improvement_percent": (start - end) * 100.0 / start,
            "catastrophic_regression": end > start * 1.20,
        }
    return result


def classify_stage2(summary: dict[str, Any]) -> str:
    if not summary["integrity_pass"]:
        return "FAIL"
    val = summary["validation_improvement_percent"]
    evl = summary["eval_improvement_percent"]
    rebound = max(summary["validation_rebound_percent"], summary["eval_rebound_percent"])
    catastrophic = any(item["catastrophic_regression"] for item in summary["probe_regressions"].values())
    late = summary["last_three_validation_sustained_worsening"] or summary["last_three_eval_sustained_worsening"]
    if val < 2.0 or evl < 2.0 or summary["final_validation_loss"] > BASELINE_VALIDATION or summary["final_eval_loss"] > BASELINE_EVAL or rebound > 5.0 or catastrophic or late:
        return "FAIL"
    if val >= 8.0 and evl >= 8.0 and rebound <= 2.0:
        return "STRONG PASS"
    return "CONDITIONAL PASS"


def finalize_training_summary(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evaluations = load_json(RUN_DIR / "evaluations.json")
    metrics = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(metrics) != TARGET_STEP - SOURCE_STEP:
        raise ValueError("Phase 4D telemetry does not contain exactly 2,441 steps")
    if metrics[-1]["optimizer_step"] != TARGET_STEP or metrics[-1]["tokens_consumed"] != TARGET_TOKENS:
        raise ValueError("Phase 4D telemetry exceeded or missed the authorization gate")
    validation_losses = [float(evaluations[str(step)]["validation"]["loss"]) for step in MILESTONES]
    eval_losses = [float(evaluations[str(step)]["eval"]["loss"]) for step in MILESTONES]
    probe_regressions = _probe_regression(evaluations["610"]["probes"], evaluations["3051"]["probes"])
    summary = {
        "schema_version": "darkmind-v2-phase4d-stage2-training-summary-v1",
        "integrity_pass": True,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "v2_config_sha256": EXPECTED_V2_FILE_HASH,
        "authorization_file_sha256": manifest["authorization_file_sha256"],
        "starting_optimizer_step": SOURCE_STEP,
        "final_optimizer_step": TARGET_STEP,
        "starting_tokens": SOURCE_TOKENS,
        "final_tokens": TARGET_TOKENS,
        "additional_optimizer_steps": TARGET_STEP - SOURCE_STEP,
        "additional_tokens": TARGET_TOKENS - SOURCE_TOKENS,
        "additional_sequences": (TARGET_TOKENS - SOURCE_TOKENS) // 512,
        "milestones": list(MILESTONES),
        "initial_validation_loss": validation_losses[0],
        "final_validation_loss": validation_losses[-1],
        "best_validation_loss": min(validation_losses),
        "best_validation_step": min(MILESTONES, key=lambda step: evaluations[str(step)]["validation"]["loss"]),
        "initial_eval_loss": eval_losses[0],
        "final_eval_loss": eval_losses[-1],
        "best_eval_loss": min(eval_losses),
        "best_eval_step": min(MILESTONES, key=lambda step: evaluations[str(step)]["eval"]["loss"]),
        "validation_improvement_percent": (validation_losses[0] - validation_losses[-1]) * 100.0 / validation_losses[0],
        "eval_improvement_percent": (eval_losses[0] - eval_losses[-1]) * 100.0 / eval_losses[0],
        "validation_rebound_percent": rebound_percent(min(validation_losses), validation_losses[-1]),
        "eval_rebound_percent": rebound_percent(min(eval_losses), eval_losses[-1]),
        "last_three_validation_sustained_worsening": all(right > left for left, right in zip(validation_losses[-3:], validation_losses[-2:])),
        "last_three_eval_sustained_worsening": all(right > left for left, right in zip(eval_losses[-3:], eval_losses[-2:])),
        "consecutive_worsening_validation_evaluations": _max_consecutive_worsening(validation_losses),
        "consecutive_worsening_eval_evaluations": _max_consecutive_worsening(eval_losses),
        "final_perplexity": evaluations["3051"]["eval"]["perplexity"],
        "validation_progression": {str(step): evaluations[str(step)]["validation"] for step in MILESTONES},
        "eval_progression": {str(step): evaluations[str(step)]["eval"] for step in MILESTONES},
        "train_loss_progression": {str(step): evaluations[str(step)]["train"] for step in MILESTONES if "train" in evaluations[str(step)]},
        "learning_rate_progression": {str(step): learning_rate_for_policy(step, config["schedule"]) for step in MILESTONES},
        "gradient_norm_p50": percentile([item["gradient_norm"] for item in metrics], 0.50),
        "gradient_norm_p95": percentile([item["gradient_norm"] for item in metrics], 0.95),
        "gradient_norm_max": max(item["gradient_norm"] for item in metrics),
        "clipped_step_fraction": sum(int(item["clipped"]) for item in metrics) / len(metrics),
        "update_to_weight_maximum": max(item["update_to_weight"]["maximum"] for item in metrics),
        "non_finite_events": sum(item["non_finite_event_count"] for item in metrics),
        "probe_regressions": probe_regressions,
        "evaluations": evaluations,
        "diagnostic_snapshots": manifest["diagnostic_snapshots"],
        "checkpoints": manifest["checkpoints"],
        "checkpoint_hashes": manifest["checkpoint_hashes"],
        "best_checkpoint": manifest["best_validation_checkpoint"],
        "active_tokens_per_second": (TARGET_TOKENS - SOURCE_TOKENS) / sum(item["optimizer_step_duration_seconds"] for item in metrics),
        "wall_tokens_per_second": (TARGET_TOKENS - SOURCE_TOKENS) / sum(segment["elapsed_seconds"] for segment in manifest["segments"]),
        "peak_allocated_vram_bytes": max(segment["peak_allocated_bytes"] for segment in manifest["segments"]),
        "peak_reserved_vram_bytes": max(segment["peak_reserved_bytes"] for segment in manifest["segments"]),
        "process_restart": {
            "result": "PASS",
            "fresh_processes": len(manifest["process_ids"]) == 2 and len(set(manifest["process_ids"])) == 2,
            "segment_boundary_exact": [segment["segment_token_range"] for segment in manifest["segments"]] == [[4_997_120, 14_999_552], [14_999_552, 24_993_792]],
            "optimizer_continuity": all(segment["optimizer_continuity"] for segment in manifest["segments"]),
            "scheduler_continuity": all(segment["scheduler_continuity"] for segment in manifest["segments"]),
            "rng_continuity": all(segment["rng_continuity"] for segment in manifest["segments"]),
            "data_position_continuity": all(segment["data_position_continuity"] for segment in manifest["segments"]),
            "no_repeated_or_skipped_sequence": all(segment["no_repeated_or_skipped_sequence"] for segment in manifest["segments"]),
        },
        "phase_100m_authorized": False,
    }
    if not all(value for key, value in summary["process_restart"].items() if key != "result"):
        summary["integrity_pass"] = False
        summary["process_restart"]["result"] = "FAIL"
    summary["classification"] = classify_stage2(summary)
    summary["recommend_100m_approval_request"] = summary["classification"] == "STRONG PASS"
    atomic_write_json(RUN_DIR / "training_summary.json", summary)
    manifest["classification"] = summary["classification"]
    manifest["status"] = "training_complete"
    atomic_write_json(RUN_DIR / "run_manifest.json", manifest)
    atomic_write_json(RUN_DIR / "process_restart_validation.json", summary["process_restart"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("preflight")
    commands.add_parser("baseline-probes")
    commands.add_parser("validate-final-resume")
    disposable = commands.add_parser("disposable-resume")
    disposable.add_argument("--steps", type=int, default=2)
    segment = commands.add_parser("segment")
    segment.add_argument("--target-step", type=int, choices=(1831, 3051), required=True)
    args = parser.parse_args()
    functions = {
        "prepare": prepare,
        "preflight": immutable_preflight,
        "baseline-probes": record_baseline_probes,
        "validate-final-resume": validate_final_resume,
        "disposable-resume": lambda: disposable_resume_test(args.steps),
        "segment": lambda: train_segment(args.target_step),
    }
    payload = functions[args.command]()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
