"""Deterministic two-process Phase 2C tiny full-epoch capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.data_pipeline.validate_full_tokenized_corpus import validate_full_tokenized_corpus
from darkmind_v2.evaluation.generate_fixed_prompts import generate_fixed_prompts, load_prompts
from darkmind_v2.evaluation.validate_generation_health import (
    classify_generation_health,
    enforce_generation_policy,
)
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import (
    DEFAULT_FROZEN_DIR,
    EXPECTED_HASHES,
    FrozenTokenizer,
    sha256_file,
    verify_frozen_tokenizer,
)
from darkmind_v2.training.checkpointing import load_checkpoint, save_checkpoint
from darkmind_v2.training.probe_training_environment import probe_environment
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, dataset_summary, tokenized_manifest_hash
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_full_epoch_config import (
    EXPECTED_MILESTONES,
    learning_rate_for_step,
    load_and_validate_full_epoch_config,
)


EXPECTED_MANIFEST_CONTENT_HASH = "23e92169ae6ef3b0b0f11c4d0ca327ef60d59b0d8b697a1c4261218a233cce28"
EXPECTED_BOUNDARY_HASH = "b15ab6da0c30a8c7da53475c4435d1b0640828817381c15c35e83bde8387cb5a"
EXPECTED_VALIDATION_HASH = "999bc95f53559e8a9f16aed40b1e9ac2678e49577aae6a4fd6298dfaf16bb25b"
EXPECTED_SHARD_CHECKSUMS_HASH = "d674d38cf5bf9f7e86d8c752237cd66bb4d02c9ad1599091c05ed128decd255d"
EXPECTED_SPLIT_TOKENS = {"train": 11_744_226, "validation": 654_032, "eval": 651_956}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_name(step: int, tokens: int) -> str:
    return f"step_{step:06d}_tokens_{tokens:09d}"


def build_optimizer(model: DarkMindV2ForCausalLM, config: dict[str, Any]) -> torch.optim.AdamW:
    values = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=values["peak_learning_rate"],
        betas=(values["beta1"], values["beta2"]),
        weight_decay=values["weight_decay"],
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    peak = config["optimizer"]["peak_learning_rate"]
    total = config["maximum_optimizer_steps"]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_for_step(min(epoch + 1, total), config) / peak,
    )


@torch.no_grad()
def evaluate_loss(
    model: DarkMindV2ForCausalLM,
    dataset: TokenShardDataset,
    *,
    sequence_length: int,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    sequences = dataset.total_tokens // sequence_length
    if sequences <= 0:
        raise ValueError("evaluation split has no complete sequences")
    model.eval()
    weighted_loss = 0.0
    completed = 0
    while completed < sequences:
        batch_size = min(micro_batch_size, sequences - completed)
        batch = dataset.batch(
            offset=completed * sequence_length,
            micro_batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("non-finite evaluation loss")
        weighted_loss += float(output.loss) * batch_size
        completed += batch_size
    model.train()
    loss = weighted_loss / sequences
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 80.0)),
        "sequences": sequences,
        "evaluated_tokens": sequences * sequence_length,
        "excluded_split_tail_tokens": dataset.total_tokens - sequences * sequence_length,
    }


def generation_snapshot(
    model: DarkMindV2ForCausalLM,
    tokenizer: FrozenTokenizer,
    *,
    output_path: Path,
    max_new_tokens: int,
    seed: int,
    checkpoint_stage: str,
) -> dict[str, Any]:
    prompts = load_prompts(Path("darkmind_v2/eval/fixed_base_prompts.jsonl"))
    snapshot = generate_fixed_prompts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=seed,
    )
    warning_counts: dict[str, int] = {}
    for item in snapshot["results"]:
        policy = classify_generation_health(
            item["generation"],
            item["token_ids"],
            checkpoint_stage=checkpoint_stage,
            maximum_repetition_ratio=item["health"]["maximum_repetition_ratio"],
            token_trace=item.get("token_trace"),
        )
        item["policy"] = policy
        for warning in policy["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    enforce_generation_policy(snapshot["results"])
    atomic_write_json(output_path, snapshot)
    return {
        "path": str(output_path),
        "content_hash": snapshot["deterministic_content_hash"],
        "prompts": len(snapshot["results"]),
        "hard_failures": 0,
        "warning_counts": dict(sorted(warning_counts.items())),
        "checkpoint_stage": checkpoint_stage,
    }


def git_state() -> dict[str, Any]:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "status_short": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "diff_check": subprocess.run(["git", "diff", "--check"], text=True, capture_output=True).stdout.splitlines(),
    }


def _range_provenance(manifest: dict[str, Any], split: str, offset: int, count: int) -> list[dict[str, Any]]:
    remaining = count
    cursor = offset
    split_start = 0
    records = []
    for shard in (item for item in manifest["shards"] if item["split"] == split):
        shard_end = split_start + int(shard["tokens"])
        if cursor < shard_end and remaining:
            local_start = max(0, cursor - split_start)
            take = min(remaining, int(shard["tokens"]) - local_start)
            records.append({
                "filename": shard["filename"],
                "shard_sha256": shard["sha256"],
                "local_token_start": local_start,
                "local_token_end_exclusive": local_start + take,
                "tokens": take,
            })
            cursor += take
            remaining -= take
        split_start = shard_end
    if remaining:
        raise ValueError("failed to map excluded-tail provenance")
    return records


def data_manifests(tokenized_dir: Path, train_data: TokenShardDataset, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = tokenized_dir / "tokenized_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = dataset_summary(tokenized_dir)
    if manifest["deterministic_content_hash"] != EXPECTED_MANIFEST_CONTENT_HASH:
        raise ValueError("tokenized deterministic content hash mismatch")
    if manifest["statistics"]["split_tokens"] != EXPECTED_SPLIT_TOKENS:
        raise ValueError("tokenized split counts changed")
    if summary["shards"] != 8:
        raise ValueError("tokenized shard count changed")
    tail_start = config["maximum_total_training_tokens"]
    tail_count = config["data"]["excluded_tail_tokens"]
    tail = train_data.read(tail_start, tail_count).astype("<u2", copy=False).tobytes()
    ordering = {
        "schema_version": "darkmind-v2-phase2c-sequence-order-v1",
        "split": "train",
        "data_order_seed": config["data"]["data_order_seed"],
        "strategy": "contiguous_no_shuffle",
        "first_token_offset": 0,
        "last_token_offset_exclusive": config["maximum_total_training_tokens"],
        "sequence_length": config["data"]["sequence_length"],
        "sequences_per_optimizer_step": 16,
        "optimizer_steps": config["maximum_optimizer_steps"],
        "sequences_consumed": config["maximum_optimizer_steps"] * 16,
        "tokens_consumed": config["maximum_total_training_tokens"],
        "sample_replacement": False,
        "data_repetition": False,
        "wraparound": False,
    }
    excluded = {
        "schema_version": "darkmind-v2-phase2c-excluded-tail-v1",
        "split": "train",
        "token_start": tail_start,
        "token_end_exclusive": tail_start + tail_count,
        "tokens": tail_count,
        "sha256_uint16_le": hashlib.sha256(tail).hexdigest(),
        "provenance": _range_provenance(manifest, "train", tail_start, tail_count),
    }
    return ordering, excluded


def verify_checkpoint_reload(
    checkpoint_dir: Path,
    *,
    config: dict[str, Any],
    model_config: DarkMindV2Config,
    tokenizer_hash: str,
    data_manifest_hash: str,
) -> TrainingState:
    shadow_model = DarkMindV2ForCausalLM(model_config)
    shadow_optimizer = build_optimizer(shadow_model, config)
    shadow_scheduler = build_scheduler(shadow_optimizer, config)
    state = load_checkpoint(
        checkpoint_dir,
        model=shadow_model,
        optimizer=shadow_optimizer,
        scheduler=shadow_scheduler,
        expected_tokenizer_hash=tokenizer_hash,
        expected_data_manifest_hash=data_manifest_hash,
        restore_rng=False,
    )
    metadata = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    actual_hash = sha256_file(checkpoint_dir / "model" / "model.safetensors")
    if actual_hash != metadata["model_files"]["model_sha256"]:
        raise ValueError(f"checkpoint model hash mismatch: {checkpoint_dir}")
    return state


def initialize_run(config_path: Path, calibration_path: Path) -> dict[str, Any]:
    config = load_and_validate_full_epoch_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable Phase 2C run: {run_dir}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("result") != "PASS" or not calibration.get("disposable_model"):
        raise ValueError("disposable calibration did not pass")
    profile = calibration["selected_profile"]
    if profile["micro_batch_size"] * profile["gradient_accumulation_steps"] * 256 != 4096:
        raise ValueError("calibration changed the effective optimizer batch")

    verify_frozen_tokenizer()
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    corpus_validation = validate_full_tokenized_corpus(
        tokenized_dir,
        Path("darkmind_v2/data/phase1b/processed"),
    )
    expected_validation = {
        "result": "PASS",
        "manifest_hash": EXPECTED_MANIFEST_CONTENT_HASH,
        "boundaries_hash": EXPECTED_BOUNDARY_HASH,
        "validation_content_hash": EXPECTED_VALIDATION_HASH,
        "shard_checksums_hash": EXPECTED_SHARD_CHECKSUMS_HASH,
    }
    if corpus_validation.get("failures") or any(
        corpus_validation.get(key) != value for key, value in expected_validation.items()
    ):
        raise ValueError("pre-run tokenized corpus validation mismatch")
    train_data = TokenShardDataset(tokenized_dir, "train")
    validation_data = TokenShardDataset(tokenized_dir, "validation")
    eval_data = TokenShardDataset(tokenized_dir, "eval")
    if train_data.total_tokens != config["data"]["train_corpus_tokens"]:
        raise ValueError("train token count changed")
    ordering, excluded = data_manifests(tokenized_dir, train_data, config)
    data_summary = dataset_summary(tokenized_dir)
    model_config = DarkMindV2Config.from_json_file(config["model_config"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2C initialization requires CUDA BF16")

    set_deterministic_seed(config["initialization_seed"])
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(model_config).to(device)
    if not model.embeddings_are_tied() or model.parameter_count() != 9_369_088:
        raise ValueError("tiny model architecture contract changed")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    state = TrainingState()
    initial_validation = evaluate_loss(
        model,
        validation_data,
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    initial_eval = evaluate_loss(
        model,
        eval_data,
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    state.last_validation_loss = float(initial_validation["loss"])

    run_dir.mkdir(parents=True)
    for name in ("checkpoints", "validations", "eval", "generations", "evaluations", "manifests"):
        (run_dir / name).mkdir()
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(run_dir / "manifests" / "sequence_ordering.json", ordering)
    atomic_write_json(run_dir / "manifests" / "excluded_tail.json", excluded)
    atomic_write_json(run_dir / "manifests" / "tokenized_corpus_validation.json", corpus_validation)
    generation = generation_snapshot(
        model,
        FrozenTokenizer(),
        output_path=run_dir / "generations" / "step_000000.json",
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
        seed=config["initialization_seed"],
        checkpoint_stage="initialization",
    )
    checkpoint = run_dir / "checkpoints" / checkpoint_name(0, 0)
    checkpoint_metadata = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_summary["manifest_hash"],
    )
    reloaded = verify_checkpoint_reload(
        checkpoint,
        config=config,
        model_config=model_config,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_summary["manifest_hash"],
    )
    if reloaded.step != 0 or reloaded.data_position != 0:
        raise ValueError("initial checkpoint reload state mismatch")

    atomic_write_json(run_dir / "resolved_model_config.json", model_config.architecture_dict())
    atomic_write_json(run_dir / "resolved_training_config.json", {**config, "selected_profile": profile})
    atomic_write_json(run_dir / "environment.json", probe_environment(run_dir))
    atomic_write_json(run_dir / "git_state.json", git_state())
    atomic_write_json(run_dir / "validations" / "step_000000.json", initial_validation)
    atomic_write_json(run_dir / "eval" / "step_000000.json", initial_eval)
    append_jsonl(run_dir / "metrics.jsonl", {
        "event": "initialization",
        "optimizer_step": 0,
        "consumed_tokens": 0,
        "data_position": 0,
        "validation_loss": initial_validation["loss"],
        "eval_loss": initial_eval["loss"],
        "learning_rate_for_next_step": optimizer.param_groups[0]["lr"],
    })
    source_manifest = Path("darkmind_v2/data/phase1b/processed/corpus_manifest.json")
    run_manifest = {
        "schema_version": "darkmind-v2-phase2c-full-epoch-run-v1",
        "run_name": config["run"]["name"],
        "status": "initialized",
        "description": "Clean from-scratch tiny-model capacity diagnostic",
        "stage1_weights_reused": False,
        "initialization_seed": config["initialization_seed"],
        "data_order_seed": config["data"]["data_order_seed"],
        "instruction_tuned": False,
        "chat_model": False,
        "public_release_approved": False,
        "huggingface_upload_authorized": False,
        "target_optimizer_steps": config["maximum_optimizer_steps"],
        "target_training_tokens": config["maximum_total_training_tokens"],
        "train_corpus_tokens": config["data"]["train_corpus_tokens"],
        "excluded_tail_tokens": config["data"]["excluded_tail_tokens"],
        "coverage_percent": 100 * config["maximum_total_training_tokens"] / config["data"]["train_corpus_tokens"],
        "selected_profile": profile,
        "model_parameters": model.parameter_count(),
        "model_config_hash": model_config_hash(model_config),
        "model_initialization_sha256": checkpoint_metadata["model_files"]["model_sha256"],
        "tokenizer_hashes": EXPECTED_HASHES,
        "tokenized_corpus_manifest_sha256": data_summary["manifest_hash"],
        "tokenized_corpus_content_hash": data_summary["manifest_content_hash"],
        "source_corpus_manifest_sha256": sha256_file(source_manifest),
        "sequence_ordering_manifest_sha256": sha256_file(run_dir / "manifests" / "sequence_ordering.json"),
        "excluded_tail_manifest_sha256": sha256_file(run_dir / "manifests" / "excluded_tail.json"),
        "tokenized_corpus_validation_sha256": sha256_file(
            run_dir / "manifests" / "tokenized_corpus_validation.json"
        ),
        "initial_checkpoint": str(checkpoint),
        "initial_validation": initial_validation,
        "initial_eval": initial_eval,
        "initial_generation": generation,
        "best_checkpoint": str(checkpoint),
        "best_validation_loss": initial_validation["loss"],
        "optimizer_steps": 0,
        "consumed_tokens": 0,
        "data_position": 0,
    }
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest


def _checkpoint_stage(step: int, config: dict[str, Any]) -> str:
    if step == 0:
        return "initialization"
    if step == config["segment_a_target_step"]:
        return "midpoint"
    if step == config["maximum_optimizer_steps"]:
        return "stage1_final"
    return "stage1"


def train_segment(config_path: Path, checkpoint_dir: Path, *, target_step: int) -> dict[str, Any]:
    config = load_and_validate_full_epoch_config(config_path)
    if target_step not in {config["segment_a_target_step"], config["segment_b_target_step"]}:
        raise ValueError("target step is outside the approved Phase 2C segment boundaries")
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    data_hash = tokenized_manifest_hash(tokenized_dir)
    if data_hash != run_manifest["tokenized_corpus_manifest_sha256"]:
        raise ValueError("tokenized corpus manifest changed after initialization")
    verify_frozen_tokenizer()
    model_config = DarkMindV2Config.from_json_file(config["model_config"])
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(model_config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    state = load_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=data_hash,
    )
    tokens_per_step = config["data"]["effective_tokens_per_optimizer_step"]
    if state.step * tokens_per_step != state.tokens_seen or state.data_position != state.tokens_seen:
        raise ValueError("checkpoint step/token/data position are inconsistent")
    if state.step >= target_step:
        raise ValueError("checkpoint already reached the segment target")
    expected_lr = learning_rate_for_step(state.step + 1, config)
    if not math.isclose(optimizer.param_groups[0]["lr"], expected_lr, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("scheduler continuity failure before segment start")
    if scheduler.last_epoch != state.step:
        raise ValueError("scheduler epoch does not match optimizer step")

    profile = run_manifest["selected_profile"]
    train_data = TokenShardDataset(tokenized_dir, "train")
    validation_data = TokenShardDataset(tokenized_dir, "validation")
    eval_data = TokenShardDataset(tokenized_dir, "eval")
    micro_batch = profile["micro_batch_size"]
    accumulation = profile["gradient_accumulation_steps"]
    sequence_length = config["data"]["sequence_length"]
    tokenizer = FrozenTokenizer()
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    torch.cuda.reset_peak_memory_stats(device)
    segment_started = time.perf_counter()
    segment_start_step = state.step
    last_checkpoint = checkpoint_dir
    try:
        model.train()
        while state.step < target_step:
            if interrupted:
                state.interrupted = True
                raise KeyboardInterrupt("safe interruption requested before next optimizer step")
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            intended_step = state.step + 1
            learning_rate = optimizer.param_groups[0]["lr"]
            if not math.isclose(learning_rate, learning_rate_for_step(intended_step, config), abs_tol=1e-15):
                raise ValueError(f"scheduler continuity failure at optimizer step {intended_step}")
            micro_losses = []
            for micro_step in range(accumulation):
                offset = state.data_position + micro_step * micro_batch * sequence_length
                batch = train_data.batch(
                    offset=offset,
                    micro_batch_size=micro_batch,
                    sequence_length=sequence_length,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch, labels=batch)
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise FloatingPointError("non-finite Phase 2C training loss")
                    loss = output.loss / accumulation
                loss.backward()
                micro_losses.append(float(output.loss.detach()))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"]))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite Phase 2C gradient norm")
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            state.step += 1
            state.tokens_seen += tokens_per_step
            state.data_position += tokens_per_step
            if state.data_position > config["maximum_total_training_tokens"]:
                raise ValueError("unplanned data wraparound")
            raw_loss = sum(micro_losses) / len(micro_losses)
            state.last_training_loss = raw_loss
            state.smoothed_training_loss = (
                raw_loss if state.smoothed_training_loss is None
                else 0.9 * state.smoothed_training_loss + 0.1 * raw_loss
            )
            elapsed = time.perf_counter() - step_started
            metric: dict[str, Any] = {
                "event": "optimizer_step",
                "optimizer_step": state.step,
                "consumed_tokens": state.tokens_seen,
                "data_position": state.data_position,
                "raw_training_loss": raw_loss,
                "smoothed_training_loss": state.smoothed_training_loss,
                "learning_rate": learning_rate,
                "learning_rate_for_next_step": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
                "tokens_per_second": tokens_per_step / elapsed,
                "step_duration_seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }

            if state.step in EXPECTED_MILESTONES:
                validation = evaluate_loss(
                    model,
                    validation_data,
                    sequence_length=sequence_length,
                    micro_batch_size=micro_batch,
                    device=device,
                )
                evaluation = evaluate_loss(
                    model,
                    eval_data,
                    sequence_length=sequence_length,
                    micro_batch_size=micro_batch,
                    device=device,
                )
                state.last_validation_loss = float(validation["loss"])
                checkpoint_path = run_dir / "checkpoints" / checkpoint_name(state.step, state.tokens_seen)
                if state.best_validation_loss is None or float(validation["loss"]) < state.best_validation_loss:
                    state.best_validation_loss = float(validation["loss"])
                    state.best_checkpoint = str(checkpoint_path)
                atomic_write_json(run_dir / "validations" / f"step_{state.step:06d}.json", validation)
                atomic_write_json(run_dir / "eval" / f"step_{state.step:06d}.json", evaluation)
                metric["validation_loss"] = validation["loss"]
                metric["validation_perplexity"] = validation["perplexity"]
                metric["eval_loss"] = evaluation["loss"]
                metric["eval_perplexity"] = evaluation["perplexity"]
                metric["generation"] = generation_snapshot(
                    model,
                    tokenizer,
                    output_path=run_dir / "generations" / f"step_{state.step:06d}.json",
                    max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
                    seed=config["initialization_seed"],
                    checkpoint_stage=_checkpoint_stage(state.step, config),
                )
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state=state,
                    tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
                    data_manifest_hash=data_hash,
                )
                reloaded = verify_checkpoint_reload(
                    checkpoint_path,
                    config=config,
                    model_config=model_config,
                    tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
                    data_manifest_hash=data_hash,
                )
                if (reloaded.step, reloaded.tokens_seen, reloaded.data_position) != (
                    state.step, state.tokens_seen, state.data_position
                ):
                    raise ValueError("saved milestone checkpoint reload mismatch")
                last_checkpoint = checkpoint_path
                run_manifest.update({
                    "latest_checkpoint": str(checkpoint_path),
                    "best_checkpoint": state.best_checkpoint,
                    "best_validation_loss": state.best_validation_loss,
                    "optimizer_steps": state.step,
                    "consumed_tokens": state.tokens_seen,
                    "data_position": state.data_position,
                })
                atomic_write_json(run_dir / "run_manifest.json", run_manifest)

            append_jsonl(run_dir / "metrics.jsonl", metric)
            if state.step % 32 == 0 or state.step in EXPECTED_MILESTONES:
                print(
                    f"step={state.step} tokens={state.tokens_seen} loss={raw_loss:.6f} "
                    f"lr={learning_rate:.8f} tok_s={metric['tokens_per_second']:.2f}",
                    flush=True,
                )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    expected_tokens = target_step * tokens_per_step
    if (state.step, state.tokens_seen, state.data_position) != (target_step, expected_tokens, expected_tokens):
        raise ValueError("segment stopped outside its exact approved boundary")
    if Path(last_checkpoint).name != checkpoint_name(state.step, state.tokens_seen):
        raise ValueError("segment boundary is missing its immutable checkpoint")
    summary = {
        "result": "PASS",
        "segment_start_step": segment_start_step,
        "segment_end_step": state.step,
        "segment_end_tokens": state.tokens_seen,
        "next_optimizer_step": state.step + 1 if state.step < config["maximum_optimizer_steps"] else None,
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": state.best_checkpoint,
        "best_validation_loss": state.best_validation_loss,
        "last_validation_loss": state.last_validation_loss,
        "scheduler_last_epoch": scheduler.last_epoch,
        "learning_rate_for_next_step": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": time.perf_counter() - segment_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    atomic_write_json(run_dir / f"segment_to_step_{target_step:06d}.json", summary)
    run_manifest["status"] = "midpoint_complete" if target_step == config["segment_a_target_step"] else "full_epoch_complete"
    run_manifest["latest_checkpoint"] = str(last_checkpoint)
    run_manifest["best_checkpoint"] = state.best_checkpoint
    run_manifest["best_validation_loss"] = state.best_validation_loss
    run_manifest["optimizer_steps"] = state.step
    run_manifest["consumed_tokens"] = state.tokens_seen
    run_manifest["data_position"] = state.data_position
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    return summary


def validate_checkpoint_process(config_path: Path, checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite resume validation: {output_path}")
    config = load_and_validate_full_epoch_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    profile = run_manifest["selected_profile"]
    model_config = DarkMindV2Config.from_json_file(config["model_config"])
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(model_config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    state = load_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=run_manifest["tokenized_corpus_manifest_sha256"],
    )
    if state.step != config["segment_a_target_step"]:
        raise ValueError("resume validation requires the exact midpoint checkpoint")
    expected_tokens = state.step * config["data"]["effective_tokens_per_optimizer_step"]
    if (state.tokens_seen, state.data_position) != (expected_tokens, expected_tokens):
        raise ValueError("midpoint data traversal state mismatch")
    recorded = json.loads((run_dir / "validations" / f"step_{state.step:06d}.json").read_text(encoding="utf-8"))
    validation = evaluate_loss(
        model,
        TokenShardDataset(Path(config["data"]["tokenized_dir"]), "validation"),
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    if abs(float(validation["loss"]) - float(recorded["loss"])) > 1e-6:
        raise ValueError("midpoint validation loss is not reproducible")
    resume = torch.load(checkpoint_dir / "resume_state.pt", map_location="cpu", weights_only=False)
    expected_next_lr = learning_rate_for_step(state.step + 1, config)
    actual_next_lr = optimizer.param_groups[0]["lr"]
    checks = {
        "model_hash_matches_metadata": sha256_file(checkpoint_dir / "model" / "model.safetensors") == json.loads(
            (checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )["model_files"]["model_sha256"],
        "optimizer_state_present": bool(resume["optimizer"]["state"]),
        "scheduler_state_present": bool(resume["scheduler"]),
        "rng_state_present": bool(resume["rng"]),
        "scheduler_last_epoch_matches": scheduler.last_epoch == state.step,
        "learning_rate_continuity": math.isclose(actual_next_lr, expected_next_lr, abs_tol=1e-15),
        "data_position_continuity": state.data_position == expected_tokens,
        "validation_reproducible": abs(float(validation["loss"]) - float(recorded["loss"])) <= 1e-6,
        "tokenizer_hash_matches": run_manifest["tokenizer_hashes"]["tokenizer.model"] == EXPECTED_HASHES["tokenizer.model"],
        "corpus_hash_matches": tokenized_manifest_hash(Path(config["data"]["tokenized_dir"])) == run_manifest["tokenized_corpus_manifest_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"midpoint resume integrity failure: {checks}")
    report = {
        "schema_version": "darkmind-v2-phase2c-midpoint-resume-v1",
        "result": "PASS",
        "checkpoint": str(checkpoint_dir),
        "model_weight_sha256": sha256_file(checkpoint_dir / "model" / "model.safetensors"),
        "optimizer_step": state.step,
        "consumed_tokens": state.tokens_seen,
        "data_position": state.data_position,
        "sequence_index": state.data_position // config["data"]["sequence_length"],
        "scheduler_last_epoch": scheduler.last_epoch,
        "expected_next_optimizer_step": state.step + 1,
        "expected_next_learning_rate": expected_next_lr,
        "actual_next_learning_rate": actual_next_lr,
        "validation_loss": validation["loss"],
        "recorded_validation_loss": recorded["loss"],
        "checks": checks,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_full_epoch.json"))
    initialize.add_argument(
        "--calibration",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/profiling/tiny_full_epoch_calibration.json"),
    )
    segment = commands.add_parser("segment")
    segment.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_full_epoch.json"))
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--target-step", type=int, required=True)
    validate = commands.add_parser("validate-midpoint")
    validate.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_full_epoch.json"))
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        report = initialize_run(args.config, args.calibration)
    elif args.command == "segment":
        report = train_segment(args.config, args.checkpoint, target_step=args.target_step)
    else:
        report = validate_checkpoint_process(args.config, args.checkpoint, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
