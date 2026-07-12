"""Bounded two-process Stage-1 pretraining for the DarkMind v2 tiny base."""

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

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
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
from darkmind_v2.training.validate_stage1_config import load_and_validate_stage1_config


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_optimizer(model: DarkMindV2ForCausalLM, config: dict[str, Any]) -> torch.optim.AdamW:
    optimizer = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer["peak_learning_rate"],
        betas=(optimizer["beta1"], optimizer["beta2"]),
        weight_decay=optimizer["weight_decay"],
    )


def learning_rate_factor(step: int, config: dict[str, Any]) -> float:
    warmup = config["schedule"]["warmup_optimizer_steps"]
    total_steps = config["maximum_total_training_tokens"] // config["data"]["effective_tokens_per_optimizer_step"]
    minimum_factor = (
        config["schedule"]["minimum_learning_rate"] / config["optimizer"]["peak_learning_rate"]
    )
    if step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_factor + (1.0 - minimum_factor) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_factor(step, config),
    )


@torch.no_grad()
def evaluate_loss(
    model: DarkMindV2ForCausalLM,
    dataset: TokenShardDataset,
    *,
    sequences: int,
    sequence_length: int,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if sequences <= 0:
        raise ValueError("evaluation sequence count must be positive")
    model.eval()
    losses = []
    offset = 0
    remaining = sequences
    while remaining:
        batch_size = min(micro_batch_size, remaining)
        batch = dataset.batch(
            offset=offset,
            micro_batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, labels=batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("non-finite evaluation loss")
        losses.append((float(output.loss), batch_size))
        offset += batch_size * sequence_length
        remaining -= batch_size
    model.train()
    loss = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    return {"loss": loss, "perplexity": math.exp(min(loss, 80.0)), "sequences": sequences}


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
    for item in snapshot["results"]:
        item["policy"] = classify_generation_health(
            item["generation"],
            item["token_ids"],
            checkpoint_stage=checkpoint_stage,
            maximum_repetition_ratio=item["health"]["maximum_repetition_ratio"],
            tokenizer_hash_match=True,
            model_config_hash_match=True,
            logits_finite=True,
            loss_finite=True,
        )
    enforce_generation_policy(snapshot["results"])
    atomic_write_json(output_path, snapshot)
    warning_counts: dict[str, int] = {}
    for item in snapshot["results"]:
        for warning in item["policy"]["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    return {
        "path": str(output_path),
        "content_hash": snapshot["deterministic_content_hash"],
        "prompts": len(snapshot["results"]),
        "health_failures": sum(bool(item["health"]["failures"]) for item in snapshot["results"]),
        "hard_failures": 0,
        "warning_counts": dict(sorted(warning_counts.items())),
        "checkpoint_stage": checkpoint_stage,
    }


def git_state() -> dict[str, Any]:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "status_short": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
    }


def checkpoint_name(state: TrainingState) -> str:
    return f"step_{state.step:06d}_tokens_{state.tokens_seen:09d}"


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
    if sha256_file(checkpoint_dir / "model" / "model.safetensors") != json.loads(
        (checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )["model_files"]["model_sha256"]:
        raise ValueError(f"checkpoint model hash mismatch: {checkpoint_dir}")
    return state


def initialize_run(config_path: Path, calibration_path: Path) -> dict[str, Any]:
    config = load_and_validate_stage1_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage-1 run: {run_dir}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("result") != "PASS":
        raise ValueError("calibration did not pass")
    profile = calibration["selected_profile"]
    if profile["micro_batch_size"] * profile["gradient_accumulation_steps"] * 256 != 4096:
        raise ValueError("calibration profile changed the effective token batch")
    verify_frozen_tokenizer()
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    data_summary = dataset_summary(tokenized_dir)
    model_config = DarkMindV2Config.from_json_file(config["model_config"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage-1 initialization requires CUDA BF16")
    set_deterministic_seed(config["seed"])
    device = torch.device("cuda")
    model = DarkMindV2ForCausalLM(model_config).to(device)
    if not model.embeddings_are_tied():
        raise ValueError("model embeddings are not tied")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    state = TrainingState()
    validation_data = TokenShardDataset(tokenized_dir, "validation")
    eval_data = TokenShardDataset(tokenized_dir, "eval")
    initial_validation = evaluate_loss(
        model,
        validation_data,
        sequences=config["evaluation"]["validation_sequences"],
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    initial_eval = evaluate_loss(
        model,
        eval_data,
        sequences=config["evaluation"]["validation_sequences"],
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    state.last_validation_loss = initial_validation["loss"]
    tokenizer = FrozenTokenizer()
    run_dir.mkdir(parents=True)
    for directory in ("checkpoints", "best_candidates", "validations", "generations", "evaluations"):
        (run_dir / directory).mkdir()
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    generation = generation_snapshot(
        model,
        tokenizer,
        output_path=run_dir / "generations" / "step_000000.json",
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
        seed=config["seed"],
        checkpoint_stage="initialization",
    )
    initial_checkpoint = run_dir / "checkpoints" / "initial_step_000000"
    checkpoint_metadata = save_checkpoint(
        initial_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_summary["manifest_hash"],
    )
    reloaded = verify_checkpoint_reload(
        initial_checkpoint,
        config=config,
        model_config=model_config,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_summary["manifest_hash"],
    )
    if reloaded.step != 0 or reloaded.tokens_seen != 0:
        raise ValueError("initial checkpoint reload state mismatch")
    atomic_write_json(run_dir / "resolved_model_config.json", model_config.architecture_dict())
    atomic_write_json(run_dir / "resolved_training_config.json", {**config, "selected_profile": profile})
    atomic_write_json(run_dir / "environment.json", probe_environment(run_dir))
    atomic_write_json(run_dir / "git_state.json", git_state())
    atomic_write_json(run_dir / "validations" / "step_000000.json", initial_validation)
    atomic_write_json(run_dir / "evaluations" / "step_000000.json", initial_eval)
    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "optimizer_step": 0,
            "consumed_tokens": 0,
            "validation_loss": initial_validation["loss"],
            "validation_perplexity": initial_validation["perplexity"],
            "event": "initialization",
        },
    )
    corpus_manifest_hash = sha256_file(Path("darkmind_v2/data/phase1b/processed/corpus_manifest.json"))
    run_manifest = {
        "schema_version": "darkmind-v2-stage1-run-v1",
        "run_name": config["run"]["name"],
        "status": "initialized",
        "description": "First real DarkMind v2 pretrained checkpoint run",
        "run_type": "controlled Stage-1 research run",
        "instruction_tuned": False,
        "conversational_assistant": False,
        "huggingface_upload_authorized": False,
        "target_training_tokens": config["maximum_total_training_tokens"],
        "selected_profile": profile,
        "model_parameters": model.parameter_count(),
        "model_config_hash": model_config_hash(model_config),
        "model_initialization_sha256": checkpoint_metadata["model_files"]["model_sha256"],
        "tokenizer_model_sha256": EXPECTED_HASHES["tokenizer.model"],
        "tokenizer_freeze_manifest_sha256": EXPECTED_HASHES["tokenizer_freeze_manifest.json"],
        "tokenized_corpus_manifest_sha256": data_summary["manifest_hash"],
        "tokenized_corpus_content_hash": data_summary["manifest_content_hash"],
        "source_corpus_manifest_sha256": corpus_manifest_hash,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_validation": initial_validation,
        "initial_eval": initial_eval,
        "initial_generation": generation,
        "aborted_predecessor": config["run"].get("aborted_predecessor"),
        "aborted_predecessor_training_tokens": config["run"].get("aborted_predecessor_training_tokens", 0),
        "aborted_predecessor_optimizer_steps": config["run"].get("aborted_predecessor_optimizer_steps", 0),
        "lineage_note": config["run"].get("lineage_note"),
        "weights_carried_from_predecessor": False,
    }
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest


def train_segment(
    config_path: Path,
    checkpoint_dir: Path,
    *,
    target_tokens: int,
) -> dict[str, Any]:
    config = load_and_validate_stage1_config(config_path)
    if target_tokens not in {config["segment_a_target_tokens"], config["segment_b_target_tokens"]}:
        raise ValueError("segment target is outside the approved Stage-1 boundaries")
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    profile = run_manifest["selected_profile"]
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
    if state.tokens_seen >= target_tokens:
        raise ValueError("checkpoint has already reached or exceeded the segment target")
    expected_target_step = target_tokens // tokens_per_step
    train_data = TokenShardDataset(tokenized_dir, "train")
    validation_data = TokenShardDataset(tokenized_dir, "validation")
    tokenizer = FrozenTokenizer()
    micro_batch = profile["micro_batch_size"]
    accumulation = profile["gradient_accumulation_steps"]
    sequence_length = config["data"]["sequence_length"]
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
        while state.step < expected_target_step:
            if interrupted:
                state.interrupted = True
                raise KeyboardInterrupt("safe interruption requested before next optimizer step")
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            learning_rate = optimizer.param_groups[0]["lr"]
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
                        raise FloatingPointError("non-finite Stage-1 training loss")
                    loss = output.loss / accumulation
                loss.backward()
                micro_losses.append(float(output.loss.detach()))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"]))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite Stage-1 gradient norm")
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            state.step += 1
            state.tokens_seen += tokens_per_step
            state.data_position += tokens_per_step
            raw_loss = sum(micro_losses) / len(micro_losses)
            state.last_training_loss = raw_loss
            state.smoothed_training_loss = (
                raw_loss
                if state.smoothed_training_loss is None
                else 0.9 * state.smoothed_training_loss + 0.1 * raw_loss
            )
            step_duration = time.perf_counter() - step_started
            metric: dict[str, Any] = {
                "event": "optimizer_step",
                "optimizer_step": state.step,
                "consumed_tokens": state.tokens_seen,
                "data_position": state.data_position,
                "raw_training_loss": raw_loss,
                "smoothed_training_loss": state.smoothed_training_loss,
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm,
                "tokens_per_second": tokens_per_step / step_duration,
                "step_duration_seconds": step_duration,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }

            validation = None
            if state.step % config["evaluation"]["validation_interval_steps"] == 0:
                validation = evaluate_loss(
                    model,
                    validation_data,
                    sequences=config["evaluation"]["validation_sequences"],
                    sequence_length=sequence_length,
                    micro_batch_size=micro_batch,
                    device=device,
                )
                state.last_validation_loss = validation["loss"]
                metric["validation_loss"] = validation["loss"]
                metric["validation_perplexity"] = validation["perplexity"]
                atomic_write_json(run_dir / "validations" / f"step_{state.step:06d}.json", validation)
                if state.best_validation_loss is None or validation["loss"] < state.best_validation_loss:
                    state.best_validation_loss = validation["loss"]
                    best_dir = run_dir / "best_candidates" / checkpoint_name(state)
                    state.best_checkpoint = str(best_dir)
                    save_checkpoint(
                        best_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_state=state,
                        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
                        data_manifest_hash=data_hash,
                    )

            if state.step % config["evaluation"]["fixed_prompt_generation_interval_steps"] == 0:
                metric["generation"] = generation_snapshot(
                    model,
                    tokenizer,
                    output_path=run_dir / "generations" / f"step_{state.step:06d}.json",
                    max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
                    seed=config["seed"],
                    checkpoint_stage=(
                        "midpoint"
                        if state.tokens_seen <= config["segment_a_target_tokens"]
                        else "stage1_final"
                    ),
                )

            append_jsonl(run_dir / "metrics.jsonl", metric)
            if state.step % config["checkpointing"]["checkpoint_interval_steps"] == 0 or state.step == expected_target_step:
                checkpoint_path = run_dir / "checkpoints" / checkpoint_name(state)
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
                if reloaded.step != state.step or reloaded.tokens_seen != state.tokens_seen:
                    raise ValueError("saved checkpoint reload state mismatch")
                last_checkpoint = checkpoint_path
            if state.step % 16 == 0:
                print(
                    f"step={state.step} tokens={state.tokens_seen} loss={raw_loss:.6f} "
                    f"lr={learning_rate:.8f} tok_s={metric['tokens_per_second']:.2f}",
                    flush=True,
                )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if state.tokens_seen != target_tokens or state.step != expected_target_step:
        raise ValueError("segment stopped outside its exact approved boundary")
    segment_summary = {
        "result": "PASS",
        "segment_start_step": segment_start_step,
        "segment_end_step": state.step,
        "segment_end_tokens": state.tokens_seen,
        "first_optimizer_step": segment_start_step + 1,
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": state.best_checkpoint,
        "best_validation_loss": state.best_validation_loss,
        "last_validation_loss": state.last_validation_loss,
        "elapsed_seconds": time.perf_counter() - segment_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    atomic_write_json(run_dir / f"segment_to_{target_tokens:09d}.json", segment_summary)
    run_manifest["status"] = "midpoint_complete" if target_tokens == config["segment_a_target_tokens"] else "stage1_complete"
    run_manifest["latest_checkpoint"] = str(last_checkpoint)
    run_manifest["best_checkpoint"] = state.best_checkpoint
    run_manifest["best_validation_loss"] = state.best_validation_loss
    run_manifest["consumed_tokens"] = state.tokens_seen
    run_manifest["optimizer_steps"] = state.step
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    return segment_summary


def validate_checkpoint_process(
    config_path: Path,
    checkpoint_dir: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint validation: {output_path}")
    config = load_and_validate_stage1_config(config_path)
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
    validation_data = TokenShardDataset(Path(config["data"]["tokenized_dir"]), "validation")
    validation = evaluate_loss(
        model,
        validation_data,
        sequences=config["evaluation"]["validation_sequences"],
        sequence_length=config["data"]["sequence_length"],
        micro_batch_size=profile["micro_batch_size"],
        device=device,
    )
    recorded = json.loads((run_dir / "validations" / f"step_{state.step:06d}.json").read_text(encoding="utf-8"))
    tolerance = 1e-6
    if abs(validation["loss"] - recorded["loss"]) > tolerance:
        raise ValueError(
            f"validation loss is not reproducible: actual={validation['loss']} recorded={recorded['loss']}"
        )
    generation = generation_snapshot(
        model,
        FrozenTokenizer(),
        output_path=run_dir / "generations" / f"reload_step_{state.step:06d}.json",
        max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
        seed=config["seed"],
        checkpoint_stage=(
            "initialization"
            if state.tokens_seen == 0
            else (
                "midpoint"
                if state.tokens_seen == config["segment_a_target_tokens"]
                else "stage1_final"
            )
        ),
    )
    resume_state = torch.load(checkpoint_dir / "resume_state.pt", map_location="cpu", weights_only=False)
    report = {
        "result": "PASS",
        "checkpoint": str(checkpoint_dir),
        "model_weight_sha256": sha256_file(checkpoint_dir / "model" / "model.safetensors"),
        "optimizer_state_present": bool(resume_state["optimizer"]["state"]),
        "optimizer_state_expected": state.step > 0,
        "scheduler_last_epoch": resume_state["scheduler"]["last_epoch"],
        "rng_state_present": bool(resume_state["rng"]),
        "optimizer_step": state.step,
        "consumed_tokens": state.tokens_seen,
        "data_position": state.data_position,
        "tokenizer_hash": run_manifest["tokenizer_model_sha256"],
        "model_config_hash": run_manifest["model_config_hash"],
        "tokenized_manifest_hash": run_manifest["tokenized_corpus_manifest_sha256"],
        "validation_loss": validation["loss"],
        "recorded_validation_loss": recorded["loss"],
        "validation_tolerance": tolerance,
        "generation": generation,
        "next_optimizer_step": state.step + 1,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1.json"))
    initialize.add_argument(
        "--calibration",
        type=Path,
        default=Path("darkmind_v2/data/phase2a/profiling/tiny_stage1_calibration.json"),
    )

    segment = subparsers.add_parser("segment")
    segment.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1.json"))
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--target-tokens", type=int, required=True)

    validate = subparsers.add_parser("validate-checkpoint")
    validate.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1.json"))
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "initialize":
        report = initialize_run(args.config, args.calibration)
    elif args.command == "segment":
        report = train_segment(args.config, args.checkpoint, target_tokens=args.target_tokens)
    else:
        report = validate_checkpoint_process(args.config, args.checkpoint, output_path=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
