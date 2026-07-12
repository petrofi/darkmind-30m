"""Deterministic fixture-only training validation for DarkMind v2."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import sha256_file
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer
from darkmind_v2.training.checkpointing import load_checkpoint, save_checkpoint
from darkmind_v2.training.training_state import TrainingState


@dataclass
class SmokeResult:
    device: str
    precision: str
    sequence_length: int
    micro_batch_size: int
    steps: int
    initial_loss: float
    final_loss: float
    loss_decreased: bool
    step_time_seconds: float
    tokens_per_second: float
    peak_cuda_bytes: int
    final_gradient_norm: float
    checkpoint_reloaded: bool
    resumed_step: int
    resume_continued_to_step: int
    generation_token_range_valid: bool


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def precision_for_device(device: torch.device, requested: str) -> tuple[str, torch.dtype | None]:
    if requested == "fp32" or device.type == "cpu":
        return "fp32", None
    if requested == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is unavailable")
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        if device.type == "cuda":
            return "fp16", torch.float16
    return "fp32", None


def warmup_cosine_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def fixture_tokens(tokenizer: FrozenTokenizer) -> list[int]:
    texts = (
        "Tiny model pipeline validation.",
        "Kucuk model veri hattini dogrular.",
        "The next token prediction loss should decrease.",
    )
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode_document(text))
    return tokens


def make_fixed_batch(
    tokens: list[int],
    *,
    sequence_length: int,
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if len(tokens) < 2:
        raise ValueError("fixture needs at least two tokens")
    required = sequence_length * micro_batch_size
    repeated = (tokens * (required // len(tokens) + 1))[:required]
    return torch.tensor(repeated, dtype=torch.long, device=device).view(micro_batch_size, sequence_length)


def run_fixture_smoke(
    *,
    model_config: DarkMindV2Config,
    checkpoint_dir: Path,
    steps: int = 12,
    sequence_length: int = 32,
    micro_batch_size: int = 1,
    learning_rate: float = 0.001,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "auto",
    device_name: str = "auto",
    data_manifest_hash: str,
) -> SmokeResult:
    if sequence_length > model_config.block_size:
        raise ValueError("fixture sequence length exceeds model block size")
    set_deterministic_seed(model_config.seed)
    device = resolve_device(device_name)
    precision, autocast_dtype = precision_for_device(device, mixed_precision)
    tokenizer = FrozenTokenizer()
    model = DarkMindV2ForCausalLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_lambda(step, warmup_steps=min(2, steps), total_steps=steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    batch = make_fixed_batch(
        fixture_tokens(tokenizer),
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        device=device,
    )
    state = TrainingState()
    losses: list[float] = []
    gradient_norm = 0.0
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    completed_steps = 0
    try:
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(gradient_accumulation_steps):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    output = model(batch, labels=batch)
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise FloatingPointError("non-finite fixture loss")
                    loss = output.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                accumulated_loss += float(output.loss.detach())
            scaler.unscale_(optimizer)
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_norm = float(gradient_norm_tensor)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            completed_steps = step + 1
            state.step = completed_steps
            state.tokens_seen += batch.numel() * gradient_accumulation_steps
            state.last_training_loss = accumulated_loss / gradient_accumulation_steps
            losses.append(state.last_training_loss)
            if interrupted:
                state.interrupted = True
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    metadata = save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        data_manifest_hash=data_manifest_hash,
    )
    reloaded_model = DarkMindV2ForCausalLM(model_config).to(device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded_model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    reloaded_scheduler = torch.optim.lr_scheduler.LambdaLR(
        reloaded_optimizer,
        lambda step: warmup_cosine_lambda(step, warmup_steps=min(2, steps), total_steps=steps),
    )
    resumed_state = load_checkpoint(
        checkpoint_dir,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
        expected_tokenizer_hash=EXPECTED_HASHES["tokenizer.model"],
        expected_data_manifest_hash=data_manifest_hash,
    )
    checkpoint_reloaded = metadata["training_state"]["step"] == resumed_state.step
    reloaded_model.train()
    reloaded_optimizer.zero_grad(set_to_none=True)
    resumed_output = reloaded_model(batch, labels=batch)
    if resumed_output.loss is None or not torch.isfinite(resumed_output.loss):
        raise FloatingPointError("non-finite resumed fixture loss")
    resumed_output.loss.backward()
    torch.nn.utils.clip_grad_norm_(reloaded_model.parameters(), 1.0)
    reloaded_optimizer.step()
    reloaded_scheduler.step()
    resumed_state.step += 1
    resumed_state.tokens_seen += batch.numel()
    generated = reloaded_model.generate_tokens(batch[:1, :4], max_new_tokens=4)
    generation_valid = bool(torch.all((generated >= 0) & (generated < model_config.vocab_size)))
    return SmokeResult(
        device=str(device),
        precision=precision,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        steps=completed_steps,
        initial_loss=losses[0],
        final_loss=losses[-1],
        loss_decreased=losses[-1] < losses[0],
        step_time_seconds=elapsed / max(1, completed_steps),
        tokens_per_second=state.tokens_seen / max(elapsed, 1e-9),
        peak_cuda_bytes=int(peak_cuda_bytes),
        final_gradient_norm=gradient_norm,
        checkpoint_reloaded=checkpoint_reloaded,
        resumed_step=metadata["training_state"]["step"],
        resume_continued_to_step=resumed_state.step,
        generation_token_range_valid=generation_valid,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=Path("darkmind_v2/config/model_tiny_smoke.json"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("darkmind_v2/data/phase2a/checkpoints/fixture_smoke"),
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("darkmind_v2/data/phase2a/tokenized/tokenized_corpus_manifest.json"),
    )
    args = parser.parse_args()
    if not args.data_manifest.is_file():
        raise FileNotFoundError(f"fixture tokenized-data manifest not found: {args.data_manifest}")
    config = DarkMindV2Config.from_json_file(args.model_config)
    result = run_fixture_smoke(
        model_config=config,
        checkpoint_dir=args.checkpoint_dir,
        steps=args.steps,
        sequence_length=args.sequence_length,
        device_name=args.device,
        mixed_precision=args.precision,
        data_manifest_hash=sha256_file(args.data_manifest),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
