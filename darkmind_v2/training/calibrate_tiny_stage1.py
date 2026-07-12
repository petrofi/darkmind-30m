"""One-step real-data CUDA calibration for approved Stage-1 profiles."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.training.token_shard_dataset import TokenShardDataset, tokenized_manifest_hash
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed
from darkmind_v2.training.validate_stage1_config import load_and_validate_stage1_config


def calibrate(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {output_path}")
    config = load_and_validate_stage1_config(config_path)
    model_config = DarkMindV2Config.from_json_file(config["model_config"])
    tokenized_dir = Path(config["data"]["tokenized_dir"])
    train_data = TokenShardDataset(tokenized_dir, "train")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage-1 calibration requires CUDA with BF16")
    device = torch.device("cuda")
    attempts: list[dict[str, Any]] = []
    selected = None

    for profile in config["profiles"]:
        set_deterministic_seed(config["seed"])
        model = DarkMindV2ForCausalLM(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["peak_learning_rate"],
            betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
            weight_decay=config["optimizer"]["weight_decay"],
        )
        micro_batch = profile["micro_batch_size"]
        accumulation = profile["gradient_accumulation_steps"]
        sequence_length = config["data"]["sequence_length"]
        attempt: dict[str, Any] = {
            "profile": profile["name"],
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "sequence_length": sequence_length,
            "effective_tokens": micro_batch * accumulation * sequence_length,
            "precision": "bf16",
            "oom": False,
        }
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            losses = []
            for micro_step in range(accumulation):
                offset = micro_step * micro_batch * sequence_length
                batch = train_data.batch(
                    offset=offset,
                    micro_batch_size=micro_batch,
                    sequence_length=sequence_length,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch, labels=batch)
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise FloatingPointError("non-finite calibration loss")
                    loss = output.loss / accumulation
                loss.backward()
                losses.append(float(output.loss.detach()))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"]))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite calibration gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            attempt.update(
                {
                    "forward_backward_success": True,
                    "optimizer_step_success": True,
                    "loss": sum(losses) / len(losses),
                    "gradient_norm": gradient_norm,
                    "step_duration_seconds": elapsed,
                    "tokens_per_second": attempt["effective_tokens"] / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "bf16_stable": True,
                }
            )
            attempts.append(attempt)
            selected = profile
            break
        except torch.OutOfMemoryError as exc:
            attempt.update(
                {
                    "forward_backward_success": False,
                    "optimizer_step_success": False,
                    "oom": True,
                    "error": str(exc),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "bf16_stable": False,
                }
            )
            attempts.append(attempt)
            optimizer.zero_grad(set_to_none=True)
            del optimizer, model
            torch.cuda.empty_cache()
            continue
        finally:
            if "optimizer" in locals():
                del optimizer
            if "model" in locals():
                del model
            torch.cuda.empty_cache()

    if selected is None:
        raise RuntimeError("all approved Stage-1 calibration profiles failed")
    report = {
        "result": "PASS",
        "tokenized_manifest_sha256": tokenized_manifest_hash(tokenized_dir),
        "attempts": attempts,
        "selected_profile": selected,
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("darkmind_v2/data/phase2a/profiling/tiny_stage1_calibration.json"),
    )
    args = parser.parse_args()
    print(json.dumps(calibrate(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
