"""Fail-closed runtime guards for Phase 3B finalist training profiles."""

from __future__ import annotations

import sys
from typing import Any

import torch

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config


UNSAFE_WINDOWS_CHECKPOINTING_SIGNATURE = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "mlp_hidden_size": 3072,
    "block_size": 512,
}


def is_candidate_c(config: DarkMindV2Config) -> bool:
    return all(getattr(config, key) == value for key, value in UNSAFE_WINDOWS_CHECKPOINTING_SIGNATURE.items())


def validate_training_environment(
    config: DarkMindV2Config,
    *,
    micro_batch_size: int,
    precision: str,
    platform_name: str | None = None,
    torch_version: str | None = None,
    allow_unsafe_diagnostic: bool = False,
) -> dict[str, Any]:
    """Reject the checkpointing profile and warn on the broader C signature."""
    platform_value = sys.platform if platform_name is None else platform_name
    torch_value = torch.__version__ if torch_version is None else torch_version
    affected = (
        platform_value == "win32"
        and torch_value.startswith("2.4.1+cu121")
        and is_candidate_c(config)
        and config.attention_implementation == "sdpa"
        and micro_batch_size == 2
        and precision == "bf16"
    )
    hard_rejected = affected and config.gradient_checkpointing
    warnings: list[str] = []
    if affected and not config.gradient_checkpointing:
        warnings.append(
            "Candidate C produced two intermittent process-level terminations in the "
            "recommended checkpointing-off profile; use Candidate D for production training"
        )
    decision = {
        "result": "WARN" if (warnings or (hard_rejected and allow_unsafe_diagnostic)) else "PASS",
        "affected": affected,
        "hard_rejected": hard_rejected,
        "platform": platform_value,
        "torch_version": torch_value,
        "warnings": warnings,
        "policy": "gradient checkpointing disabled for production base v1 profiles",
    }
    if hard_rejected and not allow_unsafe_diagnostic:
        raise RuntimeError(
            "unsafe Candidate C profile rejected: Windows + PyTorch 2.4.1+cu121 + BF16 + "
            "micro-batch 2 + SDPA + gradient checkpointing has a recorded native fail-fast; "
            "use Candidate D for production training or an explicitly diagnosed environment"
        )
    return decision
