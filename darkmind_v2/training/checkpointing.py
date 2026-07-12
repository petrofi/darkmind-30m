"""Checkpoint model weights safely and retain internal resume state."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.modeling.model_io import model_config_hash, save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.training.training_state import TrainingState


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_state: TrainingState,
    tokenizer_hash: str,
    data_manifest_hash: str,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_hashes = save_model_package(model, checkpoint_dir / "model", allow_overwrite=allow_overwrite)
    resume_payload = {
        "format": "darkmind-v2-internal-resume-state-v1",
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "training_state": training_state.to_dict(),
    }
    temporary = checkpoint_dir / ".resume_state.pt.tmp"
    torch.save(resume_payload, temporary)
    os.replace(temporary, checkpoint_dir / "resume_state.pt")
    metadata = {
        "schema_version": "darkmind-v2-checkpoint-metadata-v1",
        "model_config_hash": model_config_hash(model.config),
        "tokenizer_model_sha256": tokenizer_hash,
        "tokenized_data_manifest_hash": data_manifest_hash,
        "training_state": training_state.to_dict(),
        "model_files": model_hashes,
        "resume_state_format": "trusted-local-pytorch-state-not-for-public-release",
    }
    atomic_write_json(checkpoint_dir / "checkpoint_metadata.json", metadata)
    return metadata


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_tokenizer_hash: str,
    expected_data_manifest_hash: str,
    restore_rng: bool = True,
) -> TrainingState:
    metadata = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "model_config_hash": model_config_hash(model.config),
        "tokenizer_model_sha256": expected_tokenizer_hash,
        "tokenized_data_manifest_hash": expected_data_manifest_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint resume compatibility failure: {key}")
    try:
        from safetensors.torch import load_model
    except ImportError as exc:
        raise RuntimeError("safetensors is required to resume DarkMind v2") from exc
    load_model(model, str(checkpoint_dir / "model" / "model.safetensors"), device=str(next(model.parameters()).device))
    model.tie_weights()
    resume = torch.load(checkpoint_dir / "resume_state.pt", map_location="cpu", weights_only=False)
    if resume.get("format") != "darkmind-v2-internal-resume-state-v1":
        raise ValueError("unsupported internal resume-state format")
    optimizer.load_state_dict(resume["optimizer"])
    scheduler.load_state_dict(resume["scheduler"])
    if restore_rng:
        restore_rng_state(resume["rng"])
    return TrainingState.from_dict(resume["training_state"])
