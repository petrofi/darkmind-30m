"""Safe model-weight persistence and deterministic config hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_config_hash(config: DarkMindV2Config) -> str:
    return canonical_json_hash(config.architecture_dict())


def save_model_package(
    model: DarkMindV2ForCausalLM,
    output_dir: Path,
    *,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_model
    except ImportError as exc:
        raise RuntimeError("safetensors is required for DarkMind v2 model weights") from exc
    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    config_path.write_text(
        json.dumps(model.config.architecture_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_model(model, str(weights_path))
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "architecture_hash": model_config_hash(model.config),
    }


def load_model_package(path: Path, *, device: str = "cpu") -> DarkMindV2ForCausalLM:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_model
    except ImportError as exc:
        raise RuntimeError("safetensors is required for DarkMind v2 model weights") from exc
    config = DarkMindV2Config.from_json_file(path / "config.json")
    weights_path = path / "model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        stored_dtype = None
        for key in weights.keys():
            candidate_dtype = weights.get_tensor(key).dtype
            if candidate_dtype.is_floating_point:
                stored_dtype = candidate_dtype
                break
        if stored_dtype is None:
            raise ValueError("model package contains no floating-point weights")
    model = DarkMindV2ForCausalLM(config).to(device=device, dtype=stored_dtype)
    load_model(model, str(weights_path), device=device)
    model.tie_weights()
    model.eval()
    return model
