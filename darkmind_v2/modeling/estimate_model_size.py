"""Estimate GPT-style decoder-only model parameter counts before training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSizeEstimate:
    vocab_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    head_dimension: int
    mlp_hidden_size: int
    block_size: int
    tied_embeddings: bool
    bias: bool
    token_embedding_params: int
    position_embedding_params: int
    attention_params: int
    mlp_params: int
    layer_norm_params: int
    attention_weight_params: int
    attention_bias_params: int
    mlp_weight_params: int
    mlp_bias_params: int
    normalization_weight_params: int
    normalization_bias_params: int
    normalization_and_bias_params: int
    lm_head_params: int
    total_params: int
    vocab_related_params: int
    vocab_related_percentage: float
    non_vocab_params: int
    transformer_body_params: int
    bf16_checkpoint_bytes: int
    adamw_state_bytes: int


PRESETS: dict[str, dict[str, int | bool]] = {
    "tiny_smoke": {
        "vocab_size": 24000,
        "n_layers": 4,
        "n_heads": 4,
        "n_embd": 256,
        "block_size": 256,
        "tied_embeddings": True,
        "bias": True,
    },
    "candidate_base_45m_class": {
        "vocab_size": 24000,
        "n_layers": 10,
        "n_heads": 8,
        "n_embd": 512,
        "block_size": 512,
        "tied_embeddings": True,
        "bias": True,
    },
    "candidate_base_60m_class": {
        "vocab_size": 24000,
        "n_layers": 12,
        "n_heads": 8,
        "n_embd": 512,
        "block_size": 512,
        "tied_embeddings": True,
        "bias": True,
    },
}


def estimate_model_size(
    *,
    vocab_size: int,
    n_layers: int,
    n_heads: int,
    n_embd: int,
    block_size: int,
    tied_embeddings: bool,
    bias: bool,
    mlp_hidden_size: int | None = None,
) -> ModelSizeEstimate:
    values = (vocab_size, n_layers, n_heads, n_embd, block_size)
    if any(value <= 0 for value in values):
        raise ValueError("all model dimensions must be positive")
    if n_embd % n_heads:
        raise ValueError("n_embd must be divisible by n_heads")
    if mlp_hidden_size is not None and mlp_hidden_size <= 0:
        raise ValueError("mlp_hidden_size must be positive when provided")
    hidden_size = mlp_hidden_size or 4 * n_embd

    token_embedding = vocab_size * n_embd
    position_embedding = block_size * n_embd
    attention_weights = n_layers * 4 * n_embd * n_embd
    attention_biases = n_layers * 4 * n_embd if bias else 0
    mlp_weights = n_layers * 2 * n_embd * hidden_size
    mlp_biases = n_layers * (hidden_size + n_embd) if bias else 0
    normalization_weights = (2 * n_layers + 1) * n_embd
    normalization_biases = normalization_weights if bias else 0
    attention = attention_weights + attention_biases
    mlp = mlp_weights + mlp_biases
    layer_norm = normalization_weights + normalization_biases
    lm_head = 0 if tied_embeddings else vocab_size * n_embd + (vocab_size if bias else 0)
    total = token_embedding + position_embedding + attention + mlp + layer_norm + lm_head
    vocab_related = token_embedding + lm_head
    return ModelSizeEstimate(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_embd,
        head_dimension=n_embd // n_heads,
        mlp_hidden_size=hidden_size,
        block_size=block_size,
        tied_embeddings=tied_embeddings,
        bias=bias,
        token_embedding_params=token_embedding,
        position_embedding_params=position_embedding,
        attention_params=attention,
        mlp_params=mlp,
        layer_norm_params=layer_norm,
        attention_weight_params=attention_weights,
        attention_bias_params=attention_biases,
        mlp_weight_params=mlp_weights,
        mlp_bias_params=mlp_biases,
        normalization_weight_params=normalization_weights,
        normalization_bias_params=normalization_biases,
        normalization_and_bias_params=(
            attention_biases + mlp_biases + normalization_weights + normalization_biases
        ),
        lm_head_params=lm_head,
        total_params=total,
        vocab_related_params=vocab_related,
        vocab_related_percentage=round(vocab_related / total * 100, 4),
        non_vocab_params=total - vocab_related,
        transformer_body_params=total - vocab_related,
        bf16_checkpoint_bytes=total * 2,
        adamw_state_bytes=total * 8,
    )


def estimate_presets() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name, config in PRESETS.items():
        tied = estimate_model_size(**config)
        untied_config = dict(config)
        untied_config["tied_embeddings"] = False
        untied = estimate_model_size(**untied_config)
        reports[name] = {
            "recommended_tied": asdict(tied),
            "untied_warning": asdict(untied),
            "warning": "Untied LM heads are not allowed by default for the frozen 24k tokenizer.",
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GPT decoder-only model size with tied/untied vocabulary cost.")
    parser.add_argument("--preset", choices=["all", *PRESETS], default="all")
    args = parser.parse_args()
    reports = estimate_presets()
    payload = {
        "result": "PASS",
        "embedding_policy": "tied_input_output_embeddings",
        "presets": reports if args.preset == "all" else {args.preset: reports[args.preset]},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
