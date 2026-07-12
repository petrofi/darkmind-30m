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
    block_size: int
    tied_embeddings: bool
    bias: bool
    token_embedding_params: int
    position_embedding_params: int
    attention_params: int
    mlp_params: int
    layer_norm_params: int
    lm_head_params: int
    total_params: int
    vocab_related_params: int
    vocab_related_percentage: float
    non_vocab_params: int


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
) -> ModelSizeEstimate:
    values = (vocab_size, n_layers, n_heads, n_embd, block_size)
    if any(value <= 0 for value in values):
        raise ValueError("all model dimensions must be positive")
    if n_embd % n_heads:
        raise ValueError("n_embd must be divisible by n_heads")

    token_embedding = vocab_size * n_embd
    position_embedding = block_size * n_embd
    attention_per_layer = 4 * n_embd * n_embd + (4 * n_embd if bias else 0)
    mlp_per_layer = 8 * n_embd * n_embd + (5 * n_embd if bias else 0)
    attention = n_layers * attention_per_layer
    mlp = n_layers * mlp_per_layer
    layer_norm = (2 * n_layers + 1) * n_embd * (2 if bias else 1)
    lm_head = 0 if tied_embeddings else vocab_size * n_embd + (vocab_size if bias else 0)
    total = token_embedding + position_embedding + attention + mlp + layer_norm + lm_head
    vocab_related = token_embedding + lm_head
    return ModelSizeEstimate(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_embd,
        block_size=block_size,
        tied_embeddings=tied_embeddings,
        bias=bias,
        token_embedding_params=token_embedding,
        position_embedding_params=position_embedding,
        attention_params=attention,
        mlp_params=mlp,
        layer_norm_params=layer_norm,
        lm_head_params=lm_head,
        total_params=total,
        vocab_related_params=vocab_related,
        vocab_related_percentage=round(vocab_related / total * 100, 4),
        non_vocab_params=total - vocab_related,
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
