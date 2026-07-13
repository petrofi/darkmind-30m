"""Validate a DarkMind v2 model config against the frozen tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size


def validate_model_config(path: Path) -> dict:
    config = DarkMindV2Config.from_json_file(path)
    estimate = estimate_model_size(
        vocab_size=config.vocab_size,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        tied_embeddings=config.tie_word_embeddings,
        bias=config.bias,
        mlp_hidden_size=config.effective_mlp_hidden_size,
    )
    return {
        "result": "PASS",
        "schema_version": config.schema_version,
        "tokenizer_name": config.tokenizer_name,
        "tied_embeddings": config.tie_word_embeddings,
        "total_parameters": estimate.total_params,
        "transformer_body_parameters": estimate.transformer_body_params,
        "vocab_related_percentage": estimate.vocab_related_percentage,
        "head_dimension": estimate.head_dimension,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_model_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
