"""Strict configuration for DarkMind v2 decoder-only models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from transformers import PretrainedConfig
except ImportError:  # pragma: no cover - exercised only in minimal environments
    class PretrainedConfig:  # type: ignore[no-redef]
        model_type = "darkmind_v2"

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


SCHEMA_VERSION = "darkmind-v2-model-config-v1"
TOKENIZER_NAME = "darkmind_v2_sp_bpe24k_v1"
SPECIAL_TOKEN_IDS = {
    "pad_token_id": 0,
    "unk_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 3,
    "system_token_id": 4,
    "user_token_id": 5,
    "assistant_token_id": 6,
    "end_token_id": 7,
}


class DarkMindV2Config(PretrainedConfig):
    """Configuration with fail-closed tokenizer and architecture validation."""

    model_type = "darkmind_v2"

    def __init__(
        self,
        *,
        vocab_size: int = 24000,
        block_size: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        normalization: str = "pre_layer_norm",
        position_embedding_type: str = "learned_absolute",
        attention_type: str = "causal_self_attention",
        tie_word_embeddings: bool = True,
        tokenizer_name: str = TOKENIZER_NAME,
        schema_version: str = SCHEMA_VERSION,
        initializer_range: float = 0.02,
        seed: int = 20260712,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        system_token_id: int = 4,
        user_token_id: int = 5,
        assistant_token_id: int = 6,
        end_token_id: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.normalization = normalization
        self.position_embedding_type = position_embedding_type
        self.attention_type = attention_type
        self.tie_word_embeddings = tie_word_embeddings
        self.tokenizer_name = tokenizer_name
        self.schema_version = schema_version
        self.initializer_range = initializer_range
        self.seed = seed
        self.unk_token_id = unk_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id
        self.end_token_id = end_token_id
        self.validate()

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in SPECIAL_TOKEN_IDS}

    def validate(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "mlp_ratio": self.mlp_ratio,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size != 24000:
            raise ValueError("vocab_size must match the frozen 24,000-token vocabulary")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.tokenizer_name != TOKENIZER_NAME:
            raise ValueError(f"unsupported tokenizer: {self.tokenizer_name!r}")
        if self.tie_word_embeddings is not True:
            raise ValueError("DarkMind v2 requires tied input/output token embeddings")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation != "gelu":
            raise ValueError("only GELU is supported")
        if self.normalization != "pre_layer_norm":
            raise ValueError("only pre-layer normalization is supported")
        if self.position_embedding_type != "learned_absolute":
            raise ValueError("only learned absolute positional embeddings are supported")
        if self.attention_type != "causal_self_attention":
            raise ValueError("only causal self-attention is supported")
        if self.special_token_ids != SPECIAL_TOKEN_IDS:
            raise ValueError("special-token IDs do not match the frozen tokenizer")

    def architecture_dict(self) -> dict[str, Any]:
        keys = (
            "activation",
            "attention_type",
            "bias",
            "block_size",
            "dropout",
            "initializer_range",
            "mlp_ratio",
            "model_type",
            "n_embd",
            "n_head",
            "n_layer",
            "normalization",
            "position_embedding_type",
            "schema_version",
            "seed",
            "tie_word_embeddings",
            "tokenizer_name",
            "vocab_size",
        )
        payload = {key: getattr(self, key) for key in keys}
        payload.update(self.special_token_ids)
        return payload

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DarkMindV2Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model config must be a JSON object")
        return cls(**payload)
