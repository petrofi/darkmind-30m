"""DarkMind v2 decoder-only causal language model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .configuration_darkmind_v2 import DarkMindV2Config
except ImportError:  # pragma: no cover - direct script compatibility
    from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config

try:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:  # pragma: no cover - only for minimal local environments
    class PreTrainedModel(nn.Module):  # type: ignore[no-redef]
        config_class = DarkMindV2Config

        def __init__(self, config: DarkMindV2Config) -> None:
            super().__init__()
            self.config = config

    @dataclass
    class CausalLMOutputWithPast:  # type: ignore[no-redef]
        loss: torch.Tensor | None
        logits: torch.Tensor
        past_key_values: None = None


class LayerNorm(nn.Module):
    def __init__(self, dimension: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension)) if bias else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DarkMindV2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attention_implementation = config.attention_implementation
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def _allowed_mask(
        self,
        batch_size: int,
        sequence_length: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        allowed = self.causal_mask[:, :, :sequence_length, :sequence_length]
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must have shape batch x sequence")
            allowed = allowed & attention_mask[:, None, None, :].to(dtype=torch.bool)
        return allowed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.split(self.n_embd, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        use_sdpa = self.attention_implementation in {"auto", "sdpa"} and hasattr(
            F, "scaled_dot_product_attention"
        )
        if self.attention_implementation == "sdpa" and not use_sdpa:
            raise RuntimeError("scaled-dot-product attention is unavailable in this PyTorch build")
        if use_sdpa:
            attention_bias = None
            is_causal = attention_mask is None
            if attention_mask is not None:
                attention_bias = self._allowed_mask(batch_size, sequence_length, attention_mask)
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_bias,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
            output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.n_embd)
            return self.resid_dropout(self.proj(output))

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = self._allowed_mask(batch_size, sequence_length, attention_mask)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.n_embd)
        return self.resid_dropout(self.proj(output))


class MLP(nn.Module):
    def __init__(self, config: DarkMindV2Config) -> None:
        super().__init__()
        hidden_size = config.effective_mlp_hidden_size
        self.fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(hidden_states))))


class TransformerBlock(nn.Module):
    def __init__(self, config: DarkMindV2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class DarkMindV2ForCausalLM(PreTrainedModel):
    """Small decoder-only model used to validate the DarkMind v2 pipeline."""

    config_class = DarkMindV2Config
    base_model_prefix = "darkmind_v2"
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True

    def __init__(self, config: DarkMindV2Config) -> None:
        config.validate()
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_deterministically(config.seed)
        self.tie_weights()

    def _initialize_deterministically(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, LayerNorm):
                    nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings is not True:
            raise ValueError("untied LM heads are forbidden")
        self.lm_head.weight = self.token_embedding.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        if value.num_embeddings != self.config.vocab_size:
            raise ValueError("replacement embedding vocabulary size is incompatible")
        self.token_embedding = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def embeddings_are_tied(self) -> bool:
        return self.token_embedding.weight is self.lm_head.weight

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if gradient_checkpointing_kwargs not in (None, {}, {"use_reentrant": False}):
            raise ValueError("only non-reentrant gradient checkpointing is supported")
        self.gradient_checkpointing = True
        self.config.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        self.config.gradient_checkpointing = False

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor | None, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape batch x sequence")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0 or sequence_length > self.config.block_size:
            raise ValueError(f"sequence length must be in [1, {self.config.block_size}]")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain tokens outside the configured vocabulary")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden_states = self.dropout(hidden_states)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden_states = checkpoint(
                    lambda states, current_block=block: current_block(states, attention_mask),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states, attention_mask)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        use_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not use_return_dict:
            return loss, logits
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        seed: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and (top_k <= 0 or top_k > self.config.vocab_size):
            raise ValueError("top_k must be in [1, vocab_size]")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the configured vocabulary")
        generated = input_ids
        generator = None
        if seed is not None:
            generator = torch.Generator(device=input_ids.device)
            generator.manual_seed(seed)
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.block_size :]
            logits = self(model_input).logits[:, -1, :] / temperature
            if not torch.all(torch.isfinite(logits)):
                raise FloatingPointError("generation logits are not finite")
            if do_sample:
                if top_k is not None:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
        self.train(was_training)
        return generated
