from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = emb_dim // self.n_head

        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        y = attention_weights @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)

        y = self.proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # GPT tarzı weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len}, block_size değerinden büyük: {self.config.block_size}"
            )

        token_emb = self.token_embedding(idx)

        positions = torch.arange(0, seq_len, device=idx.device)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())