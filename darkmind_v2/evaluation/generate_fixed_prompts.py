"""Generate deterministic continuations for the immutable TR/EN prompt set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.evaluation.validate_generation_health import validate_text_health
from darkmind_v2.evaluation.trace_byte_fallback import trace_generated_tokens
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


def load_prompts(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    prompts = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return prompts if limit is None else prompts[:limit]


def generate_fixed_prompts(
    model: DarkMindV2ForCausalLM,
    tokenizer: FrozenTokenizer,
    prompts: list[dict[str, Any]],
    *,
    max_new_tokens: int = 8,
    do_sample: bool = False,
    seed: int = 20260712,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    results = []
    for item in prompts:
        prompt_ids = tokenizer.encode(item["prompt"], add_bos=True)
        input_ids = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
        generated = model.generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            seed=seed,
        )[0].tolist()
        continuation_ids = generated[len(input_ids[0]) :]
        text = tokenizer.decode(continuation_ids)
        token_trace = trace_generated_tokens(tokenizer, continuation_ids, text)
        results.append(
            {
                "id": item["id"],
                "language": item["language"],
                "prompt": item["prompt"],
                "generation": text,
                "token_ids": continuation_ids,
                "token_trace": token_trace,
                "health": validate_text_health(
                    text,
                    continuation_ids,
                    maximum_repetition_ratio=float(item.get("maximum_repetition_ratio", 0.35)),
                    token_trace=token_trace,
                ),
            }
        )
    settings = {
        "max_new_tokens": max_new_tokens,
        "mode": "seeded_sampling" if do_sample else "greedy",
        "seed": seed,
    }
    core = {"settings": settings, "results": results}
    digest = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**core, "deterministic_content_hash": digest}
