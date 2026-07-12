"""Base-model loss, perplexity, generation-health, and comparison utilities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.evaluation.generate_fixed_prompts import generate_fixed_prompts, load_prompts
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


@torch.no_grad()
def evaluate_validation_loss(model: torch.nn.Module, sequences: torch.Tensor) -> dict[str, float]:
    model.eval()
    output = model(sequences, labels=sequences)
    if output.loss is None or not torch.isfinite(output.loss):
        raise FloatingPointError("validation loss is not finite")
    loss = float(output.loss)
    return {"validation_loss": loss, "perplexity": math.exp(min(loss, 80.0))}


def compare_evaluation_manifests(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_by_id = {item["id"]: item for item in before.get("results", [])}
    after_by_id = {item["id"]: item for item in after.get("results", [])}
    shared = sorted(set(before_by_id) & set(after_by_id))
    changed = [
        prompt_id
        for prompt_id in shared
        if before_by_id[prompt_id].get("token_ids") != after_by_id[prompt_id].get("token_ids")
    ]
    return {"shared_prompts": len(shared), "changed_generations": changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_package", type=Path)
    parser.add_argument("--prompts", type=Path, default=Path("darkmind_v2/eval/fixed_base_prompts.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    tokenizer = FrozenTokenizer()
    model = load_model_package(args.model_package)
    prompts = load_prompts(args.prompts, limit=args.limit)
    manifest = generate_fixed_prompts(model, tokenizer, prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "generations": len(manifest["results"])}, indent=2))


if __name__ == "__main__":
    main()
