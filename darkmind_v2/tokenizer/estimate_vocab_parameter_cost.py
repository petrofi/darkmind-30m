"""Estimate vocabulary parameter cost for DarkMind v2 tokenizer candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_VOCAB_SIZES = (12000, 16000, 24000)
DEFAULT_DIMS = (384, 512)
DEFAULT_MODEL_TARGETS = (45_000_000, 60_000_000)
DTYPE_BYTES = {"fp32": 4, "fp16": 2}


@dataclass(frozen=True)
class VocabCost:
    vocab_size: int
    embedding_dim: int
    tied_output: bool
    embedding_parameters: int
    output_head_parameters: int
    combined_parameters: int
    fp32_storage_bytes: int
    fp16_storage_bytes: int
    percent_of_45m: float
    percent_of_60m: float


def estimate_vocab_cost(
    vocab_size: int,
    embedding_dim: int,
    *,
    tied_output: bool,
    model_targets: tuple[int, int] = DEFAULT_MODEL_TARGETS,
) -> VocabCost:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if len(model_targets) != 2 or any(target <= 0 for target in model_targets):
        raise ValueError("model_targets must contain two positive values")

    embedding_parameters = vocab_size * embedding_dim
    output_head_parameters = 0 if tied_output else vocab_size * embedding_dim
    combined_parameters = embedding_parameters + output_head_parameters
    return VocabCost(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        tied_output=tied_output,
        embedding_parameters=embedding_parameters,
        output_head_parameters=output_head_parameters,
        combined_parameters=combined_parameters,
        fp32_storage_bytes=combined_parameters * DTYPE_BYTES["fp32"],
        fp16_storage_bytes=combined_parameters * DTYPE_BYTES["fp16"],
        percent_of_45m=round(combined_parameters / model_targets[0] * 100, 4),
        percent_of_60m=round(combined_parameters / model_targets[1] * 100, 4),
    )


def build_cost_table(
    vocab_sizes: Iterable[int] = DEFAULT_VOCAB_SIZES,
    dims: Iterable[int] = DEFAULT_DIMS,
) -> list[dict[str, int | bool | float]]:
    rows: list[dict[str, int | bool | float]] = []
    for vocab_size in vocab_sizes:
        for dim in dims:
            for tied_output in (True, False):
                rows.append(asdict(estimate_vocab_cost(vocab_size, dim, tied_output=tied_output)))
    return rows


def _parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all values must be positive")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate DarkMind v2 tokenizer vocabulary parameter costs.")
    parser.add_argument("--vocab-sizes", type=_parse_int_list, default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--dims", type=_parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "result": "PASS",
        "model_targets": {"small": 45_000_000, "medium": 60_000_000},
        "rows": build_cost_table(args.vocab_sizes, args.dims),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
