"""Tokenizer round-trip checks for already-existing tokenizer directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> Any:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


@dataclass(frozen=True)
class RoundTripResult:
    text: str
    decoded: str
    exact_match: bool
    token_count: int


def encoding_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, list):
        return [int(item) for item in encoded]
    if hasattr(encoded, "ids"):
        return [int(item) for item in encoded.ids]
    if hasattr(encoded, "input_ids"):
        return [int(item) for item in encoded.input_ids]
    raise TypeError("Unsupported tokenizer encode result; expected list, .ids, or .input_ids")


def load_tokenizer(tokenizer_path: Path) -> TokenizerLike:
    try:
        from tokenizers import ByteLevelBPETokenizer, Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("The optional 'tokenizers' package is required to load tokenizer files.") from exc

    tokenizer_json = tokenizer_path / "tokenizer.json"
    if tokenizer_json.exists():
        return Tokenizer.from_file(str(tokenizer_json))

    vocab_json = tokenizer_path / "vocab.json"
    merges_txt = tokenizer_path / "merges.txt"
    if vocab_json.exists() and merges_txt.exists():
        return ByteLevelBPETokenizer(str(vocab_json), str(merges_txt))

    raise FileNotFoundError(f"No tokenizer.json or vocab.json+merges.txt found under {tokenizer_path}")


def run_roundtrip(tokenizer: TokenizerLike, samples: list[str]) -> list[RoundTripResult]:
    results: list[RoundTripResult] = []
    for sample in samples:
        ids = encoding_ids(tokenizer.encode(sample))
        decoded = tokenizer.decode(ids)
        results.append(
            RoundTripResult(
                text=sample,
                decoded=decoded,
                exact_match=decoded == sample,
                token_count=len(ids),
            )
        )
    return results


def read_samples(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(str(record.get("text") or record.get("prompt") or ""))
        return samples
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact tokenizer round-trip checks.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    results = run_roundtrip(tokenizer, read_samples(args.samples))
    payload = {
        "result": "PASS" if all(item.exact_match for item in results) else "FAIL",
        "samples": [asdict(item) for item in results],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
