"""Audit an already-existing tokenizer against DarkMind v2 Phase 0 gates."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_tokenizer_manifest import discover_tokenizer_files, load_vocab
    from .test_roundtrip import encoding_ids, load_tokenizer, read_samples, run_roundtrip
    from ..corpus.detect_mojibake import detect_text, looks_like_mojibake
except ImportError:  # pragma: no cover - CLI fallback
    from build_tokenizer_manifest import discover_tokenizer_files, load_vocab
    from test_roundtrip import encoding_ids, load_tokenizer, read_samples, run_roundtrip
    from darkmind_v2.corpus.detect_mojibake import detect_text, looks_like_mojibake


TURKISH_CHARS = set("çğıİöşüÇĞIıÖŞÜ")
ASCII_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1)
    return ordered[index]


def script_name(char: str) -> str:
    if char in TURKISH_CHARS:
        return "turkish_latin"
    if char in ASCII_LETTERS:
        return "english_ascii"
    name = unicodedata.name(char, "")
    if "HEBREW" in name:
        return "hebrew"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
        return "cjk_or_japanese"
    if char.isdigit():
        return "digit"
    if char.isspace():
        return "space"
    return "other"


def token_script_distribution(tokens: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for token in tokens:
        for char in token:
            counts[script_name(char)] += 1
    return counts


def malformed_tokens(tokens: list[str]) -> list[str]:
    malformed = []
    for token in tokens:
        if "\ufffd" in token or any(unicodedata.category(char) == "Cc" for char in token):
            malformed.append(token)
    return malformed


def compare_manifest(tokenizer_path: Path, manifest_path: Path | None) -> tuple[bool, list[str]]:
    if manifest_path is None:
        return True, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("tokenizer_file_hashes", {})
    actual_hashes = discover_tokenizer_files(tokenizer_path)
    mismatches = [
        path for path, digest in expected_hashes.items() if actual_hashes.get(path) != digest
    ]
    extra = [path for path in actual_hashes if path not in expected_hashes]
    return not mismatches and not extra, mismatches + extra


def audit_tokenizer(tokenizer_path: Path, sample_path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    samples = read_samples(sample_path)
    roundtrip_results = run_roundtrip(tokenizer, samples)
    vocab = load_vocab(tokenizer_path)
    tokens = sorted(vocab, key=vocab.get) if vocab else []
    sequence_lengths = []
    total_tokens = 0
    total_chars = 0
    total_words = 0
    unknown_tokens = 0
    unk_id = vocab.get("<unk>") if vocab else None

    for sample in samples:
        ids = encoding_ids(tokenizer.encode(sample))
        sequence_lengths.append(len(ids))
        total_tokens += len(ids)
        total_chars += max(1, len(sample))
        total_words += max(1, len(sample.split()))
        if unk_id is not None:
            unknown_tokens += sum(1 for token_id in ids if token_id == unk_id)

    mojibake_vocab = [token for token in tokens if looks_like_mojibake(token) or detect_text(token)]
    replacement_tokens = [token for token in tokens if "\ufffd" in token]
    manifest_match, manifest_mismatches = compare_manifest(tokenizer_path, manifest_path)
    unknown_ratio = unknown_tokens / total_tokens if total_tokens else 0.0
    longest_samples = sorted(
        [{"sample": sample, "token_count": length} for sample, length in zip(samples, sequence_lengths)],
        key=lambda item: item["token_count"],
        reverse=True,
    )[:10]
    special_tokens = {token: token_id for token, token_id in vocab.items() if token.startswith("<") and token.endswith(">")}

    failures: list[str] = []
    if mojibake_vocab:
        failures.append("mojibake vocabulary token detected")
    if replacement_tokens:
        failures.append("replacement-character vocabulary token detected")
    if any(not item.exact_match and any(char in TURKISH_CHARS for char in item.text) for item in roundtrip_results):
        failures.append("Turkish round-trip failure")
    if any(not item.exact_match and not any(char in TURKISH_CHARS for char in item.text) for item in roundtrip_results):
        failures.append("English round-trip failure")
    if unknown_ratio > 0.001:
        failures.append("unknown-token ratio above 0.1%")
    if manifest_path is not None and not manifest_match:
        failures.append("tokenizer manifest mismatch")

    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "vocabulary_size": len(vocab) if vocab else None,
        "special_tokens": special_tokens,
        "tokens_per_character": total_tokens / total_chars if total_chars else 0.0,
        "tokens_per_word": total_tokens / total_words if total_words else 0.0,
        "unknown_token_ratio": unknown_ratio,
        "round_trip_failures": [item.__dict__ for item in roundtrip_results if not item.exact_match],
        "malformed_tokens": malformed_tokens(tokens),
        "mojibake_tokens": mojibake_vocab,
        "replacement_character_tokens": replacement_tokens,
        "script_distribution": dict(token_script_distribution(tokens)),
        "turkish_character_coverage": {char: any(char in token for token in tokens) for char in sorted(TURKISH_CHARS)},
        "english_ascii_coverage": {char: any(char in token for token in tokens) for char in sorted(ASCII_LETTERS)},
        "longest_tokenized_samples": longest_samples,
        "sequence_length_percentiles": {
            "p50": percentile(sequence_lengths, 50),
            "p90": percentile(sequence_lengths, 90),
            "p95": percentile(sequence_lengths, 95),
            "p99": percentile(sequence_lengths, 99),
        },
        "manifest_mismatches": manifest_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an already-existing tokenizer for DarkMind v2 readiness.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit_tokenizer(args.tokenizer, args.sample, args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
