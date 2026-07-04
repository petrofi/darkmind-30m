"""Strict DarkMind v2 corpus validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_corpus_manifest import build_manifest
    from .deduplicate_text import deduplicate_documents, read_documents
    from .detect_language import detect_language
    from .detect_mojibake import detect_text
    from .normalize_text import is_unsafe_control_char, normalize_text
except ImportError:  # pragma: no cover - CLI fallback
    from build_corpus_manifest import build_manifest
    from deduplicate_text import deduplicate_documents, read_documents
    from detect_language import detect_language
    from detect_mojibake import detect_text
    from normalize_text import is_unsafe_control_char, normalize_text


ALLOWED_LANGUAGES = {"tr", "en"}


def count_invalid_utf8(path: Path) -> int:
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return 1
    return 0


def validate_corpus(
    input_paths: list[Path],
    *,
    required_metadata: tuple[str, ...] = ("source", "license"),
    short_threshold: int = 40,
    long_threshold: int = 20000,
) -> tuple[dict[str, Any], list[str]]:
    documents = []
    invalid_utf8 = 0
    for path in input_paths:
        invalid_utf8 += count_invalid_utf8(path)
        if invalid_utf8 == 0:
            documents.extend(read_documents(path))

    total_lines = 0
    total_characters = 0
    normalization_changes = 0
    mojibake_count = 0
    replacement_characters = 0
    control_characters = 0
    empty_documents = 0
    very_short = 0
    very_long = 0
    missing_source = 0
    missing_license = 0
    language_counts: Counter[str] = Counter()

    for document in documents:
        text = document.text
        total_lines += max(1, len(text.splitlines()))
        total_characters += len(text)
        _, modifications = normalize_text(text)
        normalization_changes += len(modifications)
        findings = detect_text(text)
        mojibake_count += len(findings)
        replacement_characters += text.count("\ufffd")
        control_characters += sum(1 for char in text if is_unsafe_control_char(char))
        empty_documents += int(not text.strip())
        very_short += int(0 < len(text.strip()) < short_threshold)
        very_long += int(len(text) > long_threshold)
        language = detect_language(text)
        language_counts[language] += 1
        missing_source += int(not document.metadata.get("source"))
        missing_license += int(not document.metadata.get("license"))

    accepted, rejected, _ = deduplicate_documents(documents)
    duplicate_documents = sum(1 for item in rejected if item.reason == "exact_duplicate")
    near_duplicate_documents = sum(1 for item in rejected if item.reason == "near_duplicate")

    manifest_failures = 0
    try:
        manifest = build_manifest(input_paths, timestamp="1970-01-01T00:00:00+00:00")
        deterministic_hash = manifest.get("deterministic_content_hash")
        manifest_failures = 0 if deterministic_hash else 1
    except Exception:
        deterministic_hash = None
        manifest_failures = 1

    report = {
        "total_documents": len(documents),
        "total_lines": total_lines,
        "total_characters": total_characters,
        "utf8_failures": invalid_utf8,
        "unicode_normalization_changes": normalization_changes,
        "mojibake_detections": mojibake_count,
        "replacement_characters": replacement_characters,
        "control_characters": control_characters,
        "empty_documents": empty_documents,
        "language_distribution": dict(sorted(language_counts.items())),
        "duplicate_documents": duplicate_documents,
        "near_duplicate_documents": near_duplicate_documents,
        "accepted_after_deduplication": len(accepted),
        "very_short_documents": very_short,
        "very_long_documents": very_long,
        "source_metadata_missing": missing_source,
        "license_metadata_missing": missing_license,
        "deterministic_content_hash": deterministic_hash,
    }

    failures: list[str] = []
    if mojibake_count > 0:
        failures.append("mojibake count is greater than zero")
    if replacement_characters > 0:
        failures.append("replacement-character count is greater than zero")
    if invalid_utf8 > 0:
        failures.append("invalid UTF-8 exists")
    outside_languages = sorted(set(language_counts) - ALLOWED_LANGUAGES)
    if outside_languages:
        failures.append(f"language outside Turkish/English: {', '.join(outside_languages)}")
    if missing_source > 0 or missing_license > 0:
        failures.append("required source/license metadata is missing")
    if manifest_failures:
        failures.append("deterministic hashes cannot be produced")
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DarkMind v2 corpus inputs.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    report, failures = validate_corpus(args.input)
    payload = {"result": "FAIL" if failures else "PASS", "report": report, "failures": failures}
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
