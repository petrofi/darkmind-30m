"""Detect mechanical and text-encoding failures in base-model generations."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.corpus.detect_mojibake import detect_text


SPECIAL_TOKEN_PATTERN = re.compile(r"<(?:pad|unk|s|/s)>|<\|(?:system|user|assistant|end)\|>")
SCRIPT_PATTERNS = {
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "greek": re.compile(r"[\u0370-\u03ff]"),
    "cjk": re.compile(r"[\u3400-\u9fff]"),
}
HARD_FAILURES_ALL_STAGES = {
    "invalid_unicode",
    "tokenizer_corruption",
    "decoder_defect",
    "normal_piece_mojibake",
    "confirmed_text_mojibake",
    "token_range_violation",
    "tokenizer_hash_mismatch",
    "tokenizer_artifact_mutation",
    "model_config_hash_mismatch",
    "corpus_manifest_mismatch",
    "invalid_corpus_unicode",
    "non_finite_logits",
    "non_finite_loss",
    "corrupted_checkpoint",
    "checkpoint_reload_failure",
}
EARLY_STAGE_WARNINGS = {
    "generated_invalid_utf8_byte_sequence",
    "replacement_character_from_generated_bytes",
    "unexpected_script",
    "mixed_script",
    "repetition",
    "empty_output",
    "special_token_leakage",
    "incoherent_output",
    "excessively_short_output",
    "excessively_long_output",
}
EARLY_STAGES = {"initialization", "stage1", "midpoint", "stage1_final", "best_validation"}
PUBLIC_RELEASE_PROMOTED = {
    "generated_invalid_utf8_byte_sequence",
    "replacement_character_from_generated_bytes",
    "unexpected_script",
    "mixed_script",
    "repetition",
    "empty_output",
    "special_token_leakage",
}
RESEARCH_PREVIEW_HARD_FAILURES = HARD_FAILURES_ALL_STAGES | {
    "generated_invalid_utf8_byte_sequence",
    "replacement_character_from_generated_bytes",
}


def repetition_ratio(token_ids: list[int]) -> float:
    if not token_ids:
        return 1.0
    return max(Counter(token_ids).values()) / len(token_ids)


def validate_text_health(
    text: str,
    token_ids: list[int],
    *,
    maximum_repetition_ratio: float = 0.35,
    token_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unicode_valid = True
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeError:
        unicode_valid = False
    unexpected_scripts = [
        script for script, pattern in SCRIPT_PATTERNS.items() if pattern.search(text)
    ]
    has_latin = any("LATIN" in unicodedata.name(character, "") for character in text if character.isalpha())
    mixed_script = len(unexpected_scripts) > 1 or (has_latin and bool(unexpected_scripts))
    special_tokens = SPECIAL_TOKEN_PATTERN.findall(text)
    ratio = repetition_ratio(token_ids)
    failures = []
    if not text.strip():
        failures.append("empty_output")
    if not unicode_valid:
        failures.append("invalid_unicode")
    if token_trace and token_trace.get("generated_invalid_utf8_byte_sequence"):
        failures.append("generated_invalid_utf8_byte_sequence")
    if token_trace and token_trace.get("replacement_character_from_generated_bytes"):
        failures.append("replacement_character_from_generated_bytes")
    elif "\ufffd" in text:
        failures.append("decoder_defect")
    if any(item.suspicious_substring != "\ufffd" for item in detect_text(text)):
        failures.append("confirmed_text_mojibake")
    if unexpected_scripts:
        failures.append("unexpected_script")
    if mixed_script:
        failures.append("mixed_script")
    if ratio > maximum_repetition_ratio:
        failures.append("repetition")
    if special_tokens:
        failures.append("special_token_leakage")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "unicode_valid": unicode_valid,
        "unicode_normalization": unicodedata.normalize("NFC", text) == text,
        "unexpected_scripts": unexpected_scripts,
        "mixed_script": mixed_script,
        "special_tokens": special_tokens,
        "repetition_ratio": ratio,
        "maximum_repetition_ratio": maximum_repetition_ratio,
        "text": text,
    }


def classify_generation_health(
    text: str,
    token_ids: list[int],
    *,
    checkpoint_stage: str,
    vocab_size: int = 24000,
    maximum_repetition_ratio: float = 0.35,
    decode_exception: str | None = None,
    tokenizer_hash_match: bool = True,
    model_config_hash_match: bool = True,
    logits_finite: bool = True,
    loss_finite: bool = True,
    token_trace: dict[str, Any] | None = None,
    tokenizer_artifact_mutation: bool = False,
    corpus_manifest_match: bool = True,
) -> dict[str, Any]:
    if checkpoint_stage not in {*EARLY_STAGES, "research_preview", "public_release"}:
        raise ValueError(f"unsupported checkpoint stage: {checkpoint_stage}")
    low_level = validate_text_health(
        text,
        token_ids,
        maximum_repetition_ratio=maximum_repetition_ratio,
        token_trace=token_trace,
    )
    findings = set(low_level["failures"])
    text_findings = detect_text(text)
    if token_trace is not None:
        if token_trace.get("generated_invalid_utf8_byte_sequence"):
            findings.add("generated_invalid_utf8_byte_sequence")
        if token_trace.get("replacement_character_from_generated_bytes"):
            findings.add("replacement_character_from_generated_bytes")
            findings.discard("decoder_defect")
        for issue in token_trace.get("normal_piece_issues", []):
            if issue.get("contains_replacement_character"):
                findings.add("tokenizer_corruption")
            if issue.get("mojibake_matches") and not issue.get("contains_replacement_character"):
                findings.add("normal_piece_mojibake")
        traced_tokens = token_trace.get("tokens", [])
        if any(
            item.get("type") == "control_or_special_token"
            and not (item.get("id") == 3 and item.get("position") == len(traced_tokens) - 1)
            for item in traced_tokens
        ):
            findings.add("special_token_leakage")
        if token_trace.get("replacement_character_from_generated_bytes") and not any(
            item.suspicious_substring != "\ufffd" for item in text_findings
        ):
            findings.discard("confirmed_text_mojibake")
    if decode_exception:
        findings.add("decoder_defect")
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        findings.add("token_range_violation")
    if not tokenizer_hash_match:
        findings.add("tokenizer_hash_mismatch")
    if tokenizer_artifact_mutation:
        findings.add("tokenizer_artifact_mutation")
    if not model_config_hash_match:
        findings.add("model_config_hash_mismatch")
    if not corpus_manifest_match:
        findings.add("corpus_manifest_mismatch")
    if not logits_finite:
        findings.add("non_finite_logits")
    if not loss_finite:
        findings.add("non_finite_loss")
    if checkpoint_stage == "public_release":
        hard_failures = findings & (HARD_FAILURES_ALL_STAGES | PUBLIC_RELEASE_PROMOTED)
        warnings = findings - hard_failures
    elif checkpoint_stage == "research_preview":
        hard_failures = findings & RESEARCH_PREVIEW_HARD_FAILURES
        warnings = findings - hard_failures
    else:
        hard_failures = findings & HARD_FAILURES_ALL_STAGES
        warnings = findings - hard_failures
    return {
        "result": "FAIL" if hard_failures else "PASS",
        "checkpoint_stage": checkpoint_stage,
        "hard_failures": sorted(hard_failures),
        "warnings": sorted(warnings),
        "findings": sorted(findings),
        "metrics": low_level,
    }


def enforce_generation_policy(records: list[dict[str, Any]]) -> None:
    failures = [
        {"id": record.get("id"), "hard_failures": record["policy"]["hard_failures"]}
        for record in records
        if record.get("policy", {}).get("hard_failures")
    ]
    if failures:
        raise RuntimeError(f"generation hard gate failed: {failures[:5]}")


def validate_generation_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    failures = [
        {"id": item.get("id"), "failures": item["health"]["failures"]}
        for item in results
        if item.get("health", {}).get("failures")
    ]
    return {"result": "FAIL" if failures else "PASS", "failures": failures, "generations": len(results)}
