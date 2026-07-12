"""Trace generated token IDs through SentencePiece byte-fallback decoding."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from darkmind_v2.corpus.detect_mojibake import detect_text
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


BYTE_PIECE_PATTERN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def script_for_character(character: str) -> str:
    if character.isspace():
        return "whitespace"
    category = unicodedata.category(character)
    if category.startswith(("P", "N", "S")):
        return "common"
    name = unicodedata.name(character, "UNNAMED")
    for script in ("LATIN", "CYRILLIC", "ARABIC", "HEBREW", "GREEK"):
        if script in name:
            return script.lower()
    return "other"


def classify_token(tokenizer: FrozenTokenizer, token_id: int) -> dict[str, Any]:
    if token_id < 0 or token_id >= tokenizer.vocab_size:
        return {"id": token_id, "piece": None, "type": "unknown_or_invalid_token"}
    piece = tokenizer.id_to_piece(token_id)
    byte_match = BYTE_PIECE_PATTERN.fullmatch(piece)
    if byte_match:
        piece_type = "byte_fallback_piece"
    elif tokenizer.is_unknown_id(token_id):
        piece_type = "unknown_or_invalid_token"
    elif tokenizer.is_control_id(token_id):
        piece_type = "control_or_special_token"
    elif piece == "\u2581":
        piece_type = "whitespace_or_meta_piece"
    else:
        piece_type = "normal_sentencepiece_vocabulary_piece"
    return {
        "id": token_id,
        "piece": piece,
        "type": piece_type,
        "byte_hex": byte_match.group(1).upper() if byte_match else None,
    }


def _strict_utf8(raw_bytes: bytes) -> dict[str, Any]:
    try:
        decoded = raw_bytes.decode("utf-8", errors="strict")
        return {"succeeds": True, "decoded": decoded, "error_offset": None, "error_end": None, "reason": None}
    except UnicodeDecodeError as exc:
        return {
            "succeeds": False,
            "decoded": None,
            "error_offset": exc.start,
            "error_end": exc.end,
            "reason": exc.reason,
        }


def trace_generated_tokens(
    tokenizer: FrozenTokenizer,
    token_ids: list[int],
    decoded_output: str,
) -> dict[str, Any]:
    tokens = [{"position": position, **classify_token(tokenizer, token_id)} for position, token_id in enumerate(token_ids)]
    byte_runs = []
    run_start: int | None = None
    run_bytes: list[int] = []
    for position in range(len(tokens) + 1):
        token = tokens[position] if position < len(tokens) else None
        if token and token["type"] == "byte_fallback_piece":
            if run_start is None:
                run_start = position
                run_bytes = []
            run_bytes.append(int(token["byte_hex"], 16))
            continue
        if run_start is None:
            continue
        raw_bytes = bytes(run_bytes)
        run_ids = token_ids[run_start:position]
        sentencepiece_decoded = tokenizer.decode(run_ids)
        strict = _strict_utf8(raw_bytes)
        byte_runs.append(
            {
                "token_start": run_start,
                "token_end_exclusive": position,
                "token_ids": run_ids,
                "pieces": [item["piece"] for item in tokens[run_start:position]],
                "raw_bytes_hex": raw_bytes.hex(" ").upper(),
                "strict_utf8": strict,
                "sentencepiece_decoded": sentencepiece_decoded,
                "sentencepiece_decoded_escaped": ascii(sentencepiece_decoded),
                "sentencepiece_emitted_replacement_character": "\ufffd" in sentencepiece_decoded,
            }
        )
        run_start = None

    normal_piece_issues = []
    for token in tokens:
        if token["type"] != "normal_sentencepiece_vocabulary_piece":
            continue
        piece = token["piece"]
        findings = detect_text(piece)
        if "\ufffd" in piece or findings:
            normal_piece_issues.append(
                {
                    "position": token["position"],
                    "id": token["id"],
                    "piece": piece,
                    "contains_replacement_character": "\ufffd" in piece,
                    "mojibake_matches": [item.suspicious_substring for item in findings],
                }
            )

    invalid_runs = [run for run in byte_runs if not run["strict_utf8"]["succeeds"]]
    replacement_from_bytes = bool(invalid_runs) and "\ufffd" in decoded_output
    return {
        "tokens": tokens,
        "byte_runs": byte_runs,
        "invalid_utf8_byte_runs": invalid_runs,
        "generated_invalid_utf8_byte_sequence": bool(invalid_runs),
        "replacement_character_from_generated_bytes": replacement_from_bytes,
        "normal_piece_issues": normal_piece_issues,
        "token_range_ok": all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids),
    }


def audit_normal_vocabulary(tokenizer: FrozenTokenizer) -> dict[str, Any]:
    issues = []
    normal_piece_count = 0
    for token_id in range(tokenizer.vocab_size):
        token = classify_token(tokenizer, token_id)
        if token["type"] != "normal_sentencepiece_vocabulary_piece":
            continue
        normal_piece_count += 1
        piece = token["piece"]
        findings = detect_text(piece)
        if "\ufffd" in piece or findings:
            issues.append(
                {
                    "id": token_id,
                    "piece": piece,
                    "contains_replacement_character": "\ufffd" in piece,
                    "mojibake_matches": [item.suspicious_substring for item in findings],
                }
            )
    return {
        "normal_piece_count": normal_piece_count,
        "issues": issues,
        "result": "PASS" if not issues else "FAIL",
    }
