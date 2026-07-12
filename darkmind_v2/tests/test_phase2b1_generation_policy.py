import json
from pathlib import Path

import pytest

from darkmind_v2.evaluation.validate_generation_health import (
    classify_generation_health,
    enforce_generation_policy,
)
from darkmind_v2.evaluation.trace_byte_fallback import trace_generated_tokens
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


def test_untrained_unexpected_script_is_warning_not_hard_failure() -> None:
    report = classify_generation_health("valid \u0417 text", [10, 11, 12], checkpoint_stage="initialization")
    assert report["result"] == "PASS"
    assert "unexpected_script" in report["warnings"]
    assert not report["hard_failures"]


@pytest.mark.parametrize(
    ("text", "token_ids", "kwargs", "failure"),
    [
        ("\ud800", [10], {}, "invalid_unicode"),
        ("broken \ufffd", [10], {}, "decoder_defect"),
        ("T\u00c3\u0192\u00c2\u00bcrkiye", [10], {}, "confirmed_text_mojibake"),
        ("valid", [24000], {}, "token_range_violation"),
        ("valid", [10], {"tokenizer_hash_match": False}, "tokenizer_hash_mismatch"),
    ],
)
def test_integrity_findings_remain_hard_failures(text, token_ids, kwargs, failure) -> None:
    report = classify_generation_health(
        text,
        token_ids,
        checkpoint_stage="initialization",
        **kwargs,
    )
    assert report["result"] == "FAIL"
    assert failure in report["hard_failures"]


def test_midpoint_and_final_preserve_script_warning_counts() -> None:
    midpoint = classify_generation_health("text \u0417", [10, 11], checkpoint_stage="midpoint")
    final = classify_generation_health("text \u0417", [10, 11], checkpoint_stage="stage1_final")
    assert midpoint["warnings"].count("unexpected_script") == 1
    assert final["warnings"].count("unexpected_script") == 1


def test_public_release_policy_promotes_script_warning() -> None:
    report = classify_generation_health("text \u0417", [10, 11], checkpoint_stage="public_release")
    assert report["result"] == "FAIL"
    assert "unexpected_script" in report["hard_failures"]


def test_hard_failure_stops_before_optimizer_logic() -> None:
    policy = classify_generation_health("broken \ufffd", [10], checkpoint_stage="initialization")
    with pytest.raises(RuntimeError, match="generation hard gate failed"):
        enforce_generation_policy([{"id": "fixture", "policy": policy}])


def test_valid_byte_fallback_sequence_decodes_strict_utf8() -> None:
    tokenizer = FrozenTokenizer()
    token_ids = [203, 196]  # <0xC3><0xBC> -> u with diaeresis
    decoded = tokenizer.decode(token_ids)
    trace = trace_generated_tokens(tokenizer, token_ids, decoded)
    assert decoded == "\u00fc"
    assert trace["byte_runs"][0]["strict_utf8"]["succeeds"] is True
    assert trace["generated_invalid_utf8_byte_sequence"] is False


def test_invalid_byte_fallback_sequence_reports_exact_offset_and_raw_output() -> None:
    tokenizer = FrozenTokenizer()
    token_ids = [73, 155]  # <0x41><0x93>: valid ASCII followed by an invalid UTF-8 start byte
    decoded = tokenizer.decode(token_ids)
    trace = trace_generated_tokens(tokenizer, token_ids, decoded)
    run = trace["invalid_utf8_byte_runs"][0]
    assert decoded == "A\ufffd"
    assert run["raw_bytes_hex"] == "41 93"
    assert run["strict_utf8"]["error_offset"] == 1
    assert run["strict_utf8"]["reason"] == "invalid start byte"
    assert run["sentencepiece_decoded"] == decoded
    assert trace["replacement_character_from_generated_bytes"] is True


def test_stage1_invalid_sampled_bytes_are_visible_warnings_without_duplicate_mojibake() -> None:
    tokenizer = FrozenTokenizer()
    token_ids = [155]
    decoded = tokenizer.decode(token_ids)
    trace = trace_generated_tokens(tokenizer, token_ids, decoded)
    report = classify_generation_health(
        decoded,
        token_ids,
        checkpoint_stage="stage1_final",
        token_trace=trace,
    )
    assert report["result"] == "PASS"
    assert "generated_invalid_utf8_byte_sequence" in report["warnings"]
    assert "replacement_character_from_generated_bytes" in report["warnings"]
    assert "confirmed_text_mojibake" not in report["findings"]
    assert decoded == "\ufffd"
    assert report["metrics"]["text"] == decoded


def test_public_release_blocks_invalid_sampled_bytes() -> None:
    tokenizer = FrozenTokenizer()
    token_ids = [155]
    decoded = tokenizer.decode(token_ids)
    trace = trace_generated_tokens(tokenizer, token_ids, decoded)
    report = classify_generation_health(
        decoded,
        token_ids,
        checkpoint_stage="public_release",
        token_trace=trace,
    )
    assert "generated_invalid_utf8_byte_sequence" in report["hard_failures"]
    assert "replacement_character_from_generated_bytes" in report["hard_failures"]


def test_research_preview_blocks_sampled_bytes_but_keeps_repetition_as_warning() -> None:
    tokenizer = FrozenTokenizer()
    token_ids = [155]
    decoded = tokenizer.decode(token_ids)
    trace = trace_generated_tokens(tokenizer, token_ids, decoded)
    report = classify_generation_health(
        decoded,
        token_ids,
        checkpoint_stage="research_preview",
        token_trace=trace,
        maximum_repetition_ratio=0.0,
    )
    assert "generated_invalid_utf8_byte_sequence" in report["hard_failures"]
    assert "replacement_character_from_generated_bytes" in report["hard_failures"]
    assert "repetition" in report["warnings"]


def test_research_preview_script_and_repetition_findings_are_quality_warnings() -> None:
    report = classify_generation_health(
        "Latin and \u0416", [10, 10, 10], checkpoint_stage="research_preview"
    )
    assert report["result"] == "PASS"
    assert {"unexpected_script", "mixed_script", "repetition"} <= set(report["warnings"])


def test_terminal_eos_is_completion_not_special_token_leakage() -> None:
    terminal_eos_trace = {
        "tokens": [
            {"position": 0, "id": 10, "type": "normal_sentencepiece_vocabulary_piece"},
            {"position": 1, "id": 3, "type": "control_or_special_token"},
        ],
        "normal_piece_issues": [],
    }
    completed = classify_generation_health(
        "text", [10, 3], checkpoint_stage="research_preview", token_trace=terminal_eos_trace
    )
    assert "special_token_leakage" not in completed["warnings"]

    leaked_trace = {
        "tokens": [{"position": 0, "id": 4, "type": "control_or_special_token"}],
        "normal_piece_issues": [],
    }
    leaked = classify_generation_health(
        "", [4], checkpoint_stage="research_preview", token_trace=leaked_trace
    )
    assert "special_token_leakage" in leaked["warnings"]


def test_normal_piece_replacement_and_mojibake_remain_hard_failures() -> None:
    replacement_trace = {
        "normal_piece_issues": [
            {"contains_replacement_character": True, "mojibake_matches": ["\ufffd"]}
        ]
    }
    replacement = classify_generation_health(
        "\ufffd", [10], checkpoint_stage="initialization", token_trace=replacement_trace
    )
    assert "tokenizer_corruption" in replacement["hard_failures"]

    mojibake_trace = {
        "normal_piece_issues": [
            {"contains_replacement_character": False, "mojibake_matches": ["T\u00c3"]}
        ]
    }
    mojibake = classify_generation_health(
        "T\u00c3", [10], checkpoint_stage="initialization", token_trace=mojibake_trace
    )
    assert "normal_piece_mojibake" in mojibake["hard_failures"]


def test_gate_config_documents_stage_aware_policy() -> None:
    config = json.loads(Path("darkmind_v2/config/base_quality_gates.json").read_text(encoding="utf-8"))
    policy = config["stage_aware_generation_policy"]
    assert "invalid_unicode" in policy["hard_failures_all_stages"]
    assert "generated_invalid_utf8_byte_sequence" in policy["initialization_warnings"]
    assert "unexpected_script" in policy["initialization_warnings"]
    assert "generated_invalid_utf8_byte_sequence" in policy["public_release_promoted_hard_failures"]
    assert "unexpected_script" in policy["public_release_promoted_hard_failures"]
