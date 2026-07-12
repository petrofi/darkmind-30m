import json
from collections import Counter
from pathlib import Path

from darkmind_v2.evaluation.audit_public_preview import (
    EXPECTED_CATEGORY_COUNTS,
    exact_repeated_ngram_loops,
    load_audit_prompts,
    longest_repeated_token_run,
    sampling_subset,
)


PROMPTS = Path("darkmind_v2/eval/public_preview_prompts.jsonl")


def test_public_preview_prompt_manifest_has_exact_controlled_distribution() -> None:
    prompts = load_audit_prompts(PROMPTS)
    assert len(prompts) == 200
    assert Counter(f"{item['language']}:{item['category']}" for item in prompts) == EXPECTED_CATEGORY_COUNTS
    assert len(sampling_subset(prompts)) == 50


def test_repetition_detector_records_exact_ngram_loops_and_longest_run() -> None:
    loops = exact_repeated_ngram_loops([10, 11, 10, 11, 12, 12, 12])
    assert {"start": 0, "n": 2, "token_ids": [10, 11]} in loops
    assert longest_repeated_token_run([10, 11, 11, 11, 12]) == 3
    assert exact_repeated_ngram_loops([1, 2, 3, 4, 5]) == []


def test_public_preview_gate_policy_is_fail_closed_for_bytes_and_disclosures() -> None:
    gates = json.loads(Path("darkmind_v2/config/public_research_preview_gates.json").read_text(encoding="utf-8"))
    hard = set(gates["hard_failures"])
    assert "generated_invalid_utf8_byte_sequence" in hard
    assert "replacement_character" in hard
    assert "missing_model_card_disclosure" in hard
    assert gates["generation"]["minimum_greedy_generations"] == 200
    assert gates["generation"]["minimum_seeded_generations"] == 500
