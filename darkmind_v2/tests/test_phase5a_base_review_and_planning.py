from __future__ import annotations

import json
from pathlib import Path

from darkmind_v2.corpus.validate_source_registry_v4 import validate_registry
from darkmind_v2.evaluation.build_manual_review_packet import (
    review_item,
    select_stratified_records,
    validate_packet_schema,
)
from darkmind_v2.evaluation.phase5a_planning import (
    assert_nondestructive_archival_plan,
    choose_next_phase,
    classify_public_release,
    validate_archival_inventory,
    validate_automatic_metric_schema,
    validate_continuation_policies,
    validate_corpus_v4_targets,
)
from darkmind_v2.evaluation.run_base_quality_suite import (
    expand_suite,
    prompt_manifest_hash,
    validate_prompt_suite,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "darkmind_v2" / "config"
CORPUS = ROOT / "darkmind_v2" / "corpus"
REPORTS = ROOT / "darkmind_v2" / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_deterministic_base_quality_prompt_suite() -> None:
    config = load_json(CONFIG / "base_quality_suite_v1.json")
    first = expand_suite(config)
    second = expand_suite(config)
    assert first == second
    assert validate_prompt_suite(config, first)["prompt_count"] == 440
    assert prompt_manifest_hash(first) == "c82db48e4276d4a9a4d90ea1752956a55848869c55ea0e2ce590358eb39f9197"


def test_prompt_provenance_forbids_training_corpus_verbatim_content() -> None:
    config = load_json(CONFIG / "base_quality_suite_v1.json")
    prompts = expand_suite(config)
    policy = config["prompt_policy"]
    assert policy["source_type"] == "original_template"
    assert policy["training_documents_copied_verbatim"] is False
    assert policy["benchmark_prompts_used"] is False
    assert all(item["source_type"] == "original_template" for item in prompts)
    assert all("source_url" not in item and "source_document_id" not in item for item in prompts)


def test_manual_review_packet_is_balanced_and_blank() -> None:
    categories = [item["id"] for item in load_json(CONFIG / "base_quality_suite_v1.json")["categories"]]
    manifests = {}
    for mode in ("greedy", "seeded_sampling"):
        records = []
        for category in categories:
            for index in range(20):
                records.append(
                    {
                        "prompt_id": f"{category}-{index}",
                        "category": category,
                        "language": "tr" if category.startswith("turkish") else "en",
                        "prompt": f"Original prompt {category} {index}",
                        "output": f"Raw continuation {mode} {index}",
                        "policy": {"warnings": []},
                        "exact_repeated_ngram_loops": [],
                        "eos_completed": False,
                    }
                )
        manifests[mode] = {"results": records}
    items = [review_item(record) for record in select_stratified_records(manifests, 150)]
    validate_packet_schema(items)
    assert {item["decoding_mode"] for item in items} == {"greedy", "seeded_sampling"}
    assert all(all(score is None for score in item["scores"].values()) for item in items)


def test_automatic_metric_schema_rejects_semantic_claims() -> None:
    metrics = {
        "generation_count": 440,
        "repetition_rate": 0.6,
        "exact_loop_rate": 0.56,
        "eos_completion_rate": 0.02,
        "empty_output_rate": 0.0,
        "output_tokens": {"minimum": 1, "mean": 47.0, "p50": 48, "p90": 48, "maximum": 48},
        "mean_unique_token_ratio": 0.18,
        "language_id_consistency_rate": 1.0,
        "language_switch_error_rate": 0.0,
        "unicode_health": {"invalid_utf8_sequences": 0, "replacement_characters": 0, "mojibake_outputs": 0},
        "special_token_leakage_count": 0,
        "mean_prompt_token_overlap_ratio": 0.23,
        "code_structure_valid_rate": 0.65,
        "punctuation_completion_rate": 0.05,
        "human_semantic_quality_claimed": False,
    }
    validate_automatic_metric_schema(metrics)
    metrics["human_semantic_quality_claimed"] = True
    try:
        validate_automatic_metric_schema(metrics)
    except ValueError as error:
        assert "semantic quality" in str(error)
    else:
        raise AssertionError("automatic semantic-quality claim should fail")


def test_corpus_v4_targets_are_complete_and_planning_only() -> None:
    payload = load_json(CONFIG / "corpus_v4_targets.json")
    validate_corpus_v4_targets(payload)
    assert payload["tranche_2"]["target_unique_tokens"] == 200_000_000
    assert payload["tranche_2"]["language_projection"]["turkish_remains_primary"] is True


def test_source_registry_v4_candidate_contract() -> None:
    result = validate_registry(load_json(CORPUS / "source_registry.v4.candidates.json"))
    assert result["candidate_sources"] == 20
    assert result["approval_counts"] == {"approved": 5, "conditional": 10, "deferred": 2, "rejected": 3}
    assert result["approved_plus_conditional_capacity"] == 338_000_000
    assert result["downloads_performed"] is False


def test_continuation_policy_candidates_are_controlled_and_not_authorized() -> None:
    payload = load_json(CONFIG / "base_v1_continuation_policy_candidates.json")
    validate_continuation_policies(payload)
    assert payload["controlled_comparison"]["new_unique_data_slice_tokens"] == 4_997_120
    assert payload["controlled_comparison"]["training_authorized"] is False


def test_public_release_classification_separates_packaging_from_advice() -> None:
    result = classify_public_release(0.60, 0.568, 0.0205, model_weight_license_finalized=False)
    assert result["technically_uploadable"] is True
    assert result["publicly_advisable"] is False
    assert "model_weight_license_unresolved" in result["blockers"]


def test_archival_inventory_schema() -> None:
    validate_archival_inventory(
        [
            {
                "absolute_path": r"C:\DarkMindRuntime\phase4f",
                "size_bytes": 2_639_419_028,
                "purpose": "final first-pass evidence",
                "required_for_reproducibility": True,
                "required_for_future_training": True,
                "safe_to_archive": True,
                "safe_to_delete_only_after_verified_archive": False,
                "must_remain_local": True,
                "depends_on": r"C:\DarkMindRuntime\phase4d",
            }
        ]
    )


def test_archival_plan_contains_no_runtime_deletion_instruction() -> None:
    text = (REPORTS / "phase5a_external_ssd_archival_plan.md").read_text(encoding="utf-8")
    assert "never delete automatically" in text.lower()
    assert_nondestructive_archival_plan(text)


def test_final_decision_prefers_unique_data_and_reports_stop_scope() -> None:
    assert choose_next_phase(True, False, False, True) == "expand_unique_corpus"
    assert choose_next_phase(True, False, True, True) == "architecture_capacity_review"
    assert choose_next_phase(False, True, False, False) == "controlled_sft_preparation"
    report = (REPORTS / "phase5a_next_phase_decision.md").read_text(encoding="utf-8")
    assert "DARKMIND V2 BASE V1 REQUIRES CORPUS V4 EXPANSION BEFORE INSTRUCTION TUNING OR PUBLIC RELEASE" in report
    assert "No training was run" in report
