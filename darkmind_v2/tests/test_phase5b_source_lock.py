from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from darkmind_v2.corpus.phase5b_source_lock import (
    CATEGORY_TARGETS,
    category_capacity,
    classify_source_lock,
    validate_capacity,
    validate_code_policy,
    validate_concentration,
    validate_official_evidence,
    validate_overlap_plan,
    validate_pii_policy,
    validate_state_transition,
)
from darkmind_v2.corpus.validate_acquisition_manifest_v4 import validate_acquisition_plan
from darkmind_v2.corpus.validate_source_registry_v4 import validate_registry


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "darkmind_v2" / "config"
CORPUS = ROOT / "darkmind_v2" / "corpus"
REPORTS = ROOT / "darkmind_v2" / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def registry() -> dict:
    return load_json(CORPUS / "source_registry.v4.candidates.json")


@pytest.fixture
def plan() -> dict:
    return load_json(CONFIG / "corpus_v4_acquisition_plan.json")


def source(registry: dict, source_id: str) -> dict:
    return next(item for item in registry["sources"] if item["id"] == source_id)


def test_source_state_transitions_are_explicit() -> None:
    validate_state_transition("approved", "conditional")
    validate_state_transition("conditional", "approved")
    with pytest.raises(ValueError, match="not allowed"):
        validate_state_transition("rejected", "approved")


def test_official_evidence_requirements(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "mdn_content_20260722"))
    del candidate["official_evidence"]["official_file_inventory"]
    with pytest.raises(ValueError, match="missing official evidence"):
        validate_official_evidence(candidate)


def test_content_and_database_license_are_distinct(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "postgresql_docs_18"))
    candidate["official_evidence"]["database_license"] = "unknown"
    with pytest.raises(ValueError, match="content/database"):
        validate_official_evidence(candidate)


def test_conservative_capacity_order_is_enforced(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "mdn_content_20260722"))
    validate_capacity(candidate)
    candidate["capacity"]["conservative_tokens"] = candidate["capacity"]["expected_tokens"] + 1
    with pytest.raises(ValueError, match="optimistic >= expected >= conservative"):
        validate_capacity(candidate)


def test_category_capacity_coverage_is_not_overclaimed(registry: dict) -> None:
    result = category_capacity(registry["sources"])
    assert set(result) == set(CATEGORY_TARGETS)
    assert result["technical_documentation"]["conservative"] == 3_800_000
    assert result["code_structured_text"]["conservative"] == 2_500_000
    assert all(item["locked"] is False for item in result.values())


def test_source_concentration_caps() -> None:
    entries = [
        {"source_id": "ok", "source_family": "a", "post_filter_cap_tokens": 10_000_000},
        {"source_id": "too_large", "source_family": "b", "post_filter_cap_tokens": 30_000_001},
    ]
    assert validate_concentration(entries)["violations"] == ["dataset:too_large"]


def test_acquisition_manifest_schema_and_plan_validate(registry: dict, plan: dict) -> None:
    schema = load_json(CONFIG / "corpus_v4_acquisition_manifest.schema.json")
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["execution_authorized"]["const"] is False
    result = validate_acquisition_plan(plan, registry)
    assert result["result"] == "PASS"
    assert result["approved_source_coverage"] == "3/3"


def test_unapproved_source_is_rejected(registry: dict, plan: dict) -> None:
    bad = copy.deepcopy(plan)
    conditional = source(registry, "govuk_content_ogl3_20260722")
    bad["entries"][0]["source_id"] = conditional["id"]
    with pytest.raises(ValueError, match="unapproved source ID"):
        validate_acquisition_plan(bad, registry)


def test_expected_checksum_is_required(registry: dict, plan: dict) -> None:
    bad = copy.deepcopy(plan)
    bad["entries"][0]["expected_checksum"]["value"] = ""
    with pytest.raises(ValueError, match="checksum"):
        validate_acquisition_plan(bad, registry)


def test_overlap_plan_schema(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "rust_official_docs_1_90"))
    validate_overlap_plan(candidate)
    del candidate["overlap_plan"]["near_dedup"]
    with pytest.raises(ValueError, match="overlap plan"):
        validate_overlap_plan(candidate)


def test_code_source_license_filtering(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "mdn_content_20260722"))
    candidate["code_policy"]["license_scope"] = "repository_visibility"
    with pytest.raises(ValueError, match="identifiable licensing"):
        validate_code_policy(candidate)


def test_high_pii_source_cannot_be_approved_without_mitigation(registry: dict) -> None:
    candidate = copy.deepcopy(source(registry, "stack_exchange_ccbysa_dump"))
    candidate["approval_state"] = "approved"
    candidate.pop("pii_mitigation")
    with pytest.raises(ValueError, match="high-PII"):
        validate_pii_policy(candidate)


def test_source_lock_classification(registry: dict) -> None:
    result = validate_registry(registry)
    assert result["source_lock_classification"] == "PARTIALLY LOCKED"
    locked_categories = {
        key: {"locked": True, "target": value, "expected": value, "conservative": value}
        for key, value in CATEGORY_TARGETS.items()
    }
    assert classify_source_lock(250_000_000, 200_000_000, locked_categories, True, True, True) == "LOCKED"
    assert classify_source_lock(0, 0, {}, False, False, False, evidence_materially_improved=False) == "NOT LOCKED"


def test_no_download_execution_exists(plan: dict) -> None:
    validator_text = (CORPUS / "validate_acquisition_manifest_v4.py").read_text(encoding="utf-8")
    forbidden = ("import requests", "import urllib", "urlopen(", "subprocess.run(", "Invoke-WebRequest")
    assert all(token not in validator_text for token in forbidden)
    assert plan["planning_only"] is True
    assert plan["execution_authorized"] is False
    assert plan["downloads_performed"] is False
    assert plan["scraping_performed"] is False


def test_continuation_pilot_is_equal_slice_and_unauthorized() -> None:
    pilot = load_json(CONFIG / "corpus_v4_continuation_pilot_spec.json")
    assert pilot["training_authorized"] is False
    assert pilot["pilot_data"]["unique_tokens"] == 10_000_000
    assert pilot["pilot_data"]["same_exact_slice_for_every_policy"] is True
    assert pilot["pilot_data"]["corpus_v3_repetition_allowed"] is False
    assert pilot["policies"][2]["authorized_by_default_after_source_lock"] is False


def test_required_reports_exist_and_decision_matches_registry(registry: dict) -> None:
    names = {
        "phase5b_official_source_due_diligence.md",
        "phase5b_license_matrix.md",
        "phase5b_capacity_model.md",
        "phase5b_corpus_v3_overlap_plan.md",
        "phase5b_allocation_lock.md",
        "phase5b_acquisition_storage_plan.md",
        "phase5b_manual_review_feedback_template.md",
        "phase5b_source_lock_decision.md",
    }
    assert all((REPORTS / name).is_file() for name in names)
    decision = (REPORTS / "phase5b_source_lock_decision.md").read_text(encoding="utf-8")
    assert "PARTIALLY LOCKED" in decision
    assert "REQUIRES ADDITIONAL LICENSE OR CAPACITY RESOLUTION" in decision
    assert validate_registry(registry)["source_lock_classification"] == "PARTIALLY LOCKED"
