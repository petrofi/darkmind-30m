from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from darkmind_v2.corpus.phase5b_source_lock import (
    CATEGORY_TARGETS,
    classify_phase5c_source_lock,
    validate_exclusive_category_allocation,
)
from darkmind_v2.corpus.validate_acquisition_manifest_v4 import validate_acquisition_plan
from darkmind_v2.corpus.validate_corpus_v4_attribution_manifest import validate_attribution_manifest
from darkmind_v2.corpus.validate_source_registry_v4 import PHASE5C_QUESTIONS, validate_registry


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "darkmind_v2" / "corpus"
CONFIG = ROOT / "darkmind_v2" / "config"
REPORTS = ROOT / "darkmind_v2" / "reports"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def registry() -> dict:
    return load(CORPUS / "source_registry.v4.candidates.json")


@pytest.fixture
def plan() -> dict:
    return load(CONFIG / "corpus_v4_acquisition_plan.json")


def test_all_phase5b_conditionals_have_complete_resolution_records(registry: dict) -> None:
    resolved = [item for item in registry["sources"] if "phase5c_resolution" in item]
    assert len(resolved) == 11
    for source in resolved:
        record = source["phase5c_resolution"]
        assert record["question_count"] == 15
        assert set(record["answers"]) == PHASE5C_QUESTIONS
        assert record["final_status"] == source["approval_state"]
        assert record["decision_reason"]
        assert record["next_action"]


def test_new_candidates_have_official_evidence_and_capacity_basis(registry: dict) -> None:
    discovered = [item for item in registry["sources"] if item.get("phase5c_discovery")]
    assert len(discovered) == 8
    for source in discovered:
        evidence = source["new_candidate_evidence"]
        assert evidence["official_artifact_url"].startswith("https://")
        assert evidence["official_metadata_url"].startswith("https://")
        assert evidence["official_license_url"].startswith("https://")
        assert evidence["exact_snapshot_version_commit_date"]
        assert evidence["capacity_basis"]
        assert evidence["source_cap_tokens"] <= 30_000_000


def test_dgt_turkish_assumption_is_rejected(registry: dict) -> None:
    dgt = next(item for item in registry["sources"] if item["id"] == "dgt_acquis_tr_en_20260722")
    assert dgt["approval_state"] == "rejected"
    assert dgt["capacity"]["conservative_tokens"] == 0
    assert "Turkish is not covered" in dgt["phase5c_correction"]["reason"]


def test_approved_capacity_is_exclusively_allocated(registry: dict) -> None:
    result = validate_exclusive_category_allocation(registry["sources"])
    assert result == {"expected_tokens": 10_000_000, "conservative_tokens": 6_300_000}
    bad = copy.deepcopy(registry["sources"])
    approved = next(item for item in bad if item["approval_state"] == "approved")
    first = next(iter(approved["approved_category_capacity"].values()))
    first["expected_tokens"] += 1
    with pytest.raises(ValueError, match="exclusively allocated"):
        validate_exclusive_category_allocation(bad)


def test_capacity_deficits_and_reserves_are_explicit(registry: dict) -> None:
    capacity = registry["capacity_model"]
    assert capacity["approved"] == {"expected_tokens": 10_000_000, "conservative_tokens": 6_300_000}
    assert capacity["formal_deficit"] == {"expected_tokens": 240_000_000, "conservative_tokens": 193_700_000}
    assert capacity["preferred_reserve_deficit"] == {
        "expected_tokens": 265_000_000,
        "conservative_tokens": 213_700_000,
    }
    assert capacity["conditional_capacity_is_scenario_only"] is True


def test_concentration_families_and_registry_validate(registry: dict) -> None:
    result = validate_registry(registry)
    assert result["result"] == "PASS"
    assert result["candidate_sources"] == 28
    assert result["concentration"]["result"] == "PASS"
    families = [item["source_family"] for item in registry["approved_acquisition_caps"]]
    assert len(families) == len(set(families))


def test_attribution_manifest_is_complete(registry: dict) -> None:
    manifest = load(CONFIG / "corpus_v4_attribution_manifest.json")
    result = validate_attribution_manifest(manifest, registry)
    assert result["result"] == "PASS"
    assert result["approved_source_coverage"] == "3/3"


def test_acquisition_maps_only_approved_sources(registry: dict, plan: dict) -> None:
    result = validate_acquisition_plan(plan, registry)
    assert result["result"] == "PASS"
    approved = {item["id"] for item in registry["sources"] if item["approval_state"] == "approved"}
    assert {item["source_id"] for item in plan["entries"]} == approved
    assert plan["acquisition_enabled"] is False


def test_conditional_source_execution_is_rejected(registry: dict, plan: dict) -> None:
    bad = copy.deepcopy(plan)
    conditional = next(item for item in registry["sources"] if item["approval_state"] == "conditional")
    bad["entries"][0]["source_id"] = conditional["id"]
    with pytest.raises(ValueError, match="unapproved source ID"):
        validate_acquisition_plan(bad, registry)


def test_category_lock_and_limited_reserve_classification() -> None:
    locked = {
        name: {"locked": True, "target": target, "conservative": target + int(target * 0.05)}
        for name, target in CATEGORY_TARGETS.items()
    }
    assert classify_phase5c_source_lock(250_000_000, 210_000_000, locked, True, True, True, True) == "LOCKED WITH LIMITED RESERVE"
    locked[next(iter(locked))]["locked"] = False
    assert classify_phase5c_source_lock(250_000_000, 210_000_000, locked, True, True, True, True) == "PARTIALLY LOCKED"


def test_bounded_samples_and_acquisition_are_disabled(registry: dict, plan: dict) -> None:
    samples = [
        item["phase5c_resolution"]["future_sample_plan"]
        for item in registry["sources"]
        if item.get("phase5c_resolution", {}).get("resolution_type") == "bounded_sample_required"
    ]
    assert samples
    assert all(item["execution_authorized"] is False and item["training_authorized"] is False for item in samples)
    assert plan["execution_authorized"] is False
    assert plan["execution_controls"]["extract_or_execute_downloaded_content"] is False
    generator = (CORPUS / "build_phase5c_source_resolution.py").read_text(encoding="utf-8")
    forbidden = ("import requests", "import urllib", "urlopen(", "subprocess.run(", "datasets.load_dataset")
    assert all(token not in generator for token in forbidden)


def test_phase5c_reports_and_decision_exist(registry: dict) -> None:
    names = {
        "phase5c_conditional_source_resolution.md",
        "phase5c_replacement_source_discovery.md",
        "phase5c_official_evidence_matrix.md",
        "phase5c_capacity_model.md",
        "phase5c_allocation_lock.md",
        "phase5c_license_and_attribution.md",
        "phase5c_source_lock_decision.md",
    }
    assert all((REPORTS / name).is_file() for name in names)
    decision = (REPORTS / "phase5c_source_lock_decision.md").read_text(encoding="utf-8")
    assert "## PARTIALLY LOCKED" in decision
    assert registry["source_lock_decision"] in decision
