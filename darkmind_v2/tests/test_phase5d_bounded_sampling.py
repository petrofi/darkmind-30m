from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from darkmind_v2.corpus.phase5b_source_lock import CATEGORY_TARGETS, classify_phase5d_source_lock
from darkmind_v2.corpus.run_phase5d_bounded_sampling import (
    BoundedRedirectHandler,
    selection_rank,
    verify_expected_checksum,
)
from darkmind_v2.corpus.validate_sample_inspection_authorization import (
    validate_sample_authorization,
    url_is_authorized,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "darkmind_v2" / "config"
CORPUS = ROOT / "darkmind_v2" / "corpus"
REPORTS = ROOT / "darkmind_v2" / "reports"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def authorization() -> dict:
    return load(CONFIG / "corpus_v4_sample_inspection_authorization.json")


@pytest.fixture
def registry() -> dict:
    return load(CORPUS / "source_registry.v4.candidates.json")


@pytest.fixture
def results() -> dict:
    return load(CONFIG / "corpus_v4_sample_inspection_results.json")


def test_sample_authorization_passes_hard_limits(authorization: dict, registry: dict) -> None:
    result = validate_sample_authorization(authorization, registry)
    assert result["result"] == "PASS"
    assert result["authorized_candidates"] == 6
    assert authorization["maximum_total_downloaded_bytes"] == 10_000_000_000
    assert authorization["authorized_entry_byte_sum"] == 1_500_000_000


def test_total_candidate_and_source_byte_limits_are_rejected(authorization: dict, registry: dict) -> None:
    bad = copy.deepcopy(authorization)
    bad["maximum_total_downloaded_bytes"] = 10_000_000_001
    with pytest.raises(ValueError, match="hard limit"):
        validate_sample_authorization(bad, registry)
    bad = copy.deepcopy(authorization)
    bad["maximum_candidates_sampled"] = 9
    with pytest.raises(ValueError, match="hard limit"):
        validate_sample_authorization(bad, registry)
    bad = copy.deepcopy(authorization)
    bad["entries"][0]["maximum_raw_bytes"] = 2_000_000_001
    with pytest.raises(ValueError, match="per-source"):
        validate_sample_authorization(bad, registry)


def test_official_url_and_redirect_domains_are_enforced(authorization: dict) -> None:
    entry = authorization["entries"][0]
    assert url_is_authorized(entry, entry["official_artifact_url"])
    assert not url_is_authorized(entry, "https://example.com/sample.json")
    handler = BoundedRedirectHandler(set(entry["permitted_redirect_domains"]))
    with pytest.raises(ValueError, match="unapproved domain"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://example.com/sample.json")


def test_checksum_verification_passes_and_rejects_mismatch(tmp_path: Path, authorization: dict) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"phase5d")
    digest = "4f39c830984040cd3aca8b859cfe76d1202157f3869535e4e3c49a78c8dd8934"
    entry = copy.deepcopy(authorization["entries"][3])
    entry["expected_checksum"] = {"algorithm": "sha256", "value": digest}
    assert verify_expected_checksum(entry, path, digest)["result"] == "PASS"
    with pytest.raises(ValueError, match="mismatch"):
        verify_expected_checksum(entry, path, "0" * 64)


def test_deterministic_selection_is_stable() -> None:
    inventory = ["c", "a", "b", "d"]
    first = sorted(inventory, key=lambda item: (selection_rank(20260723, item), item))
    second = sorted(reversed(inventory), key=lambda item: (selection_rank(20260723, item), item))
    assert first == second
    assert first != sorted(inventory)


def test_no_training_markers_are_complete(authorization: dict, results: dict) -> None:
    assert authorization["training_use"] is False
    assert authorization["production_tokenization"] is False
    assert authorization["execute_downloaded_content"] is False
    assert all(item["no_training"] is True for item in authorization["entries"])
    assert results["training_use"] is False
    assert results["production_tokenization"] is False
    assert results["downloaded_content_executed"] is False


def test_extraction_and_capacity_metric_schemas(results: dict) -> None:
    extraction = {
        "raw_records", "selected_records", "extracted_records", "opened_files",
        "parse_failures", "extraction_success_rate", "empty_extraction_rate",
        "parse_failure_rate", "ocr_dependent_documents", "ocr_used",
    }
    capacity = {
        "inventory_count", "sample_documents", "sample_coverage", "optimistic_tokens",
        "expected_tokens", "conservative_tokens", "uncertainty_band",
        "extraction_loss_rate", "quality_filter_loss_rate", "pii_secret_loss_rate",
        "internal_dedup_loss_rate", "corpus_v3_overlap_loss_rate",
    }
    for source in results["sources"]:
        assert extraction <= set(source["extraction"])
        assert capacity <= set(source["capacity"])
        assert 0 <= source["extraction"]["extraction_success_rate"] <= 1
        assert source["capacity"]["optimistic_tokens"] >= source["capacity"]["expected_tokens"]
        assert source["capacity"]["expected_tokens"] >= source["capacity"]["conservative_tokens"]


def test_corpus_v3_overlap_metrics_are_explicit(results: dict) -> None:
    assert results["corpus_v3_reference_documents"] == 447_127
    assert results["corpus_v3_near_reference_documents"] == 28_226
    for source in results["sources"]:
        overlap = source["deduplication"]
        assert overlap["corpus_v3_exact_overlap_documents"] >= 0
        assert overlap["corpus_v3_near_overlap_documents"] >= 0
        assert overlap["corpus_v3_reference_documents"] == 447_127


def test_phase5d_strategy_classification() -> None:
    categories = {
        key: {"target": target, "conservative": 0, "locked": False}
        for key, target in CATEGORY_TARGETS.items()
    }
    assert classify_phase5d_source_lock(
        23_145_265, 15_033_948, categories, True, True, True, True
    ) == "OPEN-ONLY LIMITED"


def test_turkish_and_strategy_scenarios_are_explicit() -> None:
    turkish = (REPORTS / "phase5d_turkish_data_feasibility.md").read_text(encoding="utf-8")
    strategy = (REPORTS / "phase5d_corpus_v4_strategy_scenarios.md").read_text(encoding="utf-8")
    assert "Scenario T1" in turkish and "Scenario T2" in turkish and "Scenario T3" in turkish
    assert "Approved Turkish general/educational capacity is **0 tokens**" in turkish
    assert all(f"Strategy {letter}" in strategy for letter in "ABCD")
    assert "15,033,948 tokens" in strategy


def test_full_acquisition_remains_disabled(registry: dict) -> None:
    plan = load(CONFIG / "corpus_v4_acquisition_plan.json")
    assert registry["source_lock_classification"] == "OPEN-ONLY LIMITED"
    assert registry["acquisition_enabled"] is False
    assert plan["acquisition_enabled"] is False
    assert plan["execution_authorized"] is False
    assert all(item["approval"]["execution_approved"] is False for item in plan["entries"])


def test_runtime_samples_are_outside_git(results: dict) -> None:
    runtime = Path(results["runtime_evidence_root"])
    assert not runtime.is_relative_to(ROOT)
    assert results["runtime_samples_committed"] is False
    assert all("raw_text" not in item and "accepted_records" not in item for item in results["sources"])
