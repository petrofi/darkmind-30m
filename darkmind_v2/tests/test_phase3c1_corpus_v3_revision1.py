from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from darkmind_v2.corpus import download_corpus_v3_revision1 as downloader
from darkmind_v2.corpus.build_corpus_v3_revision1 import (
    classify_seed_category,
    next_inventory_root,
    select_jsonl_to_target,
    seed_split,
    technical_capacity_gate,
)
from darkmind_v2.corpus.finalize_corpus_v3_revision1 import (
    _select_unique_entries,
    canonical_category,
    compare_file_sets,
    target_within_tolerance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "darkmind_v2" / "config" / "corpus_v3_tranche1_revision1.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_revision1_config_uses_aggregate_quotas_and_approved_sources() -> None:
    config = load_config()
    result = downloader.validate_config(config)
    assert result["result"] == "PASS"
    assert config["allocation_policy"]["source_minimums_enabled"] is False
    assert all("target_tokens" not in source for source in config["sources"])
    assert config["targets"]["languages"] == {"tr": 60_000_000, "en": 40_000_000}
    assert config["targets"]["categories"] == {
        "general_prose": 85_000_000,
        "technical_educational": 15_000_000,
    }


def test_all_remote_sources_use_exact_dated_snapshots_and_official_host() -> None:
    config = load_config()
    for source in config["sources"]:
        if source["source_id"] not in downloader.REMOTE_SOURCE_IDS:
            continue
        assert source["snapshot"].startswith("2026-")
        assert "latest" not in source["snapshot"]
        assert source["official_host"] == "dumps.wikimedia.org"
        for item in source["files"]:
            assert "/2026" in item["url"]
            assert "latest" not in item["url"]
            assert item["md5"] and item["sha1"]


@pytest.mark.parametrize("value", [0, 3, 32])
def test_wikimedia_concurrency_above_two_or_below_one_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 2"):
        downloader.validate_transfer_concurrency(value)


def test_preflight_defaults_to_one_transfer_and_stays_under_raw_cap() -> None:
    config = load_config()
    plan = downloader.build_plan(config, "supplemental", 1)
    assert plan["transfers"] == 1
    assert plan["source_ids"] == sorted(downloader.SUPPLEMENTAL_SOURCE_IDS)
    assert plan["cumulative_raw_bytes"] < config["maximum_raw_download_bytes"]
    wikipedia = downloader.build_plan(config, "wikipedia", 2)
    assert wikipedia["prior_verified_raw_bytes"] == (
        config["already_verified_phase3c_bytes"] + plan["planned_download_bytes"]
    )
    assert wikipedia["cumulative_raw_bytes"] < config["maximum_raw_download_bytes"]


def test_retry_after_supports_delta_seconds_and_http_date() -> None:
    now = dt.datetime(2026, 7, 15, 12, 0, tzinfo=dt.timezone.utc)
    assert downloader.retry_after_seconds("17", now=now) == 17
    assert downloader.retry_after_seconds("Wed, 15 Jul 2026 12:00:23 GMT", now=now) == 23


def test_429_retry_after_is_honored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy = load_config()["rate_limit_policy"] | {"jitter_fraction": 0.0, "maximum_attempts": 2}
    item = {
        "filename": "fixture.bin",
        "url": "https://dumps.wikimedia.org/fixture/20260701/fixture.bin",
        "expected_bytes": 3,
        "sha1": hashlib.sha1(b"abc").hexdigest(),
    }
    attempts = 0

    def fake_download(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise downloader.RetryableDownloadError("HTTP 429", retry_after=19)
        partial = args[1]
        partial.write_bytes(b"abc")
        return {}

    sleeps: list[float] = []
    monkeypatch.setattr(downloader, "_download_once", fake_download)
    result = downloader.download_file(item, tmp_path / "fixture.bin", policy, sleep=sleeps.append)
    assert result["status"] == "downloaded_verified"
    assert sleeps == [19]


def test_partial_file_is_never_counted_as_complete(tmp_path: Path) -> None:
    item = {"filename": "fixture.bin", "expected_bytes": 4, "sha1": hashlib.sha1(b"abcd").hexdigest()}
    partial = tmp_path / "fixture.bin.partial"
    partial.write_bytes(b"ab")
    assert not (tmp_path / "fixture.bin").exists()
    with pytest.raises(ValueError, match="byte-count mismatch"):
        downloader._verify_file(partial, item)


def test_checksum_failure_preserves_partial_and_prevents_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = load_config()["rate_limit_policy"] | {"jitter_fraction": 0.0, "maximum_attempts": 1}
    item = {
        "filename": "fixture.bin",
        "url": "https://dumps.wikimedia.org/fixture/20260701/fixture.bin",
        "expected_bytes": 3,
        "sha1": hashlib.sha1(b"abc").hexdigest(),
    }

    def fake_download(*args, **kwargs):
        args[1].write_bytes(b"bad")
        return {}

    monkeypatch.setattr(downloader, "_download_once", fake_download)
    destination = tmp_path / "fixture.bin"
    with pytest.raises(ValueError, match="preserved partial"):
        downloader.download_file(item, destination, policy)
    assert not destination.exists()
    assert destination.with_name("fixture.bin.partial").read_bytes() == b"bad"


def test_seed_category_mapping_and_98_1_1_split_are_deterministic() -> None:
    config = load_config()
    technical = set(config["seed_inputs"]["technical_content_types"])
    assert classify_seed_category("technical_documentation", technical) == "technical_educational"
    assert classify_seed_category("educational_articles", technical) == "technical_educational"
    assert classify_seed_category("encyclopedic_articles", technical) == "general_prose"
    first = seed_split("doc:1", "a" * 64, config["split_policy"])
    assert first == seed_split("doc:1", "a" * 64, config["split_policy"])
    assignments = [
        seed_split(f"doc:{index}", hashlib.sha256(str(index).encode()).hexdigest(), config["split_policy"])
        for index in range(10_000)
    ]
    assert 9_700 <= assignments.count("train") <= 9_900
    assert 50 <= assignments.count("validation") <= 150
    assert 50 <= assignments.count("eval") <= 150


def test_technical_capacity_gate_enforces_source_caps() -> None:
    result = technical_capacity_gate(
        3_000_000,
        2_900_000,
        {"a": 20_000_000, "b": 4_000_000},
        {"a": 5_000_000, "b": 4_000_000},
    )
    assert result["supplemental_cap_eligible_tokens"] == {"a": 5_000_000, "b": 4_000_000}
    assert result["combined_unique_technical_capacity"] == 14_900_000
    assert result["result"] == "PASS"


def test_technical_capacity_gate_fails_below_98_percent_minimum() -> None:
    result = technical_capacity_gate(
        3_000_000,
        2_000_000,
        {"a": 9_699_999},
        {"a": 20_000_000},
    )
    assert result["minimum_tokens"] == 14_700_000
    assert result["combined_unique_technical_capacity"] == 14_699_999
    assert result["result"] == "FAIL"
    assert result["shortfall_to_minimum"] == 1


def test_interrupted_inventory_is_preserved_under_a_new_run_name(tmp_path: Path) -> None:
    first = tmp_path / "supplemental_inventory"
    first.mkdir()
    (first / "inventory_progress.json").write_text("{}", encoding="utf-8")
    assert next_inventory_root(tmp_path) == tmp_path / "supplemental_inventory_retry1"
    (tmp_path / "supplemental_inventory_retry1").mkdir()
    assert next_inventory_root(tmp_path) == tmp_path / "supplemental_inventory_retry2"


def test_deterministic_jsonl_allocator_uses_hash_order_and_source_cap(tmp_path: Path) -> None:
    path = tmp_path / "accepted.jsonl"
    records = [
        {"id": "z", "normalized_content_sha256": "c", "token_count": 7},
        {"id": "a", "normalized_content_sha256": "a", "token_count": 5},
        {"id": "b", "normalized_content_sha256": "b", "token_count": 6},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    selected, tokens = select_jsonl_to_target(path, 10, 12)
    assert [entry[1] for entry in selected] == ["a", "b"]
    assert tokens == 11
    with pytest.raises(ValueError, match="within the source cap"):
        select_jsonl_to_target(path, 13, 12)


def test_final_category_mapping_preserves_seed_and_aggregates_new_technical_sources() -> None:
    assert canonical_category("phase1b_seed", "general_prose") == "general_prose"
    assert canonical_category("phase1b_seed", "technical_educational") == "technical_educational"
    assert canonical_category("python_docs_en_3_14_6", "technical_documentation") == "technical_educational"
    assert canonical_category("wikimedia_enwiki_20260701", "general_prose") == "general_prose"
    with pytest.raises(ValueError, match="unsupported Phase 1B"):
        canonical_category("phase1b_seed", "unknown")


def test_aggregate_target_tolerance_is_inclusive() -> None:
    assert target_within_tolerance(99_000_000, 100_000_000, 1)
    assert target_within_tolerance(101_000_000, 100_000_000, 1)
    assert not target_within_tolerance(98_999_999, 100_000_000, 1)
    assert not target_within_tolerance(101_000_001, 100_000_000, 1)


def test_file_set_comparison_reports_first_hash_divergence(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.txt").write_text("same", encoding="utf-8")
    (right / "a.txt").write_text("same", encoding="utf-8")
    (left / "b.txt").write_text("left", encoding="utf-8")
    (right / "b.txt").write_text("right", encoding="utf-8")

    result = compare_file_sets(left, right, ("a.txt", "b.txt"))

    assert result["result"] == "FAIL"
    assert result["compared"] == {"a.txt": hashlib.sha256(b"same").hexdigest()}
    assert result["first_divergence"]["file"] == "b.txt"


def test_final_unique_selection_rejects_seed_duplicate_and_refills(tmp_path: Path) -> None:
    path = tmp_path / "accepted.jsonl"
    records = [
        {
            "id": "new:duplicate",
            "source_id": "approved_source",
            "source_native_id": "1",
            "normalized_content_sha256": "a",
            "token_count": 5,
        },
        {
            "id": "new:unique",
            "source_id": "approved_source",
            "source_native_id": "2",
            "normalized_content_sha256": "b",
            "token_count": 6,
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    selected, tokens, rejected = _select_unique_entries(
        path,
        select_jsonl_to_target(path, 11, 11)[0],
        {"seed:representative"},
        {"a": "seed:representative"},
        target_tokens=5,
        source_cap_tokens=11,
    )

    assert [entry[1] for entry in selected] == ["new:unique"]
    assert tokens == 6
    assert rejected[0]["reason"] == "final_cross_source_exact_duplicate"
    assert rejected[0]["representative_id"] == "seed:representative"
