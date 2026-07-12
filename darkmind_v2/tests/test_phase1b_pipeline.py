import io
import json
import tarfile
from pathlib import Path

import pytest

from darkmind_v2.corpus.build_tokenizer_pilot_corpus import run_inventory
from darkmind_v2.corpus.download_phase1b_sources import compute_hashes, safe_extract_tar, safe_target
from darkmind_v2.corpus.validate_source_registry import validate_registry_payload
from darkmind_v2.corpus.validate_tokenizer_pilot_corpus import validate_processed_corpus


ROOT = Path("darkmind_v2")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase1b_source_registry_passes_existing_validator() -> None:
    payload = load_json(ROOT / "corpus" / "source_registry.phase1b.json")
    report, failures = validate_registry_payload(payload)
    assert failures == []
    assert report["source_count"] == 7
    assert report["approved_count"] == 7
    assert report["language_distribution"] == {"en": 2, "mixed_tr_en": 1, "tr": 4}


def test_phase1b_registry_uses_dated_wikimedia_snapshot() -> None:
    payload = load_json(ROOT / "corpus" / "source_registry.phase1b.json")
    wikimedia = payload["sources"][0]
    assert "latest" not in wikimedia["official_download_url"]
    assert wikimedia["snapshot_date"] == "2026-06-01"
    assert wikimedia["official_checksums"]["sha1"] == "761c077bd91561d84fca986a20d1ea30e909ef4e"
    assert wikimedia["expected_compressed_bytes"] < payload["hard_download_cap_bytes"]


def test_phase1b_download_helpers_hash_and_block_traversal(tmp_path: Path) -> None:
    payload = b"sample"
    sample = tmp_path / "sample.bin"
    sample.write_bytes(payload)
    hashes = compute_hashes(sample)
    assert hashes["sha256"] == "af2bdbe1aa9b6ec1e2ade1d694f41fc71a831d0268e9891562113d8a62add1bf"

    root = tmp_path / "extract"
    assert safe_target(root, "nested/file.txt") == (root / "nested" / "file.txt").resolve()
    with pytest.raises(ValueError):
        safe_target(root, "../evil.txt")


def test_phase1b_safe_tar_extraction_blocks_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        data = b"bad"
        member = tarfile.TarInfo("../evil.txt")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError):
        safe_extract_tar(archive_path, tmp_path / "out")


def test_phase1b_inventory_fails_closed_when_targets_are_not_met(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    archive_dir = data_dir / "raw" / "archives"
    archive_dir.mkdir(parents=True)
    archive_name = "python-docs-test.tar.bz2"
    archive_path = archive_dir / archive_name
    paragraph = "The Python function returns the documented value for this deterministic test paragraph."
    with tarfile.open(archive_path, "w:bz2") as archive:
        raw = paragraph.encode("utf-8")
        member = tarfile.TarInfo("python-3.14-docs-text/library/test.txt")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))

    registry = {
        "hard_download_cap_bytes": 1000000000,
        "sources": [
            {
                "source_id": "python_docs_test",
                "source_name": "Python docs test",
                "official_homepage": "https://docs.python.org/3/",
                "official_domains": ["docs.python.org"],
                "language": "en",
                "content_type": "technical_documentation",
                "license_id": "PSF-2.0-docs-and-0BSD-examples",
                "official_license_url": "https://docs.python.org/3/license.html",
                "attribution_requirements": "Preserve source path.",
                "redistribution_requirements": "Preserve license notice.",
                "commercial_use_status": "allowed_with_license_compliance",
                "modification_status": "allowed_with_license_compliance",
                "jurisdiction_warning": "",
                "source_version": "test",
                "snapshot_date": "2026-07-05",
                "estimated_download_size": "fixture",
                "estimated_download_size_bytes": archive_path.stat().st_size,
                "intended_sample_size_characters": 100,
                "checksum_available": False,
                "retrieval_method": "direct_download",
                "max_download_bytes": 1000000,
                "max_sample_characters": 1000,
                "approved": True,
                "approval_reason": "fixture",
                "risk_level": "low",
                "notes": "fixture",
                "filename": archive_name,
                "local_archive_name": archive_name,
                "archive_type": "tar_bz2_text",
                "processor": "python_docs_text_tar_bz2",
                "final_pilot_character_cap": 1000,
                "official_download_url": "https://docs.python.org/3/archives/fixture.tar.bz2",
            }
        ],
    }
    plan = {
        "target_normalized_characters": 10000,
        "language_mix": {
            "tr": {"target_characters": 6000},
            "en": {"target_characters": 4000},
        },
        "quality_gates": {
            "min_document_characters": 40,
            "max_document_characters": 20000,
        },
        "source_caps": {"max_single_source_characters": 20000},
    }
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    payload = run_inventory(registry_path, plan_path, data_dir, tmp_path / "reports", build=False)
    assert payload["result"] == "FAIL"
    assert any("outside 50M +/-1% target" in failure for failure in payload["failures"])
    assert (tmp_path / "reports" / "phase1b_source_gap_report.md").exists()


def test_phase1b_build_writes_deterministic_final_corpus_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    archive_dir = data_dir / "raw" / "archives"
    archive_dir.mkdir(parents=True)
    archive_name = "sentences_detailed.tar.bz2"
    archive_path = archive_dir / archive_name
    tr_sentences = [
        "Bu Turkce cumle dil hedefi icin yeterince uzun ve acik bir ornek metindir.",
        "Bu ikinci Turkce cumle kaynak tahsisini test etmek icin yazilmistir.",
    ]
    en_sentences = [
        "The English sentence is long enough for the deterministic corpus build test.",
        "This English sentence and the prior sentence validate the balanced split output.",
    ]
    rows = [
        f"{index}\t{language}\t{sentence}\tfixture_author"
        for index, (language, sentence) in enumerate(
            [("tur", text) for text in tr_sentences] + [("eng", text) for text in en_sentences],
            start=1,
        )
    ]
    raw = ("\n".join(rows) + "\n").encode("utf-8")
    with tarfile.open(archive_path, "w:bz2") as archive:
        member = tarfile.TarInfo("sentences_detailed.csv")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))

    tr_total = sum(len(text) for text in tr_sentences)
    en_total = sum(len(text) for text in en_sentences)
    registry = {
        "hard_download_cap_bytes": 1000000000,
        "sources": [
            {
                "source_id": "tatoeba_build_test",
                "source_name": "Tatoeba build test",
                "official_homepage": "https://tatoeba.org/",
                "official_download_url": "https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2",
                "official_license_url": "https://tatoeba.org/en/terms_of_use",
                "source_version": "fixture",
                "snapshot_date": "2026-07-09",
                "official_domains": ["tatoeba.org"],
                "language": "mixed_tr_en",
                "content_type": "dialogue_and_sentence_prose",
                "license_id": "CC-BY-2.0-FR",
                "attribution_manifest_fields": ["source_id", "sentence_id", "language", "username", "license_id", "snapshot_date"],
                "filename": archive_name,
                "local_archive_name": archive_name,
                "archive_type": "tar_bz2_tsv",
                "processor": "tatoeba_sentences_detailed_tar_bz2",
                "max_sample_characters": tr_total + en_total,
                "final_pilot_character_cap": tr_total + en_total,
                "approved": True,
            }
        ],
    }
    plan = {
        "target_normalized_characters": tr_total + en_total,
        "language_mix": {
            "tr": {"target_characters": tr_total},
            "en": {"target_characters": en_total},
        },
        "splits": {
            "seed": 17,
            "train": {"ratio": 0.9},
            "validation": {"ratio": 0.05},
            "test": {"ratio": 0.05},
        },
        "quality_gates": {
            "min_document_characters": 40,
            "max_document_characters": 20000,
        },
        "source_caps": {"max_single_source_characters": tr_total + en_total},
    }
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    first = run_inventory(registry_path, plan_path, data_dir, tmp_path / "reports", build=True)
    second = run_inventory(registry_path, plan_path, data_dir, tmp_path / "reports", build=True)

    assert first["result"] == "PASS"
    assert second["result"] == "PASS"
    assert first["output_hashes"] == second["output_hashes"]
    processed = data_dir / "processed"
    for filename in (
        "tokenizer_train.txt",
        "tokenizer_validation.txt",
        "tokenizer_eval.txt",
        "corpus_manifest.json",
        "attribution_manifest.jsonl",
        "rejected_documents.jsonl",
        "source_allocation.json",
        "split_manifest.json",
    ):
        assert (processed / filename).exists()
    assert "TOKENIZER CANDIDATE TRAINING IS READY FOR USER APPROVAL" in (
        tmp_path / "reports" / "phase1b_corpus_report.md"
    ).read_text(encoding="utf-8")
    validation_report, validation_failures = validate_processed_corpus(processed, plan_path)
    assert validation_failures == []
    assert validation_report["result"] == "PASS"
