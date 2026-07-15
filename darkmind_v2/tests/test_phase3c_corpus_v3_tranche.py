import bz2
import copy
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from darkmind_v2.corpus.build_corpus_v3_tranche import (
    ExtractedDocument,
    ShardWriter,
    _source_record,
    allocate_documents,
    clean_wiki_markup,
    detect_language_confidence,
    deterministic_split,
    find_near_duplicate,
    iter_python_documents,
    iter_wikipedia_documents,
    load_contamination_material,
    load_phase1b_hashes,
    normalized_hash_text,
    quality_filter,
    quota_status,
    simhash64,
    simhash_bands,
    stable_document_id,
    tokenize_allocation,
)
from darkmind_v2.corpus.download_corpus_v3_tranche import (
    _verify_file,
    parse_checksum_manifest,
    validate_config,
)
from darkmind_v2.tokenizer.load_frozen_tokenizer import verify_frozen_tokenizer


ROOT = Path("darkmind_v2")
CONFIG_PATH = ROOT / "config" / "corpus_v3_tranche1.json"
REGISTRY_PATH = ROOT / "corpus" / "source_registry.v3.approval.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_tranche_config_is_frozen_to_four_approved_sources() -> None:
    report = validate_config(config(), registry_path=REGISTRY_PATH)
    assert report["result"] == "PASS"
    assert set(report["source_ids"]) == {
        "wikimedia_trwiki_20260701",
        "wikimedia_enwiki_20260701",
        "python_docs_tr_3_14_6",
        "python_docs_en_3_14_6",
    }
    assert report["expected_raw_download_bytes"] == 1_496_788_787


def test_tranche_config_rejects_unapproved_or_undated_sources() -> None:
    payload = config()
    payload["sources"][0]["approval_status"] = "conditional"
    with pytest.raises(ValueError, match="not approved"):
        validate_config(payload, registry_path=REGISTRY_PATH)

    payload = config()
    payload["sources"][0]["snapshot"] = "latest"
    with pytest.raises(ValueError, match="exact snapshot"):
        validate_config(payload, registry_path=REGISTRY_PATH)


def test_tranche_config_enforces_raw_download_cap_and_checksum_policy() -> None:
    payload = config()
    payload["maximum_raw_download_bytes"] = payload["expected_raw_download_bytes"] - 1
    with pytest.raises(ValueError, match="hard cap"):
        validate_config(payload, registry_path=REGISTRY_PATH)

    payload = config()
    del payload["sources"][0]["files"][0]["md5"]
    del payload["sources"][0]["files"][0]["sha1"]
    with pytest.raises(ValueError, match="checksum"):
        validate_config(payload, registry_path=REGISTRY_PATH)


def test_checksum_manifest_and_file_verification(tmp_path: Path) -> None:
    payload = b"verified payload"
    path = tmp_path / "sample.bin"
    path.write_bytes(payload)
    item = {
        "filename": path.name,
        "expected_bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
        "sha1": hashlib.sha1(payload).hexdigest(),
    }
    result = _verify_file(path, item)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "sums.txt"
    manifest.write_text(f"{item['sha1']}  *{path.name}\n", encoding="utf-8")
    assert parse_checksum_manifest(manifest)[path.name] == item["sha1"]
    item["sha1"] = "0" * 40
    with pytest.raises(ValueError, match="sha1 mismatch"):
        _verify_file(path, item)


def test_stable_document_id_is_content_and_identity_bound() -> None:
    digest = hashlib.sha256(b"text").hexdigest()
    first = stable_document_id("source", "2026-07-01", "42", digest)
    assert first == stable_document_id("source", "2026-07-01", "42", digest)
    assert first != stable_document_id("source", "2026-07-01", "43", digest)


def test_quality_filter_is_strict_and_normalizes_nfc() -> None:
    policy = config()["quality_policy"]
    text = ("Bu bir Tu\u0308rkce belge ve deneme metnidir. " * 8).strip()
    result = quality_filter(text, "tr", policy)
    assert result.reason is None
    assert result.text is not None
    assert "\u00fc" in result.text
    assert quality_filter(text + "\ufffd", "tr", policy).reason == "replacement_character"
    assert quality_filter(text + "\ud800", "tr", policy).reason == "malformed_surrogate"
    assert quality_filter(("The documented value is returned by the function. " * 8) + " FranÃ§ais", "en", policy).reason == "mojibake"
    assert quality_filter(("The function is documented for the user. " * 5) + "x" * 50, "en", policy).reason == "repeated_character"
    reserved_ip = "The reserved IP address 192.0.2.1 is used in documentation and the example is not personal. " * 3
    assert quality_filter(reserved_ip, "en", policy).reason is None
    real_email = "Contact the private account private.person@realmail.test for the documented system. " * 3
    assert quality_filter(real_email, "en", policy).reason == "material_pii_email"


def test_language_validation_distinguishes_turkish_and_english() -> None:
    tr_label, tr_confidence = detect_language_confidence("Bu belge bir işlev için ve kullanıcı ile birlikte hazırlanmıştır.")
    en_label, en_confidence = detect_language_confidence("The function is documented for the user and the system.")
    assert tr_label == "tr" and tr_confidence >= 0.55
    assert en_label == "en" and en_confidence >= 0.55


def test_wiki_markup_cleaning_and_streaming_xml_extraction(tmp_path: Path) -> None:
    raw = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>Deneme</title><ns>0</ns><id>7</id><revision><id>8</id><text>{{Bilgi}} Bu bir [[Python|Python dili]] maddesidir. &lt;ref&gt;kaynak&lt;/ref&gt;</text></revision></page>
    <page><title>Redirect</title><ns>0</ns><id>9</id><redirect title="Deneme"/><revision><text>#REDIRECT</text></revision></page>
    </mediawiki>""".encode("utf-8")
    archive = tmp_path / "wiki.xml.bz2"
    archive.write_bytes(bz2.compress(raw))
    source = {
        "source_id": "wikimedia_trwiki_20260701",
        "snapshot": "2026-07-01",
        "language": "tr",
        "category": "general_prose",
    }
    documents = list(iter_wikipedia_documents(archive, source))
    assert len(documents) == 1
    assert documents[0].native_id == "7"
    assert "Bilgi" not in documents[0].text
    assert "Python dili" in documents[0].text
    assert "kaynak" not in documents[0].text
    assert "{{" not in clean_wiki_markup("{{nested|{{value}}}} Metin")


def test_python_archive_extraction_is_sectioned_and_strict_utf8(tmp_path: Path) -> None:
    source = {
        "source_id": "python_docs_en_3_14_6",
        "snapshot": "3.14.6-2026-07-09",
        "language": "en",
        "category": "technical_documentation",
    }
    archive = tmp_path / "docs.tar.bz2"
    with tarfile.open(archive, "w:bz2") as handle:
        data = b"Title\n=====\n\nThe function returns the documented value.\n\nSecond\n------\n\nThe class is available to the user."
        member = tarfile.TarInfo("python-docs/library/example.txt")
        member.size = len(data)
        handle.addfile(member, BytesIO(data))
    documents = list(iter_python_documents(archive, source))
    assert len(documents) == 1
    assert documents[0].native_id.endswith("#1")
    assert "=====" not in documents[0].text

    invalid = tmp_path / "invalid.tar.bz2"
    with tarfile.open(invalid, "w:bz2") as handle:
        data = b"\xff\xfe"
        member = tarfile.TarInfo("python-docs/library/bad.txt")
        member.size = len(data)
        handle.addfile(member, BytesIO(data))
    with pytest.raises(UnicodeDecodeError):
        list(iter_python_documents(invalid, source))


def test_exact_and_near_duplicate_primitives_are_deterministic() -> None:
    left = "The deterministic function returns a documented value for every valid input." * 4
    right = " ".join(left.upper().split())
    assert normalized_hash_text(left) == normalized_hash_text(right)
    representative = "source:representative"
    value = simhash64(left)
    values = {representative: value}
    buckets = {band: [representative] for band in simhash_bands(value)}
    assert find_near_duplicate(value, buckets, values, 3) == (representative, 0)


def test_phase1b_cross_deduplication_loads_only_hashes(tmp_path: Path) -> None:
    for filename in ("train.txt", "validation.txt", "eval.txt"):
        (tmp_path / filename).write_text(
            "A historical paragraph is long enough to be retained for deterministic cross deduplication.\n\n",
            encoding="utf-8",
        )
    payload = config()
    payload["historical_phase1b"] = {
        "processed_directory": str(tmp_path),
        "read_only": True,
        "text_files": ["train.txt", "validation.txt", "eval.txt"],
    }
    assert len(load_phase1b_hashes(payload)) == 1
    payload["historical_phase1b"]["read_only"] = False
    with pytest.raises(ValueError, match="read-only"):
        load_phase1b_hashes(payload)


def test_evaluation_contamination_material_is_normalized(tmp_path: Path) -> None:
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text(
        json.dumps({"prompt": "The exact evaluation prompt must never enter the training corpus."}) + "\n",
        encoding="utf-8",
    )
    payload = config()
    payload["contamination_policy"]["prompt_manifests"] = [str(prompt)]
    exact, substrings = load_contamination_material(payload)
    normalized = normalized_hash_text("The exact evaluation prompt must never enter the training corpus.")
    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() in exact
    assert normalized in substrings


def test_quota_gate_fails_closed_with_exact_shortfall() -> None:
    payload = config()
    inventory = {
        "source_statistics": {
            source["source_id"]: {"accepted_tokens": source["target_tokens"]}
            for source in payload["sources"]
        }
    }
    inventory["source_statistics"]["python_docs_en_3_14_6"]["accepted_tokens"] = 1_000_000
    result = quota_status(payload, inventory)
    assert result["result"] == "FAIL"
    assert result["sources"]["python_docs_en_3_14_6"]["shortfall_to_minimum"] == 8_800_000


def test_split_assignment_is_stable_and_approximately_98_1_1() -> None:
    policy = config()["split_policy"]
    first = deterministic_split("doc", "cluster", policy)
    assert first == deterministic_split("other-doc", "cluster", policy)
    counts = {"train": 0, "validation": 0, "eval": 0}
    for index in range(20_000):
        counts[deterministic_split(f"doc-{index}", f"cluster-{index}", policy)] += 1
    assert 19_300 <= counts["train"] <= 19_850
    assert 100 <= counts["validation"] <= 300
    assert 100 <= counts["eval"] <= 300


class FakeTokenizer:
    vocab_size = 24000
    eos_token_id = 3

    @staticmethod
    def encode_document(text: str) -> list[int]:
        return [10 + index for index, _ in enumerate(text.split())] + [3]


def test_allocation_attribution_and_uint16_eos_shards(tmp_path: Path) -> None:
    payload = config()
    inventory_root = tmp_path / "inventory"
    dedup = inventory_root / "deduplicated"
    dedup.mkdir(parents=True)
    expected_total = 0
    for source in payload["sources"]:
        source["target_tokens"] = 5
        source["target_tolerance_percent"] = 20
        source["tranche_source_cap_tokens"] = 5
        record = {
            "id": f"{source['source_id']}:doc",
            "source_id": source["source_id"],
            "snapshot": source["snapshot"],
            "source_native_id": "1",
            "title": "Title",
            "language": source["language"],
            "category": source["category"],
            "text": "one two three four",
            "source_url": "https://example.invalid/1",
            "license": source["license"],
            "attribution": source["attribution_requirements"],
            "redistribution_notes": source["redistribution_notes"],
            "extraction_version": "test",
            "raw_content_sha256": "a" * 64,
            "normalized_content_sha256": "b" * 64,
            "token_count": 5,
            "duplicate_cluster_id": f"{source['source_id']}:doc",
        }
        (dedup / f"{source['source_id']}.jsonl").write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        expected_total += 5
    final_text = tmp_path / "final_text"
    allocation = allocate_documents(payload, inventory_root, final_text)
    assert allocation["total_tokens"] == expected_total
    attributions = [json.loads(line) for line in (final_text / "attribution_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(record["license"] and record["attribution"] for record in attributions)
    assert all("text" not in record for record in attributions)

    payload["tokenization_policy"]["shard_token_cap"] = 10
    output = tmp_path / "tokenized"
    manifest = tokenize_allocation(payload, final_text, output, FakeTokenizer())
    assert manifest["statistics"]["total_tokens"] == expected_total
    assert manifest["statistics"]["token_range_violations"] == 0
    boundaries = [json.loads(line) for line in (output / "document_boundaries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(record["eos_token_id"] == 3 for record in boundaries)
    for shard in manifest["shards"]:
        assert (output / shard["filename"]).stat().st_size == shard["tokens"] * 2


def test_uint16_writer_rejects_out_of_range_tokens() -> None:
    assert ShardWriter.encode_uint16_le([0, 3, 23999]) == b"\x00\x00\x03\x00\xbf]"
    with pytest.raises(ValueError, match="uint16"):
        ShardWriter.encode_uint16_le([65536])


def test_frozen_tokenizer_provenance_is_unchanged() -> None:
    manifest = verify_frozen_tokenizer()
    assert manifest["tokenizer_name"] == "darkmind_v2_sp_bpe24k_v1"
    assert manifest["vocab_size"] == 24000


def test_source_record_has_complete_attribution_and_rejection_reasons_are_named() -> None:
    source = config()["sources"][0]
    document = ExtractedDocument(
        source_id=source["source_id"],
        snapshot=source["snapshot"],
        native_id="1",
        title="Title",
        language="tr",
        category="general_prose",
        text="Bu bir belge ve deneme metnidir.",
        raw_content_sha256="a" * 64,
        source_url="https://tr.wikipedia.org/?curid=1",
    )
    record = _source_record(document, source, document.text)
    assert record["license"] and record["attribution"] and record["source_url"]
    too_short = quality_filter("kisa", "tr", config()["quality_policy"])
    assert too_short.reason == "too_short"
