import copy
import json
from collections import Counter
from pathlib import Path

from darkmind_v2.corpus.validate_source_registry import validate_registry_payload
from darkmind_v2.tokenizer.estimate_vocab_parameter_cost import estimate_vocab_cost


ROOT = Path("darkmind_v2")
CONFIG = ROOT / "config"
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_samples() -> list[dict]:
    samples = []
    for line in (TOKENIZER / "tokenizer_eval_samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def registry_payload() -> dict:
    return load_json(CORPUS / "source_registry.example.json")


def first_source(payload: dict) -> dict:
    return payload["sources"][0]


def test_source_registry_schema_contains_required_policy_fields() -> None:
    schema = load_json(CORPUS / "source_registry.schema.json")
    source_schema = schema["properties"]["sources"]["items"]
    required = set(source_schema["required"])
    for field in [
        "source_id",
        "source_name",
        "official_homepage",
        "official_download_url",
        "documented_retrieval_method",
        "language",
        "license_id",
        "official_license_url",
        "attribution_requirements",
        "redistribution_requirements",
        "commercial_use_status",
        "modification_status",
        "jurisdiction_warning",
        "source_version",
        "snapshot_date",
        "estimated_download_size",
        "estimated_download_size_bytes",
        "intended_sample_size_characters",
        "checksum_available",
        "retrieval_method",
        "max_download_bytes",
        "max_sample_characters",
        "approved",
        "approval_reason",
        "risk_level",
        "notes",
    ]:
        assert field in required or field in source_schema["properties"]
    assert source_schema["properties"]["language"]["enum"] == ["tr", "en", "mixed_tr_en"]
    assert source_schema["properties"]["risk_level"]["enum"] == ["low", "medium", "high", "rejected"]


def test_source_registry_example_passes_validator() -> None:
    report, failures = validate_registry_payload(registry_payload())
    assert failures == []
    assert report["source_count"] == 4
    assert report["approved_count"] == 4
    assert report["language_distribution"] == {"en": 1, "mixed_tr_en": 1, "tr": 2}


def test_missing_license_rejected() -> None:
    payload = copy.deepcopy(registry_payload())
    first_source(payload)["license_id"] = "unknown"
    first_source(payload)["official_license_url"] = ""
    _, failures = validate_registry_payload(payload)
    assert any("license_id is unknown or ambiguous" in failure for failure in failures)
    assert any("official_license_url" in failure for failure in failures)


def test_unapproved_source_rejected() -> None:
    payload = copy.deepcopy(registry_payload())
    first_source(payload)["approved"] = False
    _, failures = validate_registry_payload(payload)
    assert any("approved must be explicitly true" in failure for failure in failures)


def test_invalid_language_rejected() -> None:
    payload = copy.deepcopy(registry_payload())
    first_source(payload)["language"] = "de"
    _, failures = validate_registry_payload(payload)
    assert any("language must be one of" in failure for failure in failures)


def test_missing_source_version_rejected() -> None:
    payload = copy.deepcopy(registry_payload())
    first_source(payload)["source_version"] = ""
    first_source(payload)["snapshot_date"] = ""
    _, failures = validate_registry_payload(payload)
    assert any("source_version or snapshot_date is required" in failure for failure in failures)


def test_download_cap_validation_rejects_oversized_sources() -> None:
    payload = copy.deepcopy(registry_payload())
    first_source(payload)["max_download_bytes"] = payload["hard_download_cap_bytes"] + 1
    _, failures = validate_registry_payload(payload)
    assert any("max_download_bytes exceeds registry hard_download_cap_bytes" in failure for failure in failures)


def test_common_crawl_and_social_sources_are_not_auto_approved() -> None:
    payload = copy.deepcopy(registry_payload())
    source = first_source(payload)
    source["source_id"] = "fineweb_common_crawl_sample"
    source["source_name"] = "FineWeb Common Crawl sample"
    _, failures = validate_registry_payload(payload)
    assert any("Common Crawl-derived datasets" in failure for failure in failures)

    payload = copy.deepcopy(registry_payload())
    source = first_source(payload)
    source["source_id"] = "reddit_comments"
    source["source_name"] = "Reddit comments"
    _, failures = validate_registry_payload(payload)
    assert any("social, private, leaked, or personal datasets" in failure for failure in failures)


def test_tokenizer_candidate_config_validation() -> None:
    config = load_json(CONFIG / "tokenizer_candidates.json")
    candidates = config["candidates"]
    assert [candidate["id"] for candidate in candidates] == ["A", "B", "C", "D"]
    assert [(candidate["algorithm"], candidate["vocab_size"]) for candidate in candidates] == [
        ("sentencepiece_bpe", 12000),
        ("sentencepiece_bpe", 16000),
        ("sentencepiece_unigram", 16000),
        ("sentencepiece_bpe", 24000),
    ]
    assert config["shared_requirements"]["byte_fallback_or_equivalent"] is True
    assert config["shared_requirements"]["minimum_eval_sample_count"] == 200


def test_special_token_consistency() -> None:
    config = load_json(CONFIG / "tokenizer_candidates.json")
    special_tokens = config["shared_requirements"]["special_tokens"]
    special_token_ids = config["shared_requirements"]["special_token_ids"]
    assert special_tokens == [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ]
    assert set(special_tokens) == set(special_token_ids)
    assert [special_token_ids[token] for token in special_tokens] == list(range(len(special_tokens)))


def test_parameter_cost_calculations() -> None:
    tied = estimate_vocab_cost(24000, 512, tied_output=True)
    assert tied.embedding_parameters == 12_288_000
    assert tied.output_head_parameters == 0
    assert tied.combined_parameters == 12_288_000
    assert tied.fp16_storage_bytes == 24_576_000
    assert tied.percent_of_45m == 27.3067

    untied = estimate_vocab_cost(16000, 384, tied_output=False)
    assert untied.embedding_parameters == 6_144_000
    assert untied.output_head_parameters == 6_144_000
    assert untied.combined_parameters == 12_288_000
    assert untied.fp32_storage_bytes == 49_152_000


def test_acceptance_gate_schema_and_weights() -> None:
    gates = load_json(CONFIG / "tokenizer_acceptance_gates.json")
    hard_fail_ids = {item["id"] for item in gates["hard_fail_conditions"]}
    assert "mojibake_input" in hard_fail_ids
    assert "replacement_character_input" in hard_fail_ids
    assert "unapproved_source" in hard_fail_ids
    assert "special_token_mismatch" in hard_fail_ids
    assert sum(gates["scoring_weights"].values()) == 100
    assert gates["minimum_metric_targets"]["valid_eval_roundtrip_pass_ratio"] == 1.0


def test_tokenizer_pilot_corpus_config() -> None:
    plan = load_json(CONFIG / "tokenizer_pilot_corpus.json")
    assert plan["target_normalized_characters"] == 50_000_000
    assert plan["language_mix"]["tr"]["target_ratio"] == 0.6
    assert plan["language_mix"]["en"]["target_ratio"] == 0.4
    assert plan["source_caps"]["max_single_source_ratio"] == 0.4
    assert plan["hard_download_cap_bytes"] == 1_073_741_824
    assert round(sum(item["target_ratio"] for item in plan["content_mix"].values()), 6) == 1.0
    assert "No tokenizer training in Phase 1A." in plan["non_goals"]


def test_tokenizer_eval_sample_schema_and_counts() -> None:
    samples = load_eval_samples()
    assert len(samples) == 200

    ids = [sample["id"] for sample in samples]
    assert len(ids) == len(set(ids))

    required_fields = {
        "id",
        "language",
        "category",
        "text",
        "expected_valid",
        "expected_script",
        "forbidden_patterns",
        "notes",
    }
    for sample in samples:
        assert required_fields <= set(sample)
        assert isinstance(sample["text"], str) and sample["text"].strip()
        assert isinstance(sample["expected_valid"], bool)
        assert isinstance(sample["forbidden_patterns"], list)

    counts = Counter(sample["category"] for sample in samples)
    assert counts == {
        "turkish": 80,
        "english": 60,
        "technical_code_adjacent": 30,
        "source_code": 20,
        "hostile_encoding": 10,
    }
    hostile = [sample for sample in samples if sample["category"] == "hostile_encoding"]
    assert all(sample["expected_valid"] is False for sample in hostile)
    assert all("never training data" in sample["notes"] for sample in hostile)


def test_deterministic_config_serialization() -> None:
    paths = [
        CORPUS / "source_registry.example.json",
        CONFIG / "tokenizer_pilot_corpus.json",
        CONFIG / "tokenizer_candidates.json",
        CONFIG / "tokenizer_acceptance_gates.json",
    ]
    for path in paths:
        payload = load_json(path)
        first = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        second = json.dumps(json.loads(first), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert first == second
