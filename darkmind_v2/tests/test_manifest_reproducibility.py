import json
from pathlib import Path

from darkmind_v2.corpus.build_corpus_manifest import build_manifest
from darkmind_v2.eval.validate_fixed_prompts import validate_prompts
from darkmind_v2.tokenizer.build_tokenizer_manifest import build_manifest as build_tokenizer_manifest


def write_sample(path: Path) -> None:
    rows = [
        {
            "id": "doc-1",
            "text": "Türkiye'nin başkenti Ankara'dır.",
            "source": "fixture",
            "license": "MIT",
        },
        {
            "id": "doc-2",
            "text": "The capital of Turkey is Ankara.",
            "source": "fixture",
            "license": "MIT",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_corpus_manifest_content_hash_is_reproducible(tmp_path: Path) -> None:
    corpus = tmp_path / "sample.jsonl"
    write_sample(corpus)
    first = build_manifest([corpus], timestamp="2026-07-03T00:00:00+00:00")
    second = build_manifest([corpus], timestamp="2026-07-04T00:00:00+00:00")
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]
    assert first["train_validation_test_hashes"] == second["train_validation_test_hashes"]


def test_tokenizer_manifest_file_hashes_are_reproducible(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tok"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.json").write_text(json.dumps({"<unk>": 0, "a": 1}, sort_keys=True), encoding="utf-8")
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    first = build_tokenizer_manifest(
        tokenizer_dir,
        training_corpus_manifest_hash="abc",
        tokenizer_version="test-v1",
        normalization_rules={"unicode": "NFC"},
        special_tokens=["<unk>"],
        byte_fallback=True,
        unknown_token_behavior="explicit <unk> token",
        creation_command="test",
        timestamp="2026-07-03T00:00:00+00:00",
    )
    second = build_tokenizer_manifest(
        tokenizer_dir,
        training_corpus_manifest_hash="abc",
        tokenizer_version="test-v1",
        normalization_rules={"unicode": "NFC"},
        special_tokens=["<unk>"],
        byte_fallback=True,
        unknown_token_behavior="explicit <unk> token",
        creation_command="test",
        timestamp="2026-07-04T00:00:00+00:00",
    )
    assert first["tokenizer_file_hashes"] == second["tokenizer_file_hashes"]
    assert first["special_token_ids"] == second["special_token_ids"]
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]


def test_fixed_prompt_schema_validation() -> None:
    prompts = Path("darkmind_v2/eval/fixed_base_prompts.jsonl")
    report, failures = validate_prompts(prompts)
    assert failures == []
    assert report["prompt_count"] >= 40
    assert report["language_distribution"] == {"en": 24, "tr": 24}

