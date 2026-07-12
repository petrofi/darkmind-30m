import json
from pathlib import Path

from darkmind_v2.data_pipeline.tokenize_corpus import tokenize_corpus
from darkmind_v2.data_pipeline.validate_tokenized_shards import read_uint16_le, validate_tokenized_shards


FIXTURE = Path("darkmind_v2/tests/fixtures/phase2a_tiny_corpus.jsonl")


def test_tokenized_shards_are_deterministic_uint16_and_valid(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = tokenize_corpus(FIXTURE, first, max_tokens_per_shard=80)
    second_manifest = tokenize_corpus(FIXTURE, second, max_tokens_per_shard=80)
    assert first_manifest["deterministic_content_hash"] == second_manifest["deterministic_content_hash"]
    assert first_manifest["dtype"] == "uint16-le"
    assert validate_tokenized_shards(first)["result"] == "PASS"
    for shard in first_manifest["shards"]:
        assert (first / shard["filename"]).read_bytes() == (second / shard["filename"]).read_bytes()
        tokens = read_uint16_le(first / shard["filename"])
        assert tokens and max(tokens) < 24000


def test_every_document_has_eos_and_no_split_contamination(tmp_path) -> None:
    output = tmp_path / "tokens"
    manifest = tokenize_corpus(FIXTURE, output)
    shards = {
        shard["filename"]: read_uint16_le(output / shard["filename"])
        for shard in manifest["shards"]
    }
    split_ids = {}
    for document in manifest["documents"]:
        split_ids.setdefault(document["split"], set()).add(document["id"])
        tokens = shards[document["shard"]]
        assert tokens[document["end_offset"] - 1] == 3
    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["eval"])


def test_cross_split_duplicate_text_is_rejected(tmp_path) -> None:
    source = tmp_path / "contaminated.jsonl"
    records = [
        {"id": "train-1", "split": "train", "language": "en", "text": "same document"},
        {"id": "eval-1", "split": "eval", "language": "en", "text": "same document"},
    ]
    source.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    manifest = tokenize_corpus(source, output)
    rejected = (output / "rejected_records.jsonl").read_text(encoding="utf-8")
    assert manifest["statistics"]["accepted_documents"] == 1
    assert "cross_split_duplicate_text" in rejected
