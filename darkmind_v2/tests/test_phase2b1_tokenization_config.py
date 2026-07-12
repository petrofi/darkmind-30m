import hashlib
import json
import math
from pathlib import Path

from darkmind_v2.data_pipeline.tokenize_phase1b_corpus import tokenize_phase1b_corpus
from darkmind_v2.data_pipeline.validate_full_tokenized_corpus import validate_full_tokenized_corpus
from darkmind_v2.training.token_shard_dataset import TokenShardDataset
from darkmind_v2.training.train_tiny_stage1 import checkpoint_name, learning_rate_factor
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_stage1_config import load_and_validate_stage1_config


def write_processed_fixture(path: Path) -> None:
    path.mkdir()
    splits = {
        "train": [("train-1", "tr", "Kisa bir Turkce belge."), ("train-2", "en", "A short English document.")],
        "validation": [("validation-1", "tr", "Dogrulama belgesi.")],
        "eval": [("eval-1", "en", "Evaluation document.")],
    }
    attribution = []
    for split, records in splits.items():
        text = "\n\n".join(item[2] for item in records) + "\n"
        filename = {"train": "tokenizer_train.txt", "validation": "tokenizer_validation.txt", "eval": "tokenizer_eval.txt"}[split]
        (path / filename).write_text(text, encoding="utf-8", newline="\n")
        for document_id, language, document in records:
            attribution.append(
                {
                    "id": document_id,
                    "language": language,
                    "selected_split": split,
                    "selected_character_count": len(document),
                }
            )
    (path / "attribution_manifest.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in attribution),
        encoding="utf-8",
        newline="\n",
    )
    split_manifest = {
        "splits": {
            split: {"documents": len(records)}
            for split, records in splits.items()
        }
    }
    (path / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    (path / "corpus_manifest.json").write_text(
        json.dumps({"deterministic_content_sha256": "fixture-corpus-hash"}),
        encoding="utf-8",
    )


def test_full_tokenization_stream_is_deterministic_and_valid(tmp_path) -> None:
    processed = tmp_path / "processed"
    write_processed_fixture(processed)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = tokenize_phase1b_corpus(processed, first, shard_token_cap=12)
    second_manifest = tokenize_phase1b_corpus(processed, second, shard_token_cap=12)
    assert first_manifest["deterministic_content_hash"] == second_manifest["deterministic_content_hash"]
    assert first_manifest["statistics"]["accepted_documents"] == 4
    assert first_manifest["statistics"]["rejected_documents"] == 0
    expected = first_manifest["statistics"]["total_tokens"]
    first_report = validate_full_tokenized_corpus(
        first,
        processed,
        plausibility_estimate_tokens=expected,
    )
    second_report = validate_full_tokenized_corpus(
        first,
        processed,
        plausibility_estimate_tokens=expected,
    )
    assert first_report["result"] == "PASS"
    assert first_report["validation_content_hash"] == second_report["validation_content_hash"]
    dataset = TokenShardDataset(first, "train")
    values = dataset.read(0, min(8, dataset.total_tokens))
    assert len(values) == min(8, dataset.total_tokens)
    assert values.min() >= 0 and values.max() < 24000


def test_stage1_config_has_exact_bounded_training_math() -> None:
    config = load_and_validate_stage1_config(Path("darkmind_v2/config/train_tiny_stage1.json"))
    assert config["maximum_total_training_tokens"] == 1_048_576
    assert config["segment_a_target_tokens"] // 4096 == 128
    assert config["segment_b_target_tokens"] // 4096 == 256
    assert all(
        item["micro_batch_size"] * item["gradient_accumulation_steps"] * 256 == 4096
        for item in config["profiles"]
    )
    assert learning_rate_factor(0, config) == 0.05
    assert learning_rate_factor(19, config) == 1.0
    assert math.isclose(learning_rate_factor(256, config), 0.1)


def test_stage1_state_preserves_data_position_and_checkpoint_identity() -> None:
    state = TrainingState(step=128, tokens_seen=524_288, data_position=524_288)
    restored = TrainingState.from_dict(state.to_dict())
    assert restored.step + 1 == 129
    assert restored.tokens_seen == restored.data_position == 524_288
    assert checkpoint_name(restored) == "step_000128_tokens_000524288"


def test_processed_fixture_hashes_are_stable(tmp_path) -> None:
    processed = tmp_path / "processed"
    write_processed_fixture(processed)
    first = hashlib.sha256((processed / "tokenizer_train.txt").read_bytes()).hexdigest()
    second = hashlib.sha256((processed / "tokenizer_train.txt").read_bytes()).hexdigest()
    assert first == second
