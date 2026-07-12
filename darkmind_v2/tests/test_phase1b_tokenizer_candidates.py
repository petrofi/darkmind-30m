import json
from pathlib import Path

from darkmind_v2.tokenizer.build_tokenizer_manifest import build_manifest
from darkmind_v2.tokenizer.compare_tokenizer_candidates import (
    REQUIRED_AUDIT_KEYS,
    recommend_candidate,
    score_candidates,
    validate_audit_schema,
)
from darkmind_v2.tokenizer.estimate_vocab_parameter_cost import build_cost_table, estimate_vocab_cost
from darkmind_v2.tokenizer.train_tokenizer_candidates import (
    CANDIDATE_DIRECTORIES,
    build_sentencepiece_training_options,
)


ROOT = Path("darkmind_v2")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fake_audit(candidate_id: str, vocab_size: int, efficiency: float, *, gates: str = "PASS") -> dict:
    report = {
        "candidate_id": candidate_id,
        "algorithm": "sentencepiece_bpe",
        "vocabulary_size": vocab_size,
        "tokenizer_file_hashes": {"tokenizer.model": "abc", "tokenizer.vocab": "def"},
        "special_token_ids": {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3},
        "byte_fallback": {"enabled": True, "byte_piece_count": 256},
        "unknown_token_count": 0,
        "round_trip_failure_count": 0,
        "mojibake_vocabulary_tokens": [],
        "replacement_character_tokens": [],
        "malformed_tokens": [],
        "metrics": {
            "turkish_tokens_per_character": efficiency,
            "english_tokens_per_character": efficiency,
            "technical_code_tokens_per_character": efficiency,
            "tokens_per_word": efficiency * 5,
            "turkish_suffix_fragmentation": efficiency * 4,
            "english_word_fragmentation": efficiency * 3,
            "code_operator_fragmentation": efficiency * 2,
        },
        "sequence_length_percentiles": {"p50": 10, "p90": 20, "p95": int(efficiency * 100), "p99": int(efficiency * 120), "maximum": 100},
        "vocabulary_script_distribution": {"english_ascii": 100},
        "mixed_script_risk": {"token_count": 0, "ratio": 0.0},
        "parameter_costs": build_cost_table([vocab_size], [384, 512]),
        "hard_gate_result": gates,
        "hard_gate_failures": [] if gates == "PASS" else ["fixture_failure"],
    }
    assert REQUIRED_AUDIT_KEYS <= set(report)
    return report


def test_phase1b_candidate_config_and_output_mapping() -> None:
    config = load_json(ROOT / "config" / "tokenizer_candidates.json")
    assert [item["id"] for item in config["candidates"]] == ["A", "B", "C", "D"]
    assert set(CANDIDATE_DIRECTORIES) == {"A", "B", "C", "D"}
    assert len(set(CANDIDATE_DIRECTORIES.values())) == 4


def test_sentencepiece_training_options_are_deterministic_and_share_special_tokens(tmp_path: Path) -> None:
    config = load_json(ROOT / "config" / "tokenizer_candidates.json")
    candidate = config["candidates"][0]
    train = tmp_path / "train.txt"
    train.write_text("Türkçe and English", encoding="utf-8")
    first = build_sentencepiece_training_options(candidate, config["shared_requirements"], train, tmp_path / "out")
    second = build_sentencepiece_training_options(candidate, config["shared_requirements"], train, tmp_path / "out")
    assert first == second
    assert first["model_type"] == "bpe"
    assert first["vocab_size"] == 12000
    assert first["byte_fallback"] is True
    assert first["num_threads"] == 1
    assert first["shuffle_input_sentence"] is False
    assert first["pad_id"] == 0 and first["unk_id"] == 1 and first["bos_id"] == 2 and first["eos_id"] == 3
    assert first["user_defined_symbols"] == ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def test_tokenizer_manifest_deterministic_content_hash_ignores_timestamp(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text(json.dumps({"<unk>": 0, "a": 1}), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    kwargs = {
        "training_corpus_manifest_hash": "corpus",
        "tokenizer_version": "candidate-test",
        "normalization_rules": {"unicode": "NFC"},
        "special_tokens": ["<unk>"],
        "byte_fallback": True,
        "unknown_token_behavior": "explicit",
        "creation_command": "fixture",
    }
    first = build_manifest(tmp_path, timestamp="2026-01-01T00:00:00+00:00", **kwargs)
    second = build_manifest(tmp_path, timestamp="2026-01-02T00:00:00+00:00", **kwargs)
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]
    assert first["tokenizer_file_hashes"] == second["tokenizer_file_hashes"]


def test_parameter_cost_matrix_covers_tied_and_untied_384_and_512() -> None:
    rows = build_cost_table([16000], [384, 512])
    assert {(row["embedding_dim"], row["tied_output"]) for row in rows} == {
        (384, True), (384, False), (512, True), (512, False)
    }
    assert estimate_vocab_cost(24000, 384, tied_output=True).combined_parameters == 9_216_000


def test_audit_report_schema_validation() -> None:
    report = fake_audit("A", 12000, 0.4)
    assert validate_audit_schema(report) == []
    del report["metrics"]["turkish_suffix_fragmentation"]
    assert "missing audit metric: turkish_suffix_fragmentation" in validate_audit_schema(report)


def test_comparison_scoring_prefers_better_efficiency_when_gates_pass() -> None:
    scored = score_candidates([fake_audit("A", 12000, 0.5), fake_audit("B", 16000, 0.3)])
    assert scored[0]["candidate_id"] == "B"
    assert scored[0]["weighted_score"] > scored[1]["weighted_score"]


def test_hard_gate_failure_overrides_weighted_score() -> None:
    scored = score_candidates([
        fake_audit("A", 12000, 0.2, gates="FAIL"),
        fake_audit("B", 16000, 0.5, gates="PASS"),
    ])
    failed = next(item for item in scored if item["candidate_id"] == "A")
    assert failed["eligible"] is False
    assert failed["weighted_score"] == 0.0
    assert scored[0]["candidate_id"] == "B"


def test_smaller_vocabulary_wins_within_two_points() -> None:
    recommendation = recommend_candidate(
        [
            {"candidate_id": "B", "vocabulary_size": 16000, "eligible": True, "weighted_score": 90.0},
            {"candidate_id": "A", "vocabulary_size": 12000, "eligible": True, "weighted_score": 88.5},
        ]
    )
    assert recommendation["candidate_id"] == "A"
    assert recommendation["strength"] == "marginal"
