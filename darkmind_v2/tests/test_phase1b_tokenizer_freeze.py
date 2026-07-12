import hashlib
import json
from pathlib import Path

from darkmind_v2.modeling.estimate_model_size import PRESETS, estimate_model_size, estimate_presets


ROOT = Path("darkmind_v2")
FROZEN = ROOT / "tokenizer" / "frozen" / "darkmind_v2_sp_bpe24k_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_tokenizer_manifest_schema_and_policy() -> None:
    manifest = load_json(FROZEN / "tokenizer_freeze_manifest.json")
    required = {
        "tokenizer_name", "tokenizer_version", "tokenizer_type", "vocab_size",
        "sentencepiece_version", "byte_fallback", "special_tokens", "special_token_ids",
        "model_file", "vocab_file", "model_sha256", "vocab_sha256",
        "training_config_sha256", "source_candidate", "source_candidate_manifest_sha256",
        "processed_corpus_manifest_sha256", "attribution_manifest_sha256", "split_manifest_sha256",
        "tokenizer_comparison_report_sha256", "candidate_d_report_sha256", "frozen_at_policy_note",
        "immutable_after_freeze", "required_model_embedding_policy",
    }
    assert required <= set(manifest)
    assert manifest["tokenizer_name"] == "darkmind_v2_sp_bpe24k_v1"
    assert manifest["source_candidate"] == "D"
    assert manifest["vocab_size"] == 24000
    assert manifest["immutable_after_freeze"] is True
    assert manifest["required_model_embedding_policy"] == "tied_input_output_embeddings"


def test_frozen_tokenizer_file_hashes_match_manifest() -> None:
    manifest = load_json(FROZEN / "tokenizer_freeze_manifest.json")
    assert sha256_file(FROZEN / manifest["model_file"]) == manifest["model_sha256"]
    assert sha256_file(FROZEN / manifest["vocab_file"]) == manifest["vocab_sha256"]
    assert sha256_file(FROZEN / "training_config.json") == manifest["training_config_sha256"]
    assert sha256_file(FROZEN / "tokenizer_manifest.json") == manifest["source_candidate_manifest_sha256"]


def test_frozen_tokenizer_hash_record_matches_every_listed_file() -> None:
    hashes = load_json(FROZEN / "tokenizer_hashes.json")
    assert hashes["format"] == "darkmind-v2-tokenizer-freeze-hashes-v1"
    for filename, expected in hashes["frozen_files"].items():
        assert sha256_file(FROZEN / filename) == expected
    assert "self-referential" in hashes["self_hash_excluded"]


def test_frozen_special_token_ids_are_selected_ids() -> None:
    manifest = load_json(FROZEN / "tokenizer_freeze_manifest.json")
    tokens = ["<pad>", "<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    assert manifest["special_tokens"] == tokens
    assert [manifest["special_token_ids"][token] for token in tokens] == list(range(8))


def test_model_tokenizer_constraint_config_requires_tied_embeddings() -> None:
    constraints = load_json(ROOT / "config" / "model_tokenizer_constraints.json")
    assert constraints["selected_tokenizer"] == "darkmind_v2_sp_bpe24k_v1"
    assert constraints["vocab_size"] == 24000
    assert constraints["default_embedding_policy"] == "tied"
    assert constraints["untied_lm_head_allowed"] is False
    assert constraints["exact_parameter_calculator_required_before_training"] is True
    assert constraints["no_sft_before_base_quality_gates_pass"] is True


def test_model_parameter_calculator_breakdown_sums_to_total() -> None:
    estimate = estimate_model_size(**PRESETS["tiny_smoke"])
    components = (
        estimate.token_embedding_params + estimate.position_embedding_params + estimate.attention_params
        + estimate.mlp_params + estimate.layer_norm_params + estimate.lm_head_params
    )
    assert estimate.total_params == components
    assert estimate.vocab_related_params + estimate.non_vocab_params == estimate.total_params
    assert estimate.lm_head_params == 0


def test_untied_parameter_difference_and_24k_warning() -> None:
    config = PRESETS["candidate_base_45m_class"]
    tied = estimate_model_size(**config)
    untied = estimate_model_size(**{**config, "tied_embeddings": False})
    assert untied.total_params - tied.total_params == 24000 * 512 + 24000
    assert untied.vocab_related_percentage > tied.vocab_related_percentage
    reports = estimate_presets()
    assert reports["candidate_base_45m_class"]["untied_warning"]["lm_head_params"] > 0
    assert "not allowed by default" in reports["candidate_base_45m_class"]["warning"]


def test_selected_tokenizer_decision_exists_and_records_training_stop() -> None:
    decision = (FROZEN / "selected_tokenizer_decision.md").read_text(encoding="utf-8")
    assert "Selected candidate: D" in decision
    assert "Tied input/output embeddings are required by default" in decision
    assert "No model has been trained" in decision
    assert "No instruction tuning or SFT has started" in decision
    assert "Pilot500 will not be reused" in decision
