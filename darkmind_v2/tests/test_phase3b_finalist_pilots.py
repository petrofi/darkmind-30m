import hashlib
import json
from pathlib import Path

import pytest

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.modeling.phase3b_environment import validate_training_environment
from darkmind_v2.training.phase3b_decision import (
    score_finalists,
    validate_corpus_approval,
    validate_learning_rate_result,
    validate_memory_audit,
    validate_resume_continuity,
)
from darkmind_v2.training.validate_phase3b_pilot_config import (
    PILOT_TOKENS,
    load_and_validate_phase3b_pilot_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(candidate: str, **overrides) -> DarkMindV2Config:
    path = ROOT / "config" / f"model_base_candidate_{candidate.lower()}_{'100m' if candidate == 'C' else '120m'}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(overrides)
    return DarkMindV2Config(**payload)


def _generation(*, repetition: int, loops: int, unique: float, eos: float, meaningful: int):
    return {
        "greedy": {
            "generations": 200,
            "quality_warning_counts": {"repetition": repetition},
            "exact_repeated_ngram_loop_outputs": loops,
            "mean_unique_token_ratio": unique,
            "eos_completion_rate": eos,
        },
        "greedy_subsets": {"meaningful_proxy_total": meaningful},
        "sampling": {
            "generations": 500,
            "quality_warning_counts": {"repetition": repetition},
            "exact_repeated_ngram_loop_outputs": loops,
            "mean_unique_token_ratio": unique,
            "eos_completion_rate": eos,
        },
        "sampling_subsets": {"meaningful_proxy_total": meaningful},
    }


def _evidence(*, c_hard_failure: bool = False) -> dict:
    initial = _generation(repetition=150, loops=120, unique=0.3, eos=0.0, meaningful=10)
    final = _generation(repetition=50, loops=40, unique=0.6, eos=0.5, meaningful=100)
    return {
        "C": {
            "initial_validation_loss": 10.0,
            "final_validation_loss": 6.0,
            "initial_eval_loss": 10.0,
            "final_eval_loss": 6.0,
            "pilot_wall_seconds": 300.0,
            "transformer_body_params": 90,
            "initial_generation": initial,
            "final_generation": final,
            "soak_passed": True,
            "vram_headroom_percent": 82.0,
            "checkpoint_resume_reliability": 100.0,
            "implementation_backend_reliability": 100.0,
            "hard_failures": ["process crash"] if c_hard_failure else [],
        },
        "D": {
            "initial_validation_loss": 10.0,
            "final_validation_loss": 6.0,
            "initial_eval_loss": 10.0,
            "final_eval_loss": 6.0,
            "pilot_wall_seconds": 300.0,
            "transformer_body_params": 100,
            "initial_generation": initial,
            "final_generation": final,
            "soak_passed": True,
            "vram_headroom_percent": 82.0,
            "checkpoint_resume_reliability": 100.0,
            "implementation_backend_reliability": 100.0,
            "hard_failures": [],
        },
    }


@pytest.mark.parametrize("candidate,expected", [("C", 103_881_216), ("D", 118_056_960)])
def test_finalist_configs_have_exact_parameter_counts(candidate, expected) -> None:
    config = _config(candidate)
    estimate = estimate_model_size(
        vocab_size=config.vocab_size,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        tied_embeddings=config.tie_word_embeddings,
        bias=config.bias,
        mlp_hidden_size=config.effective_mlp_hidden_size,
    )
    assert estimate.total_params == expected
    assert estimate.head_dimension == 64


def test_checkpointing_guard_rejects_exact_c_profile_and_warns_without_gc() -> None:
    with pytest.raises(RuntimeError, match="unsafe Candidate C profile"):
        validate_training_environment(
            _config("C", attention_implementation="sdpa", gradient_checkpointing=True),
            micro_batch_size=2,
            precision="bf16",
            platform_name="win32",
            torch_version="2.4.1+cu121",
        )
    warning = validate_training_environment(
        _config("C", attention_implementation="sdpa", gradient_checkpointing=False),
        micro_batch_size=2,
        precision="bf16",
        platform_name="win32",
        torch_version="2.4.1+cu121",
    )
    assert warning["result"] == "WARN"
    assert warning["hard_rejected"] is False
    assert warning["warnings"]
    safe = validate_training_environment(
        _config("D", attention_implementation="sdpa", gradient_checkpointing=False),
        micro_batch_size=2,
        precision="bf16",
        platform_name="win32",
        torch_version="2.4.1+cu121",
    )
    assert safe["result"] == "PASS"


def test_pilot_contract_and_exact_equal_token_enforcement(tmp_path) -> None:
    path = ROOT / "config" / "phase3b_finalist_pilot.json"
    config = load_and_validate_phase3b_pilot_config(path)
    assert config["pilot"]["maximum_total_training_tokens"] == PILOT_TOKENS == 4_997_120
    assert PILOT_TOKENS == 610 * 8192
    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["pilot"]["maximum_total_training_tokens"] += 8192
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot token target"):
        load_and_validate_phase3b_pilot_config(broken_path)


def test_memory_learning_rate_and_resume_schemas_fail_closed() -> None:
    memory = {
        "result": "PASS",
        "results": {
            candidate: {
                "optimizer_steps": 10,
                "optimizer_state_tensors": 4,
                "reserved_headroom_percent": 80.0,
            }
            for candidate in ("C", "D")
        },
    }
    validate_memory_audit(memory)
    memory["results"]["D"]["optimizer_state_tensors"] = 0
    with pytest.raises(ValueError, match="not materialized"):
        validate_memory_audit(memory)

    calibration = {
        "candidate": "D",
        "learning_rate": 0.0003,
        "tokens": 524_288,
        "initial_validation_loss": 10.0,
        "final_validation_loss": 8.0,
        "gradient_norm_max": 4.0,
        "stable": True,
        "non_finite_events": 0,
        "result": "PASS",
    }
    validate_learning_rate_result(calibration)
    calibration["tokens"] -= 8192
    with pytest.raises(ValueError, match="token budget"):
        validate_learning_rate_result(calibration)

    resume = {
        "result": "PASS",
        "rng_continuity": True,
        "scheduler_continuity": True,
        "data_position_continuity": True,
        "no_repeated_or_skipped_sequence": True,
    }
    validate_resume_continuity(resume)
    resume["data_position_continuity"] = False
    with pytest.raises(ValueError, match="continuity"):
        validate_resume_continuity(resume)


def test_score_uses_c_tie_break_only_after_hard_gates() -> None:
    tied = score_finalists(_evidence())
    assert tied["tie_break_applied"] is True
    assert tied["selected"] == "C"
    rejected = score_finalists(_evidence(c_hard_failure=True))
    assert rejected["selected"] == "D"
    assert rejected["tie_break_applied"] is False
    assert next(row for row in rejected["rows"] if row["candidate"] == "C")["eligible"] is False


def test_frozen_base_config_hash_counts_and_immutable_constraints() -> None:
    config_path = ROOT / "config" / "model_base_v1.json"
    constraints = json.loads(
        (ROOT / "config" / "model_base_v1_constraints.json").read_text(encoding="utf-8")
    )
    config = DarkMindV2Config.from_json_file(config_path)
    assert config.n_layer == 14
    assert config.attention_implementation == "sdpa"
    assert config.gradient_checkpointing is False
    assert constraints["parameter_counts"] == {
        "total": 118_056_960,
        "transformer_body": 99_624_960,
        "vocabulary_share_percent": 15.6128,
    }
    assert constraints["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert constraints["architecture_hash"] == model_config_hash(config)
    assert constraints["immutable_after_freeze"] is True
    assert set(constraints["immutable_fields"]) >= {"n_layer", "n_head", "n_embd", "vocab_size"}


def test_token_plan_and_corpus_approval_contracts() -> None:
    plan = json.loads((ROOT / "config" / "production_training_token_plan.json").read_text(encoding="utf-8"))
    assert plan["model_parameters"] == 118_056_960
    assert [stage["tokens"] for stage in plan["total_seen_token_stages"]] == [
        5_000_000,
        25_000_000,
        100_000_000,
        250_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
    ]
    assert all(stage["requires_user_approval"] for stage in plan["total_seen_token_stages"])
    assert plan["total_seen_token_stages"][-1]["optional"] is True

    approval = json.loads(
        (ROOT / "corpus" / "source_registry.v3.approval.json").read_text(encoding="utf-8")
    )
    validate_corpus_approval(approval)
    assert approval["download_authorized"] is False
    assert approval["summary"]["status_counts"] == {
        "approved": 8,
        "conditional": 5,
        "deferred": 4,
        "rejected": 3,
    }
    assert sum(item["tokens"] for item in approval["first_100m_tranche"]) == 100_000_000


def test_tiny_model_contract_remains_unchanged() -> None:
    tiny = DarkMindV2Config.from_json_file(ROOT / "config" / "model_tiny_smoke.json")
    assert tiny.n_layer == 4
    assert tiny.n_embd == 256
    assert model_config_hash(tiny) == "61ebdadb15d4fde09db9842413450f512a12fc7f7e6fcaead7dc1347478f4414"
