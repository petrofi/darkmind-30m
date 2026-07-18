from __future__ import annotations

import json
from pathlib import Path

import pytest

from darkmind_v2.evaluation.audit_public_preview import load_audit_prompts
from darkmind_v2.evaluation.phase4d_stage2 import PROMPTS_PATH, subset_prompts
from darkmind_v2.export.export_phase4d_stage2 import OUTPUT_DIR, REQUIRED_FILES
from darkmind_v2.training.phase4d_stage2 import (
    AUTHORIZATION_PATH,
    FULL_RESUME_STEPS,
    MODEL_ONLY_STEPS,
    RUN_DIR,
    SOURCE_CHECKPOINT,
    SOURCE_STEP,
    SOURCE_TOKENS,
    TARGET_STEP,
    TARGET_TOKENS,
    TOKENS_PER_STEP,
    classify_stage2,
    validate_authorization,
)


def load_authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def passing_summary() -> dict:
    return {
        "integrity_pass": True,
        "validation_improvement_percent": 10.0,
        "eval_improvement_percent": 9.0,
        "validation_rebound_percent": 1.0,
        "eval_rebound_percent": 1.5,
        "final_validation_loss": 5.5,
        "final_eval_loss": 5.5,
        "last_three_validation_sustained_worsening": False,
        "last_three_eval_sustained_worsening": False,
        "probe_regressions": {"turkish_prose": {"catastrophic_regression": False}},
    }


def test_stage2_authorization_accepts_exact_gate() -> None:
    validate_authorization(load_authorization(), requested_step=TARGET_STEP, requested_tokens=TARGET_TOKENS)


@pytest.mark.parametrize(
    ("step", "tokens"),
    ((TARGET_STEP + 1, (TARGET_STEP + 1) * TOKENS_PER_STEP), (TARGET_STEP, TARGET_TOKENS + 1)),
)
def test_stage2_authorization_rejects_exceeding_gate(step: int, tokens: int) -> None:
    with pytest.raises((PermissionError, ValueError)):
        validate_authorization(load_authorization(), requested_step=step, requested_tokens=tokens)


def test_exact_step_and_token_stop_contract() -> None:
    authorization = load_authorization()
    assert authorization["current_optimizer_step"] == SOURCE_STEP == 610
    assert authorization["current_tokens"] == SOURCE_TOKENS == 4_997_120
    assert authorization["target_optimizer_step"] == TARGET_STEP == 3_051
    assert authorization["target_tokens"] == TARGET_TOKENS == TARGET_STEP * TOKENS_PER_STEP
    assert TARGET_TOKENS - SOURCE_TOKENS == 19_996_672


def test_expected_next_sequence_indices() -> None:
    authorization = load_authorization()
    assert authorization["current_sequence_index"] == SOURCE_TOKENS // 512 == 9_760
    assert authorization["target_sequence_index"] == TARGET_TOKENS // 512 == 48_816
    assert authorization["training_segments"] == [
        {"start_step": 611, "end_step": 1831},
        {"start_step": 1832, "end_step": 3051},
    ]


def test_resume_policy_does_not_reset_state() -> None:
    authorization = load_authorization()
    assert authorization["continuation_authorized"] is True
    assert authorization["continuation_100m_authorized"] is False
    assert authorization["scheduler_reset"] is False
    assert authorization["optimizer_reset"] is False
    assert authorization["rng_reset"] is False
    assert authorization["data_order_reset"] is False


def test_checkpoint_storage_policy() -> None:
    assert MODEL_ONLY_STEPS == {1024, 1536, 2048, 2560}
    assert FULL_RESUME_STEPS == {1831, 3051}
    assert MODEL_ONLY_STEPS.isdisjoint(FULL_RESUME_STEPS)


def test_fixed_probe_reproducibility_contract() -> None:
    path = Path(r"C:\DarkMindRuntime\phase4d\manifests\fixed_probes.json")
    if not path.is_file():
        pytest.skip("runtime probe manifest is created by Phase 4D prepare")
    first = json.loads(path.read_text(encoding="utf-8"))
    second = json.loads(path.read_text(encoding="utf-8"))
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]
    for required in ("turkish_prose", "english_prose", "turkish_technical", "english_technical"):
        assert first["probes"][required]["token_sha256"]


def test_generation_subset_is_balanced_and_reproducible() -> None:
    first = subset_prompts(load_audit_prompts(PROMPTS_PATH))
    second = subset_prompts(load_audit_prompts(PROMPTS_PATH))
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len(first) == 16
    counts: dict[str, int] = {}
    for item in first:
        key = f"{item['language']}:{item['category']}"
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        "tr:ordinary_text": 4,
        "en:ordinary_text": 4,
        "tr:technical": 2,
        "en:technical": 2,
        "tr:factual_encyclopedic": 1,
        "en:factual_encyclopedic": 1,
        "code:code_structured": 2,
    }


@pytest.mark.skipif(not SOURCE_CHECKPOINT.is_dir(), reason="immutable Phase 4C checkpoint is unavailable")
def test_exact_resume_metadata_from_step_610() -> None:
    metadata = json.loads((SOURCE_CHECKPOINT / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    state = metadata["training_state"]
    assert state["step"] == SOURCE_STEP
    assert state["tokens_seen"] == SOURCE_TOKENS
    assert state["data_position"] == SOURCE_TOKENS
    assert (SOURCE_CHECKPOINT / "resume_state.pt").is_file()


def test_stage2_classification_thresholds() -> None:
    assert classify_stage2(passing_summary()) == "STRONG PASS"
    conditional = passing_summary() | {"validation_improvement_percent": 5.0, "eval_improvement_percent": 4.0}
    assert classify_stage2(conditional) == "CONDITIONAL PASS"
    failed = passing_summary() | {"validation_improvement_percent": 1.9}
    assert classify_stage2(failed) == "FAIL"


def test_source_regression_and_integrity_force_failure() -> None:
    regressed = passing_summary()
    regressed["probe_regressions"] = {"english_technical": {"catastrophic_regression": True}}
    assert classify_stage2(regressed) == "FAIL"
    assert classify_stage2(passing_summary() | {"integrity_pass": False}) == "FAIL"


def test_future_100m_resume_compatibility_without_authorization() -> None:
    authorization = load_authorization()
    assert TARGET_STEP < 12_207
    assert FULL_RESUME_STEPS == {1831, TARGET_STEP}
    assert authorization["continuation_100m_authorized"] is False
    assert authorization["maximum_authorized_step"] == TARGET_STEP
    evidence = RUN_DIR / "future_resume_validation.json"
    if evidence.is_file():
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        assert payload["result"] == "PASS"
        assert payload["step_3052_executed"] is False
        assert payload["phase_100m_started"] is False


@pytest.mark.skipif(not (OUTPUT_DIR / "offline_validation.json").is_file(), reason="local Phase 4D export is unavailable")
def test_local_export_is_safetensors_only_and_honest() -> None:
    files = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    assert REQUIRED_FILES <= files
    assert not any(path.suffix in {".bin", ".pt", ".pth"} for path in OUTPUT_DIR.iterdir())
    validation = json.loads((OUTPUT_DIR / "offline_validation.json").read_text(encoding="utf-8"))
    assert validation["result"] == "PASS"
    card = (OUTPUT_DIR / "README.md").read_text(encoding="utf-8")
    for disclosure in (
        "not instruction-tuned",
        "not a chat model",
        "not production-ready",
        "not publicly released",
        "model-weight license is unresolved",
    ):
        assert disclosure in card
