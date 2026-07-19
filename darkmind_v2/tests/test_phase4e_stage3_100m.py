from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from darkmind_v2.evaluation import phase4e_stage3 as evaluation
from darkmind_v2.evaluation import phase4e_memorization as memorization
from darkmind_v2.export import export_phase4e_stage3 as stage3_export
from darkmind_v2.training import phase4e_stage3 as stage3


ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "darkmind_v2" / "config" / "train_base_v1_stage3_100m_authorization.json"


def authorization() -> dict:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_stage3_authorization_preserves_frozen_identities() -> None:
    payload = authorization()
    assert payload["v2_config_sha256"] == stage3.EXPECTED_V2_FILE_HASH
    assert payload["architecture_hash"] == stage3.EXPECTED_ARCHITECTURE_HASH
    assert payload["model_config_sha256"] == stage3.EXPECTED_CONFIG_SHA256
    assert payload["tokenizer_hashes"] == stage3.EXPECTED_HASHES
    assert payload["corpus_hashes"]["sequence_order"] == stage3.EXPECTED_ORDER_HASH
    assert payload["source_checkpoint_model_sha256"] == stage3.EXPECTED_SOURCE_MODEL_HASH
    assert payload["source_checkpoint_resume_state_sha256"] == stage3.EXPECTED_SOURCE_RESUME_HASH


def test_exact_unique_sequence_capacity_and_full_batch_stop() -> None:
    payload = authorization()
    assert stage3.UNIQUE_SEQUENCE_CAPACITY == 191_566
    assert stage3.UNIQUE_SEQUENCE_CAPACITY // stage3.SEQUENCES_PER_STEP == stage3.FINAL_STEP == 11_972
    assert stage3.FINAL_TOKENS == 98_074_624
    assert stage3.USABLE_SEQUENCE_CAPACITY == 191_552
    assert stage3.UNIQUE_SEQUENCE_CAPACITY - stage3.USABLE_SEQUENCE_CAPACITY == 14
    assert payload["maximum_no_wrap_sequence_index"] == stage3.USABLE_SEQUENCE_CAPACITY
    assert payload["partial_effective_batch_authorized"] is False


def test_nominal_100m_horizon_is_hard_limit_not_no_wrap_target() -> None:
    payload = authorization()
    assert payload["hard_scheduler_optimizer_step_limit"] == 12_207
    assert payload["hard_scheduler_token_limit"] == 99_999_744
    assert stage3.FINAL_STEP < stage3.HARD_STEP_LIMIT
    assert stage3.FINAL_TOKENS < stage3.HARD_TOKEN_LIMIT
    assert payload["sequence_repetition_authorized"] is False


def test_50m_and_75m_targets_are_exact() -> None:
    payload = authorization()
    assert payload["gate_50m"] == {
        "optimizer_step": 6103,
        "tokens": 49_995_776,
        "sequence_index": 97_648,
        "continuation_to_gate_authorized": True,
        "continuation_beyond_gate_conditional": True,
    }
    assert payload["gate_75m"] == {
        "optimizer_step": 9155,
        "tokens": 74_997_760,
        "sequence_index": 146_480,
        "continuation_beyond_gate_conditional": True,
    }


def test_gate_enforcement_prevents_unauthorized_crossing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage3, "_gate_path", lambda step: tmp_path / f"{step}.json")
    payload = authorization()
    stage3.validate_authorization(payload, requested_step=stage3.GATE_50_STEP)
    with pytest.raises(PermissionError, match="50M gate"):
        stage3.validate_authorization(payload, requested_step=stage3.GATE_50_STEP + 1)
    (tmp_path / f"{stage3.GATE_50_STEP}.json").write_text(
        json.dumps({"result": "PASS", "continuation_authorized": True}), encoding="utf-8"
    )
    stage3.validate_authorization(payload, requested_step=stage3.GATE_75_STEP)
    with pytest.raises(PermissionError, match="75M gate"):
        stage3.validate_authorization(payload, requested_step=stage3.GATE_75_STEP + 1)


def test_failed_or_conditional_gate_never_authorizes_crossing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage3, "_gate_path", lambda step: tmp_path / f"{step}.json")
    payload = authorization()
    for result in ("FAIL", "CONDITIONAL"):
        (tmp_path / f"{stage3.GATE_50_STEP}.json").write_text(
            json.dumps({"result": result, "continuation_authorized": False}), encoding="utf-8"
        )
        with pytest.raises(PermissionError):
            stage3.validate_authorization(payload, requested_step=stage3.GATE_50_STEP + 1)


def test_three_process_segment_order_is_structural() -> None:
    manifest = {"segments": []}
    assert stage3._expected_segment(stage3.GATE_50_STEP, manifest) == stage3.SOURCE_STEP
    manifest["segments"].append({})
    assert stage3._expected_segment(stage3.GATE_75_STEP, manifest) == stage3.GATE_50_STEP
    manifest["segments"].append({})
    assert stage3._expected_segment(stage3.FINAL_STEP, manifest) == stage3.GATE_75_STEP
    with pytest.raises(ValueError):
        stage3._expected_segment(stage3.GATE_50_STEP, manifest)


def test_every_step_telemetry_fields_are_present_in_training_source() -> None:
    source = inspect.getsource(stage3.train_segment)
    required = {
        "pre_clip_gradient_norm",
        "clipping_coefficient",
        "post_clip_gradient_norm",
        "update_to_weight",
        "learning_rate_applied",
        "raw_train_loss",
        "tokens_consumed",
        "sequence_index",
    }
    assert all(value in source for value in required)
    assert "0.005" in source
    assert "2.0 * PHASE4D_GRADIENT_P95" in source


def test_checkpoint_policy_keeps_only_gate_and_final_resume_states() -> None:
    assert stage3.FULL_RESUME_STEPS == {6_103, 9_155, 11_972}
    assert stage3.MODEL_ONLY_STEPS == {4_096, 5_120, 7_168, 8_192, 10_240, 11_264}
    assert stage3.FULL_RESUME_STEPS.isdisjoint(stage3.MODEL_ONLY_STEPS)


def test_generation_audits_have_required_counts_and_corrected_eos_policy() -> None:
    source = inspect.getsource(evaluation)
    assert "200" in source and "500" in source
    assert "terminal_eos_is_not_special_token_leakage" in source
    assert "raw_outputs_retained" in source
    assert "sanitization_performed" in source
    assert evaluation.AUDIT_STEPS == (6_103, 9_155, 11_972)


def test_second_epoch_sft_qwen_and_upload_remain_unauthorized() -> None:
    payload = authorization()
    assert payload["second_epoch_authorized"] is False
    assert payload["sft_authorized"] is False
    assert payload["qwen_teacher_generation_authorized"] is False
    assert payload["upload_authorized"] is False
    source = inspect.getsource(stage3.validate_final_resume)
    assert "second_epoch_started" in source
    assert "no_second_epoch" in source


def test_export_and_memorization_requirements_are_not_satisfied_by_training_runtime() -> None:
    training_source = inspect.getsource(stage3)
    assert "huggingface_hub" not in training_source
    assert "upload_to_hub" not in training_source
    assert "second_epoch_authorized\": True" not in training_source


def test_pii_pattern_handling_is_typed_and_deterministic() -> None:
    matches = memorization.scan_pii(
        "Contact alice@example.org or visit https://example.org/docs; phone +90 (212) 555-1212."
    )
    assert matches["email"] == ["alice@example.org"]
    assert matches["url"] == ["https://example.org/docs"]
    assert matches["phone"] == ["+90 (212) 555-1212"]
    assert memorization.scan_pii("ordinary text") == {"email": [], "url": [], "phone": []}


def test_memorization_audit_schema_rejects_zero_risk_claim() -> None:
    payload = {key: {} for key in memorization.REQUIRED_AUDIT_KEYS}
    payload.update({"risk_zero_claimed": True, "hard_release_blockers": [], "result": "PASS"})
    with pytest.raises(ValueError, match="zero extraction risk"):
        memorization.validate_audit_schema(payload)
    payload["risk_zero_claimed"] = False
    memorization.validate_audit_schema(payload)


def test_memorization_audit_refuses_conditional_stop(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(json.dumps({"optimizer_step": 9155}), encoding="utf-8")
    with pytest.raises(PermissionError, match="final-stop-only"):
        memorization.require_final_checkpoint(tmp_path)


def test_export_excludes_corpus_and_runtime_state() -> None:
    stage3_export.validate_export_file_list(["model.safetensors", "config.json", "tokenizer.model"])
    for forbidden in (
        "train-00000.bin",
        "corpus_v3_tokenized/manifest.json",
        "resume_state.pt",
        "document_boundaries.jsonl",
    ):
        with pytest.raises(ValueError):
            stage3_export.validate_export_file_list([forbidden])


def test_export_refuses_75m_conditional_stop(tmp_path: Path) -> None:
    (tmp_path / "progress.json").write_text(json.dumps({"optimizer_step": 9155}), encoding="utf-8")
    with pytest.raises(PermissionError, match="final no-wrap checkpoint"):
        stage3_export.require_export_preconditions(tmp_path)
