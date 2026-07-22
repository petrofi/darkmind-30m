from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from darkmind_v2.evaluation import phase4f_generation as generation
from darkmind_v2.evaluation import phase4f_memorization as memorization
from darkmind_v2.evaluation import phase4f_reporting as reporting
from darkmind_v2.export import export_phase4f_first_pass as phase4f_export
from darkmind_v2.training import phase4f_completion as phase4f


ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = ROOT / "darkmind_v2" / "config" / "train_base_v1_first_pass_completion_authorization.json"


def authorization() -> dict:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_frozen_identity_and_exact_no_wrap_capacity() -> None:
    payload = authorization()
    assert payload["v2_config_sha256"] == phase4f.EXPECTED_V2_FILE_HASH
    assert payload["architecture_hash"] == phase4f.EXPECTED_ARCHITECTURE_HASH
    assert payload["model_config_sha256"] == phase4f.EXPECTED_CONFIG_SHA256
    assert payload["tokenizer_hashes"] == phase4f.EXPECTED_HASHES
    assert payload["corpus_hashes"]["sequence_order"] == phase4f.EXPECTED_ORDER_HASH
    assert phase4f.UNIQUE_SEQUENCE_CAPACITY == 191_566
    assert phase4f.USABLE_SEQUENCE_CAPACITY == 191_552
    assert phase4f.EXCLUDED_TAIL_SEQUENCES == 14
    assert phase4f.FINAL_STEP == 11_972
    assert phase4f.FINAL_TOKENS == 98_074_624


def test_authorization_rejects_step_11973_and_token_98074625(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4f, "gate_passed", lambda _: True)
    payload = authorization()
    phase4f.validate_authorization(payload, requested_step=phase4f.FINAL_STEP)
    with pytest.raises(PermissionError, match="no-wrap"):
        phase4f.validate_authorization(payload, requested_step=11_973)
    with pytest.raises(PermissionError, match="token limit"):
        phase4f.validate_authorization(payload, requested_step=phase4f.FINAL_STEP, requested_tokens=98_074_625)


def test_85m_90m_95m_gates_are_sequential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4f, "gate_path", lambda step: tmp_path / f"{step}.json")
    payload = authorization()
    phase4f.validate_authorization(payload, requested_step=phase4f.GATE_85_STEP)
    with pytest.raises(PermissionError, match="85M"):
        phase4f.validate_authorization(payload, requested_step=phase4f.GATE_85_STEP + 1)
    (tmp_path / f"{phase4f.GATE_85_STEP}.json").write_text(json.dumps({"result": "PASS", "continuation_authorized": True}), encoding="utf-8")
    phase4f.validate_authorization(payload, requested_step=phase4f.GATE_90_STEP)
    with pytest.raises(PermissionError, match="90M"):
        phase4f.validate_authorization(payload, requested_step=phase4f.GATE_90_STEP + 1)
    (tmp_path / f"{phase4f.GATE_90_STEP}.json").write_text(json.dumps({"result": "PASS", "continuation_authorized": True}), encoding="utf-8")
    phase4f.validate_authorization(payload, requested_step=phase4f.GATE_95_STEP)
    with pytest.raises(PermissionError, match="95M"):
        phase4f.validate_authorization(payload, requested_step=phase4f.GATE_95_STEP + 1)


def test_failed_or_conditional_gate_never_authorizes_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4f, "gate_path", lambda step: tmp_path / f"{step}.json")
    for result in ("FAIL", "CONDITIONAL"):
        (tmp_path / f"{phase4f.GATE_85_STEP}.json").write_text(
            json.dumps({"result": result, "continuation_authorized": False}), encoding="utf-8"
        )
        with pytest.raises(PermissionError):
            phase4f.validate_authorization(authorization(), requested_step=phase4f.GATE_85_STEP + 1)


def test_three_fresh_process_segments_and_checkpoint_policy_are_structural() -> None:
    manifest = {"segments": []}
    assert phase4f.expected_segment(phase4f.GATE_85_STEP, manifest) == phase4f.SOURCE_STEP
    manifest["segments"].append({})
    assert phase4f.expected_segment(phase4f.GATE_95_STEP, manifest) == phase4f.GATE_85_STEP
    manifest["segments"].append({})
    assert phase4f.expected_segment(phase4f.FINAL_STEP, manifest) == phase4f.GATE_95_STEP
    assert phase4f.FULL_RESUME_STEPS == {10_375, 11_596, 11_972}
    assert phase4f.MODEL_ONLY_STEPS == {10_986}
    assert phase4f.FULL_RESUME_STEPS.isdisjoint(phase4f.MODEL_ONLY_STEPS)


def test_segment_b_establishes_90m_gate_in_process() -> None:
    source = inspect.getsource(phase4f.train_segment)
    assert "GATE_95_STEP: GATE_90_STEP" in source
    assert "evaluate_gate(GATE_90_STEP)" in source
    assert "validate_authorization(authorization, requested_step=step)" in source


def test_no_second_epoch_transition_and_required_telemetry() -> None:
    payload = authorization()
    assert payload["second_epoch_authorized"] is False
    assert payload["sequence_replacement_authorized"] is False
    assert payload["partial_effective_batch_authorized"] is False
    source = inspect.getsource(phase4f.train_segment)
    for field in (
        "pre_clip_gradient_norm",
        "clipping_coefficient",
        "post_clip_gradient_norm",
        "update_to_weight",
        "learning_rate_applied",
        "raw_train_loss",
        "tokens_consumed",
        "sequence_index",
    ):
        assert field in source


def test_flat_learning_classification_and_strong_pass() -> None:
    common = {
        "validation_rebound": 0.2,
        "eval_rebound": 0.2,
        "integrity_pass": True,
        "memorization_pass": True,
        "catastrophic_probe_regression": False,
        "sustained_late_worsening": False,
    }
    assert reporting.classify_first_pass(validation_improvement=0.1, eval_improvement=-0.1, **common) == "PLATEAU"
    assert reporting.classify_first_pass(validation_improvement=1.2, eval_improvement=1.1, **common) == "STRONG PASS"
    assert reporting.classify_first_pass(validation_improvement=-1.1, eval_improvement=0.2, **common) == "FAIL"


def test_best_checkpoint_selection_is_validation_only_and_deterministic() -> None:
    evaluations = {
        "10375": {"validation": {"loss": 5.0}},
        "10986": {"validation": {"loss": 4.9}},
        "11596": {"validation": {"loss": 4.9}},
        "11972": {"validation": {"loss": 4.95}},
    }
    assert reporting.select_best_validation_step(evaluations, [10_375, 10_986, 11_596, 11_972]) == 10_986


def test_memorization_schema_pii_and_hard_gates() -> None:
    matches = memorization.scan_pii("alice@example.org +90 (212) 555-1212 https://example.org/docs.")
    assert matches["email"] == ["alice@example.org"]
    assert matches["phone"] == ["+90 (212) 555-1212"]
    assert matches["url"] == ["https://example.org/docs"]
    assert memorization.is_plausible_phone_identity("+90 (212) 555-1212") is True
    assert memorization.is_plausible_phone_identity("970-970-970-970-97") is False
    assert memorization.is_plausible_phone_identity("1966 - 1966") is False
    assert memorization.is_plausible_phone_identity("100000000000000000000") is False
    blockers = memorization.determine_hard_blockers(
        material_personal_data_reproduction_count=1,
        pii_like_generation_counts={"email": 1, "phone": 0, "url": 0},
        train_long_count=8,
        train_count=20,
        heldout_long_count=1,
        heldout_count=20,
    )
    assert blockers == [
        "material_personal_data_reproduction",
        "unexplained_email_or_phone_generation",
        "materially_higher_long_span_training_extraction",
    ]
    payload = {key: {} for key in memorization.REQUIRED_AUDIT_KEYS}
    payload.update({"risk_zero_claimed": False, "hard_release_blockers": [], "result": "PASS"})
    memorization.validate_audit_schema(payload)
    payload["risk_zero_claimed"] = True
    with pytest.raises(ValueError, match="zero extraction risk"):
        memorization.validate_audit_schema(payload)


def test_generation_counts_and_corrected_eos_contract() -> None:
    source = inspect.getsource(generation)
    assert "200" in source and "500" in source
    assert "terminal_eos_is_not_special_token_leakage" in source
    assert generation.AUDIT_STEPS == (10_375, 10_986, 11_596, 11_972)


def test_export_excludes_corpus_and_runtime_state() -> None:
    phase4f_export.validate_export_file_list(["model.safetensors", "config.json", "tokenizer.model"])
    for forbidden in (
        "train-00000.bin",
        "corpus_v3_tokenized/manifest.json",
        "resume_state.pt",
        "document_boundaries.jsonl",
        "optimizer.pt",
    ):
        with pytest.raises(ValueError):
            phase4f_export.validate_export_file_list([forbidden])
