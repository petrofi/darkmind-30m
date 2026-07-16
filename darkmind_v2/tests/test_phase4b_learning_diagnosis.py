from __future__ import annotations

import json
import os
from copy import deepcopy
from collections import Counter
from pathlib import Path

import pytest

from darkmind_v2.evaluation.phase4b_factorial import factor_effects, select_stable_policy
from darkmind_v2.training import phase4b_runtime
from darkmind_v2.training.phase4b_factorial import (
    ARM_SPECS,
    OrderedTokenDataset,
    RUNTIME_ROOT,
    build_arm_config,
    factorial_contract,
    rebound_percent,
    require_initialization_hash_identity,
    training_stability,
    validate_arm_config,
)
from darkmind_v2.training.phase4b_policy import (
    build_v2_config,
    validate_25m_resume_compatibility,
    validate_confirmation_requirements,
)
from darkmind_v2.training.phase4b_runtime import (
    _order_manifest,
    deterministic_stratified_order,
    ensure_runtime_path,
)


def test_onedrive_runtime_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="OneDrive"):
        ensure_runtime_path(Path(r"C:\Users\test\OneDrive\phase4b\run"))


def test_atomic_progress_write_retries_transient_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase4b_runtime, "RUNTIME_ROOT", tmp_path)
    attempts = []

    def flaky_replace(source: str, destination: str) -> None:
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("transient lock")
        os.replace(source, destination)

    target = tmp_path / "runs" / "progress.json"
    phase4b_runtime.atomic_write_json(target, {"step": 5}, retries=3, replace=flaky_replace)
    assert len(attempts) == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"step": 5}
    assert not list(target.parent.glob("*.tmp"))


def _synthetic_labels() -> list[tuple[str, str, str]]:
    return (
        [("tr", "prose", "source_a")] * 300
        + [("en", "prose", "source_b")] * 200
        + [("tr", "technical", "source_c")] * 60
        + [("en", "technical", "source_d")] * 40
    )


def test_stratified_order_repeatability_coverage_and_balance() -> None:
    labels = _synthetic_labels()
    first = deterministic_stratified_order(labels)
    second = deterministic_stratified_order(labels)
    assert first == second
    assert len(first) == len(set(first)) == len(labels)
    assert sorted(first) == list(range(len(labels)))
    expected = Counter(labels)
    for start in range(0, len(first), 100):
        observed = Counter(labels[index] for index in first[start : start + 100])
        for label, count in expected.items():
            expected_share = count / len(labels)
            observed_share = observed[label] / len(first[start : start + 100])
            assert abs(observed_share - expected_share) <= 0.02


def test_order_manifest_hash_stability_and_no_wrap() -> None:
    labels = _synthetic_labels()
    order = deterministic_stratified_order(labels)
    first = _order_manifest("deterministic_stratified_v1", order, [labels[index] for index in order])
    second = _order_manifest("deterministic_stratified_v1", order, [labels[index] for index in order])
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]
    assert first["no_replacement"] is True
    assert first["no_wrap"] is True
    assert first["covers_all_complete_train_sequences"] is True
    assert first["excludes_validation_and_eval"] is True


def test_sequence_order_process_resume_continuity() -> None:
    dataset = OrderedTokenDataset.__new__(OrderedTokenDataset)
    dataset.order = deterministic_stratified_order(_synthetic_labels())
    sequences_per_step = 16
    midpoint_sequences = 19 * sequences_per_step
    total_sequences = 37 * sequences_per_step
    uninterrupted = dataset.source_indices(0, total_sequences * 512)
    before_restart = dataset.source_indices(0, midpoint_sequences * 512)
    after_restart = dataset.source_indices(
        midpoint_sequences * 512,
        (total_sequences - midpoint_sequences) * 512,
    )
    assert before_restart + after_restart == uninterrupted


def test_factorial_configs_differ_only_by_predeclared_factors() -> None:
    configs = [
        build_arm_config(name, spec["order"], spec["peak_learning_rate"], RUNTIME_ROOT / "runs" / name)
        for name, spec in ARM_SPECS.items()
    ]
    assert factorial_contract(configs)["result"] == "PASS"
    for config in configs:
        validate_arm_config(config)
        assert config["authorization"]["maximum_optimizer_steps"] == 610
        assert config["authorization"]["maximum_training_tokens"] == 4_997_120
        assert config["authorization"]["phase_25m_authorized"] is False


def test_initialization_hash_identity_contract() -> None:
    assert require_initialization_hash_identity({"a": "same", "b": "same"}) == "same"
    with pytest.raises(ValueError, match="initialization hashes differ"):
        require_initialization_hash_identity({"a": "one", "b": "two"})


def test_rebound_and_stability_classification() -> None:
    assert rebound_percent(8.0, 8.08) == pytest.approx(1.0)
    stable = {
        "integrity_pass": True,
        "validation_improvement_percent": 20.0,
        "eval_improvement_percent": 19.0,
        "validation_rebound_percent": 1.0,
        "eval_rebound_percent": 1.5,
        "sustained_divergence": False,
    }
    assert training_stability(stable) == "stable"
    assert training_stability({**stable, "validation_rebound_percent": 6.0}) == "unstable"
    assert training_stability({**stable, "validation_improvement_percent": 10.0}) == "partial stability"


def _arm(final_validation: float, final_eval: float, stability: str, lr: float, order: str) -> dict:
    return {
        "final_validation_loss": final_validation,
        "final_eval_loss": final_eval,
        "validation_rebound_percent": 1.0,
        "eval_rebound_percent": 1.0,
        "stability": stability,
        "peak_learning_rate": lr,
        "sequence_order": order,
    }


def test_factor_effects_and_corrected_policy_selection() -> None:
    summaries = {
        "arm_a_legacy_lr3e4": _arm(10.0, 10.2, "unstable", 0.0003, "legacy_order_v1"),
        "arm_b_legacy_lr15e5": _arm(8.5, 8.6, "partial stability", 0.00015, "legacy_order_v1"),
        "arm_c_stratified_lr3e4": _arm(8.0, 8.2, "stable", 0.0003, "deterministic_stratified_v1"),
        "arm_d_stratified_lr15e5": _arm(7.5, 7.6, "stable", 0.00015, "deterministic_stratified_v1"),
    }
    validation = factor_effects(summaries, "final_validation_loss")
    assert validation["lower_lr_main_effect_loss_reduction"] == pytest.approx(1.0)
    assert validation["stratified_order_main_effect_loss_reduction"] == pytest.approx(1.5)
    assert validation["interaction_effect"] == pytest.approx(1.0)
    assert select_stable_policy(summaries) == "arm_d_stratified_lr15e5"
    assert select_stable_policy({name: {**item, "stability": "unstable"} for name, item in summaries.items()}) is None


def test_v2_config_contract_is_deterministic_and_stays_at_5m_gate() -> None:
    policy = {"peak_learning_rate": 0.00015, "sequence_order": "deterministic_stratified_v1"}
    first = build_v2_config(policy)
    second = build_v2_config(policy)
    assert first == second
    assert first["deterministic_content_hash"] == second["deterministic_content_hash"]
    assert first["authorization"] == {
        "maximum_optimizer_steps": 610,
        "maximum_training_tokens": 4_997_120,
        "phase_25m_authorized": False,
        "phase_100m_authorized": False,
    }
    validate_confirmation_requirements(first)
    validate_25m_resume_compatibility(first)


def test_confirmation_contract_rejects_factorial_reuse_and_25m_authorization() -> None:
    config = build_v2_config({"peak_learning_rate": 0.00015, "sequence_order": "legacy_order_v1"})
    reused = deepcopy(config)
    reused["confirmation"]["reuse_factorial_run"] = True
    with pytest.raises(ValueError, match="confirmation run contract changed"):
        validate_confirmation_requirements(reused)
    extended = deepcopy(config)
    extended["authorization"]["phase_25m_authorized"] = True
    with pytest.raises(ValueError, match="exceeds the 5M gate"):
        validate_confirmation_requirements(extended)


def test_25m_resume_contract_rejects_identity_or_horizon_changes() -> None:
    config = build_v2_config({"peak_learning_rate": 0.0002, "sequence_order": "deterministic_stratified_v1"})
    changed_identity = deepcopy(config)
    changed_identity["architecture_hash"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        validate_25m_resume_compatibility(changed_identity)
    changed_horizon = deepcopy(config)
    changed_horizon["schedule"]["scheduler_horizon_optimizer_steps"] = 3_051
    with pytest.raises(ValueError, match="scheduler horizon changed"):
        validate_25m_resume_compatibility(changed_horizon)
