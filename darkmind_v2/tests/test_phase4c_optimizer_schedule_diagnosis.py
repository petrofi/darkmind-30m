from __future__ import annotations

import copy

import pytest
import torch

from darkmind_v2.evaluation.phase4c_optimizer_schedule import select_policy, subset_prompts
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.training.phase4c_diagnostics import (
    build_optimizer_groups,
    build_scheduler,
    global_cosine_lr,
    historical_policy_payload,
    learning_rate_for_policy,
    optimizer_group_records,
    staged_continuation_lr,
    tiny_config,
)
from darkmind_v2.training.phase4c_training import (
    ARM_SPECS,
    arm_config,
    classify_stability,
    normalize_diagnostic_summary,
    schedule_payload,
    validate_arm_config,
    validate_training_diagnostic_schema,
)
from darkmind_v2.training.phase4c_confirmation import validate_restart_evidence
from darkmind_v2.training.phase4c_policy import build_v2_config, validate_v2_config


def test_global_scheduler_trajectory_and_bounds() -> None:
    assert global_cosine_lr(1, peak=0.0001) == pytest.approx(0.000001)
    assert global_cosine_lr(100, peak=0.0001) == pytest.approx(0.0001)
    assert global_cosine_lr(12_207, peak=0.0001) == pytest.approx(0.00003)
    assert global_cosine_lr(610, peak=0.0001) > 0.000099
    with pytest.raises(ValueError, match="outside"):
        global_cosine_lr(0, peak=0.0001)


def test_staged_scheduler_is_continuous_and_resume_safe() -> None:
    at_gate = staged_continuation_lr(610, peak=0.0001, stage1_end_lr=0.00005)
    after_gate = staged_continuation_lr(611, peak=0.0001, stage1_end_lr=0.00005)
    assert at_gate == pytest.approx(0.00005)
    assert abs(after_gate - at_gate) < 1e-9
    assert staged_continuation_lr(12_207, peak=0.0001, stage1_end_lr=0.00005) == pytest.approx(0.00003)
    assert staged_continuation_lr(3_051, peak=0.0001, stage1_end_lr=0.00005) < at_gate


def test_scheduler_applies_step_one_then_advances_once() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.0001)
    schedule = {
        "name": "warmup_cosine_global",
        "peak_learning_rate": 0.0001,
        "minimum_learning_rate": 0.00003,
        "warmup_optimizer_steps": 100,
        "scheduler_horizon_optimizer_steps": 12_207,
    }
    scheduler = build_scheduler(optimizer, schedule)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate_for_policy(1, schedule))
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate_for_policy(2, schedule))


def test_optimizer_groups_cover_every_unique_parameter_once() -> None:
    model = DarkMindV2ForCausalLM(tiny_config())
    current = build_optimizer_groups(model, corrected=False)
    corrected = build_optimizer_groups(model, corrected=True)
    current_ids = [id(parameter) for group in current for parameter in group["params"]]
    corrected_ids = [id(parameter) for group in corrected for parameter in group["params"]]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(current_ids) == len(set(current_ids))
    assert len(corrected_ids) == len(set(corrected_ids))
    assert set(current_ids) == set(corrected_ids) == expected_ids


def test_tied_embedding_is_single_no_decay_registration() -> None:
    model = DarkMindV2ForCausalLM(tiny_config())
    records, summary = optimizer_group_records(model)
    tied = [record for record in records if record["tied_parameter_identity"]]
    assert len(tied) == 1
    assert set(tied[0]["aliases"]) == {"token_embedding.weight", "lm_head.weight"}
    assert tied[0]["recommended_optimizer_group"] == "no_decay"
    assert summary["duplicate_unique_parameters"] == 0


def test_corrected_grouping_removes_decay_from_bias_norm_and_embeddings() -> None:
    model = DarkMindV2ForCausalLM(tiny_config())
    records, _ = optimizer_group_records(model)
    for record in records:
        if record["module_type"] in {"LayerNorm", "Embedding"} or record["name"].endswith(".bias"):
            assert record["recommended_weight_decay"] == 0.0
    assert any(record["module_type"] == "Linear" and record["recommended_weight_decay"] == 0.1 for record in records)


def test_historical_policy_identifies_local_vs_global_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid constructing Base V1 twice; this test covers the deterministic policy comparison logic.
    class DummyModel:
        def to(self, **_: object) -> "DummyModel":
            return self

    monkeypatch.setattr("darkmind_v2.training.phase4c_diagnostics.DarkMindV2ForCausalLM", lambda config: DummyModel())
    monkeypatch.setattr("darkmind_v2.training.phase4c_diagnostics.tensor_state_hash", lambda model: "same")
    payload = historical_policy_payload()
    assert payload["schedule"]["phase3b_used_5m_local_while_phase4_used_100m_global"] is True
    assert payload["schedule"]["phase3b"]["lr"]["610"] == pytest.approx(0.00003)
    assert payload["schedule"]["phase4b_best"]["lr"]["610"] > 0.000149
    assert payload["model"]["rederived_initialization_identity"] is True


def test_phase4c_arm_contract_enforces_exact_5m_stop() -> None:
    config = arm_config("arm1_global_lr1e4_current_groups", ARM_SPECS["arm1_global_lr1e4_current_groups"])
    validate_arm_config(config)
    assert config["authorization"] == {
        "maximum_optimizer_steps": 610,
        "maximum_training_tokens": 4_997_120,
        "phase_25m_authorized": False,
        "phase_100m_authorized": False,
    }
    changed = copy.deepcopy(config)
    changed["authorization"]["maximum_optimizer_steps"] = 3_051
    changed["deterministic_content_hash"] = canonical_hash_without_self(changed)
    with pytest.raises(ValueError, match="exceeds the 5M"):
        validate_arm_config(changed)


def canonical_hash_without_self(payload: dict) -> str:
    from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash

    return canonical_json_hash({key: value for key, value in payload.items() if key != "deterministic_content_hash"})


def test_stability_classification_requires_rebound_and_late_curve_health() -> None:
    stable = {
        "integrity_pass": True,
        "validation_improvement_percent": 20.0,
        "eval_improvement_percent": 20.0,
        "validation_rebound_percent": 1.0,
        "eval_rebound_percent": 1.0,
        "last_three_validation_sustained_worsening": False,
        "last_three_eval_sustained_worsening": False,
        "abnormal_diagnostic_growth": False,
        "runtime_stable": True,
    }
    assert classify_stability(stable) == "stable"
    assert classify_stability({**stable, "validation_rebound_percent": 3.0}) == "partial stability"
    assert classify_stability({**stable, "last_three_eval_sustained_worsening": True}) == "partial stability"
    assert classify_stability({**stable, "validation_improvement_percent": 2.0}) == "unstable"


def test_diagnostic_normalization_ignores_tiny_bias_ratios_and_initial_residual_scale() -> None:
    summary = {
        "integrity_pass": True,
        "validation_improvement_percent": 20.0,
        "eval_improvement_percent": 20.0,
        "validation_rebound_percent": 0.0,
        "eval_rebound_percent": 0.0,
        "last_three_validation_sustained_worsening": False,
        "last_three_eval_sustained_worsening": False,
        "gradient_norm_max": 2.0,
        "runtime_stable": True,
        "stability": "partial stability",
    }
    snapshots = {
        "0": {
            "parameter_diagnostics": [],
            "activation": {"final_to_early_residual_ratio": 50.0, "logits": {"std": 0.5}},
        },
        "610": {
            "parameter_diagnostics": [
                {"parameter": "blocks.0.attn.proj.weight", "update_to_weight_ratio": 0.002},
                {"parameter": "blocks.0.attn.proj.bias", "update_to_weight_ratio": 0.04},
            ],
            "activation": {"final_to_early_residual_ratio": 140.0, "logits": {"std": 2.0}},
        },
    }
    normalized = normalize_diagnostic_summary(summary, snapshots)
    assert normalized["maximum_matrix_parameter_update_to_weight_ratio"] == pytest.approx(0.002)
    assert normalized["maximum_residual_ratio_growth_multiple_from_step0"] == pytest.approx(2.8)
    assert normalized["abnormal_diagnostic_growth"] is False
    assert normalized["stability"] == "stable"


def test_training_diagnostic_schema_requires_activation_and_update_records() -> None:
    snapshot = {
        "optimizer_step": 64,
        "parameter_diagnostics": [{"module": "blocks.0.attn.qkv", "update_to_weight_ratio": 1e-4}],
        "clipped_step_fraction": 0.1,
        "embedding_norm": {},
        "residual_projection_norms": {},
        "layernorm_weight_statistics": {},
        "activation": {"layers": [{"layer": 0}], "logits": {"std": 1.0}},
        "probe_losses": {"turkish_prose": {"loss": 8.0}},
    }
    validate_training_diagnostic_schema(snapshot)
    with pytest.raises(ValueError, match="activation diagnostic"):
        validate_training_diagnostic_schema({**snapshot, "activation": {"layers": []}})


def test_generation_subset_is_fixed_and_category_balanced() -> None:
    prompts = []
    for language, category in (
        ("tr", "ordinary_text"),
        ("en", "ordinary_text"),
        ("tr", "technical"),
        ("en", "technical"),
        ("tr", "factual_encyclopedic"),
        ("en", "factual_encyclopedic"),
        ("code", "code_structured"),
    ):
        prompts.extend(
            {"id": f"{language}-{category}-{index}", "language": language, "category": category}
            for index in range(3)
        )
    selected = subset_prompts(prompts)
    assert len(selected) == 14
    assert all(selected.count(item) == 1 for item in selected)


def test_policy_selection_prefers_correct_grouping_within_one_percent() -> None:
    base = {
        "stability": "stable",
        "generation_hard_failure": False,
        "final_validation_loss": 6.0,
        "final_eval_loss": 6.0,
        "optimizer_grouping": "current_decay_all",
        "schedule": {"name": "warmup_cosine_global", "peak_learning_rate": 0.0001},
    }
    summaries = {
        "current": base,
        "corrected": {
            **base,
            "final_validation_loss": 6.01,
            "final_eval_loss": 6.01,
            "optimizer_grouping": "corrected_adamw_v1",
        },
        "lower_lr": {
            **base,
            "final_validation_loss": 6.3,
            "final_eval_loss": 6.3,
            "schedule": {"name": "warmup_cosine_global", "peak_learning_rate": 0.000075},
        },
    }
    assert select_policy(summaries) == "corrected"


def test_v2_policy_freezes_5m_confirmation_and_future_resume_contract() -> None:
    schedule = schedule_payload("global", 0.0001)
    config = build_v2_config(
        {
            "optimizer_grouping": "corrected_adamw_v1",
            "initialization_policy": "base_v1_standard_v1",
            "schedule": schedule,
            "sequence_order": "deterministic_stratified_v1",
        }
    )
    validate_v2_config(config)
    assert config["authorization"]["maximum_training_tokens"] == 4_997_120
    assert config["stage_gates"]["25m"]["authorized"] is False
    assert config["continuation"]["next_optimizer_step_after_5m"] == 611
    assert config["schedule"]["learning_rate_milestones"]["100m_step_12207"] == pytest.approx(0.00003)


def test_process_restart_evidence_requires_three_fresh_processes_and_exact_ranges() -> None:
    checks = {
        "fresh_processes": True,
        "segment_boundary_exact": True,
        "rng_continuity": True,
        "scheduler_continuity": True,
        "data_position_continuity": True,
        "no_repeated_or_skipped_sequence": True,
        "midpoint_resume_checkpoint_present": True,
        "final_resume_checkpoint_present": True,
    }
    validate_restart_evidence({"checks": checks})
    with pytest.raises(ValueError, match="process-restart"):
        validate_restart_evidence({"checks": {**checks, "rng_continuity": False}})
