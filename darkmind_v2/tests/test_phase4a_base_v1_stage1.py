import json
import math
from pathlib import Path

import pytest
import torch

from darkmind_v2.evaluation.audit_public_preview import SAMPLING_SUBSET_COUNTS
from darkmind_v2.export.export_base_v1_stage1 import REQUIRED_FILES
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import verify_frozen_tokenizer
from darkmind_v2.training.checkpointing import load_checkpoint, save_checkpoint
from darkmind_v2.training.train_base_v1_stage1 import build_optimizer, build_scheduler, checkpoint_name
from darkmind_v2.training.training_state import TrainingState
from darkmind_v2.training.validate_phase3b_pilot_config import load_and_validate_phase3b_pilot_config
from darkmind_v2.training.validate_phase4a_config import (
    DEFAULT_CONFIG,
    SCHEDULER_STEPS,
    SCHEDULER_TOKENS,
    STAGE1_STEPS,
    STAGE1_TOKENS,
    TOKENS_PER_STEP,
    classify_stage1,
    learning_rate_for_step,
    load_and_validate_phase4a_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load_contract() -> dict:
    return load_and_validate_phase4a_config(DEFAULT_CONFIG)


def test_production_horizon_and_exact_stage1_stop() -> None:
    config = load_contract()
    assert SCHEDULER_STEPS == 12_207
    assert SCHEDULER_TOKENS == 99_999_744 == SCHEDULER_STEPS * TOKENS_PER_STEP
    assert STAGE1_STEPS == 610
    assert STAGE1_TOKENS == 4_997_120 == STAGE1_STEPS * TOKENS_PER_STEP
    assert config["stage_gates"]["25m"] == {
        "authorized": False,
        "optimizer_steps": 3_051,
        "tokens": 24_993_792,
    }
    assert config["stage_gates"]["100m"]["authorized"] is False
    assert 100_000_000 - SCHEDULER_TOKENS == 256


def test_scheduler_does_not_reset_or_end_at_stage_boundaries() -> None:
    config = load_contract()
    lr_610 = learning_rate_for_step(610, config)
    lr_611 = learning_rate_for_step(611, config)
    lr_3051 = learning_rate_for_step(3_051, config)
    lr_3052 = learning_rate_for_step(3_052, config)
    assert lr_610 > config["schedule"]["minimum_learning_rate"]
    assert lr_611 < lr_610
    assert lr_3052 < lr_3051
    assert math.isclose(learning_rate_for_step(12_207, config), 0.00003, abs_tol=1e-15)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_for_step(12_208, config)


def test_config_rejects_budget_hash_and_traversal_mutations(tmp_path: Path) -> None:
    original = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mutations = [
        ("authorization", "maximum_training_tokens", STAGE1_TOKENS + TOKENS_PER_STEP),
        ("corpus", "tokenized_manifest_hash", "0" * 64),
        ("data", "no_data_wrap", False),
        ("schedule", "scheduler_horizon_optimizer_steps", 610),
    ]
    for index, (group, key, value) in enumerate(mutations):
        payload = json.loads(json.dumps(original))
        payload[group][key] = value
        path = tmp_path / f"broken_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_and_validate_phase4a_config(path)


def test_frozen_base_tokenizer_and_corpus_contracts() -> None:
    config = load_contract()
    model = DarkMindV2Config.from_json_file(ROOT / "config" / "model_base_v1.json")
    assert model_config_hash(model) == "3a2dda86293ceae23ca4e50ea47c840b7fc46021d293c862d330110851ac8305"
    assert model.gradient_checkpointing is False
    assert model.attention_implementation == "sdpa"
    tokenizer = verify_frozen_tokenizer()
    assert tokenizer["vocab_size"] == 24_000
    assert config["corpus"]["corpus_hash"] == "e75c4aa4f39cc7a3cb4fe754e2a0e85268ced300f8504a86d443540eb609e1c5"


def test_deterministic_no_wrap_sequence_and_milestone_contract() -> None:
    config = load_contract()
    data = config["data"]
    assert data["sequence_order"] == "contiguous_complete_train_sequences_from_offset_zero"
    assert data["no_replacement_sampling"] is True
    assert data["no_sequence_repetition"] is True
    assert data["no_data_wrap"] is True
    assert config["evaluation"]["milestone_steps"] == [0, 128, 305, 458, 610]
    assert [checkpoint_name(step, step * TOKENS_PER_STEP) for step in config["evaluation"]["milestone_steps"]] == [
        "step_000000_tokens_000000000",
        "step_000128_tokens_001048576",
        "step_000305_tokens_002498560",
        "step_000458_tokens_003751936",
        "step_000610_tokens_004997120",
    ]


def test_optimizer_scheduler_rng_and_data_position_restore(tmp_path: Path) -> None:
    config = load_contract()
    small = DarkMindV2Config(block_size=16, n_layer=1, n_head=2, n_embd=32, mlp_hidden_size=64, seed=17)
    model = DarkMindV2ForCausalLM(small)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    batch = torch.randint(8, 100, (2, 16))
    loss = model(batch, labels=batch).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step()
    state = TrainingState(step=1, tokens_seen=TOKENS_PER_STEP, data_position=TOKENS_PER_STEP)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        tokenizer_hash="tokenizer-hash",
        data_manifest_hash="data-hash",
    )
    restored_model = DarkMindV2ForCausalLM(small)
    restored_optimizer = build_optimizer(restored_model, config)
    restored_scheduler = build_scheduler(restored_optimizer, config)
    restored = load_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_tokenizer_hash="tokenizer-hash",
        expected_data_manifest_hash="data-hash",
    )
    assert restored == state
    assert restored_scheduler.last_epoch == 1
    assert restored_optimizer.state_dict()["state"]


@pytest.mark.parametrize(
    "initial,final,expected",
    [(10.0, 8.0, "Strong PASS"), (10.0, 9.0, "Conditional PASS"), (10.0, 9.6, "FAIL")],
)
def test_stage1_classification_thresholds(initial: float, final: float, expected: str) -> None:
    assert classify_stage1(initial, final, initial, final)["classification"] == expected


def test_authoritative_generation_and_local_export_policy() -> None:
    config = load_contract()
    assert config["evaluation"]["authoritative_greedy_generations"] == 200
    assert config["evaluation"]["authoritative_seeded_generations"] == 500
    assert sum(SAMPLING_SUBSET_COUNTS.values()) == 50
    assert {"model.safetensors", "README.md", "provenance_manifest.json", "evaluation_results.json"} <= REQUIRED_FILES
    export_source = (ROOT / "export" / "export_base_v1_stage1.py").read_text(encoding="utf-8")
    assert "upload_performed\": False" in export_source
    assert "model-weight license is unresolved" in export_source


def test_tiny_and_phase3b_finalist_contracts_remain_backward_compatible() -> None:
    tiny = DarkMindV2Config.from_json_file(ROOT / "config" / "model_tiny_smoke.json")
    assert model_config_hash(tiny) == "61ebdadb15d4fde09db9842413450f512a12fc7f7e6fcaead7dc1347478f4414"
    phase3b = load_and_validate_phase3b_pilot_config(ROOT / "config" / "phase3b_finalist_pilot.json")
    assert phase3b["pilot"]["maximum_optimizer_steps"] == 610
    assert phase3b["candidates"]["D"].endswith("model_base_candidate_d_120m.json")
