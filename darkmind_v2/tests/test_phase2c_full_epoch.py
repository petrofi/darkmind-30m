import math
from pathlib import Path

import pytest

from darkmind_v2.training.train_tiny_full_epoch import checkpoint_name
from darkmind_v2.training.validate_full_epoch_config import (
    learning_rate_for_step,
    load_and_validate_full_epoch_config,
)


CONFIG = Path("darkmind_v2/config/train_tiny_full_epoch.json")


def test_full_epoch_config_has_exact_coverage_contract() -> None:
    config = load_and_validate_full_epoch_config(CONFIG)
    assert config["maximum_optimizer_steps"] == 2867
    assert config["maximum_total_training_tokens"] == 11_743_232
    assert config["data"]["train_corpus_tokens"] == 11_744_226
    assert config["data"]["excluded_tail_tokens"] == 994
    assert config["maximum_total_training_tokens"] / config["data"]["train_corpus_tokens"] == pytest.approx(
        0.9999153640147327
    )


def test_full_epoch_profiles_preserve_effective_batch() -> None:
    config = load_and_validate_full_epoch_config(CONFIG)
    for profile in config["profiles"]:
        assert (
            profile["micro_batch_size"]
            * profile["gradient_accumulation_steps"]
            * config["data"]["sequence_length"]
            == 4096
        )


def test_full_epoch_scheduler_hits_peak_and_final_minimum() -> None:
    config = load_and_validate_full_epoch_config(CONFIG)
    assert learning_rate_for_step(1, config) == pytest.approx(0.000003)
    assert learning_rate_for_step(100, config) == pytest.approx(0.0003)
    assert learning_rate_for_step(101, config) < learning_rate_for_step(100, config)
    assert learning_rate_for_step(1435, config) < learning_rate_for_step(1434, config)
    assert math.isclose(learning_rate_for_step(2867, config), 0.00003, abs_tol=1e-15)


def test_full_epoch_scheduler_rejects_out_of_plan_steps() -> None:
    config = load_and_validate_full_epoch_config(CONFIG)
    with pytest.raises(ValueError, match="outside the planned run"):
        learning_rate_for_step(0, config)
    with pytest.raises(ValueError, match="outside the planned run"):
        learning_rate_for_step(2868, config)


def test_full_epoch_checkpoint_names_are_unambiguous() -> None:
    assert checkpoint_name(0, 0) == "step_000000_tokens_000000000"
    assert checkpoint_name(1434, 5_873_664) == "step_001434_tokens_005873664"
    assert checkpoint_name(2867, 11_743_232) == "step_002867_tokens_011743232"
