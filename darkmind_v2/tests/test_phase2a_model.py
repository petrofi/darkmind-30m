import json

import pytest
import torch

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.model_io import load_model_package, save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM


CONFIG_PATH = "darkmind_v2/config/model_tiny_smoke.json"


def small_config(**overrides) -> DarkMindV2Config:
    values = {
        "vocab_size": 24000,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 32,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "bias": True,
        "seed": 123,
    }
    values.update(overrides)
    return DarkMindV2Config(**values)


def test_tiny_config_schema_and_exact_parameter_count() -> None:
    config = DarkMindV2Config.from_json_file(CONFIG_PATH)
    model = DarkMindV2ForCausalLM(config)
    estimate = estimate_model_size(
        vocab_size=config.vocab_size,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        tied_embeddings=config.tie_word_embeddings,
        bias=config.bias,
    )
    assert config.schema_version == "darkmind-v2-model-config-v1"
    assert estimate.total_params == 9_369_088
    assert model.parameter_count() == estimate.total_params


def test_forward_shape_shifted_loss_and_tied_weights() -> None:
    model = DarkMindV2ForCausalLM(small_config())
    input_ids = torch.tensor([[2, 101, 102, 103, 3], [2, 201, 202, 0, 0]])
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 5, 24000)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.embeddings_are_tied()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


def test_causal_mask_blocks_future_information() -> None:
    model = DarkMindV2ForCausalLM(small_config()).eval()
    left = torch.tensor([[2, 10, 11, 12, 13]])
    right = torch.tensor([[2, 10, 11, 999, 998]])
    with torch.no_grad():
        left_logits = model(left).logits
        right_logits = model(right).logits
    assert torch.equal(left_logits[:, :3], right_logits[:, :3])


def test_deterministic_initialization() -> None:
    first = DarkMindV2ForCausalLM(small_config(seed=777))
    second = DarkMindV2ForCausalLM(small_config(seed=777))
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)


def test_safetensors_save_reload_equality(tmp_path) -> None:
    model = DarkMindV2ForCausalLM(small_config()).eval()
    input_ids = torch.tensor([[2, 7, 8, 9]])
    expected = model(input_ids).logits
    package = tmp_path / "model"
    save_model_package(model, package)
    loaded = load_model_package(package)
    actual = loaded(input_ids).logits
    assert torch.equal(expected, actual)
    assert loaded.embeddings_are_tied()


@pytest.mark.parametrize(
    "override",
    [
        {"tie_word_embeddings": False},
        {"vocab_size": 23999},
        {"pad_token_id": 9},
        {"activation": "relu"},
    ],
)
def test_invalid_model_config_rejected(override) -> None:
    with pytest.raises(ValueError):
        small_config(**override)


def test_config_json_contains_no_inferred_architecture_defaults() -> None:
    payload = json.loads(open(CONFIG_PATH, encoding="utf-8").read())
    required = {
        "schema_version",
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "mlp_ratio",
        "normalization",
        "activation",
        "position_embedding_type",
        "attention_type",
        "tie_word_embeddings",
        "special_token_ids",
    }
    represented = set(payload) | ({"special_token_ids"} if all(key in payload for key in (
        "pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id",
        "system_token_id", "user_token_id", "assistant_token_id", "end_token_id",
    )) else set())
    assert required <= represented


def test_sampling_profiles_are_deterministic_and_top_k_one_matches_greedy() -> None:
    model = DarkMindV2ForCausalLM(small_config()).eval()
    input_ids = torch.tensor([[2, 10, 11]])
    first = model.generate_tokens(
        input_ids, max_new_tokens=4, do_sample=True, temperature=0.7, top_k=40, top_p=0.9, seed=99
    )
    second = model.generate_tokens(
        input_ids, max_new_tokens=4, do_sample=True, temperature=0.7, top_k=40, top_p=0.9, seed=99
    )
    assert torch.equal(first, second)
    greedy = model.generate_tokens(input_ids, max_new_tokens=4)
    top_one = model.generate_tokens(input_ids, max_new_tokens=4, do_sample=True, top_k=1, seed=99)
    assert torch.equal(greedy, top_one)


def test_generation_stops_on_eos_and_rejects_invalid_sampling_settings() -> None:
    model = DarkMindV2ForCausalLM(small_config()).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    input_ids = torch.tensor([[2, 10]])
    generated = model.generate_tokens(input_ids, max_new_tokens=8, eos_token_id=0)
    assert generated.shape[1] == input_ids.shape[1] + 1
    assert generated[0, -1].item() == 0
    with pytest.raises(ValueError):
        model.generate_tokens(input_ids, max_new_tokens=1, top_k=0)
    with pytest.raises(ValueError):
        model.generate_tokens(input_ids, max_new_tokens=1, top_p=0.0)
