import json

import pytest
import torch

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.model_io import load_model_package, model_config_hash, save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.training.benchmark_base_candidates import validate_benchmark_report


CANDIDATES = {
    "A": ("darkmind_v2/config/model_base_candidate_a_60m.json", 62_989_312),
    "B": ("darkmind_v2/config/model_base_candidate_b_80m.json", 79_694_720),
    "C": ("darkmind_v2/config/model_base_candidate_c_100m.json", 103_881_216),
    "D": ("darkmind_v2/config/model_base_candidate_d_120m.json", 118_056_960),
}


def small_config(**overrides) -> DarkMindV2Config:
    payload = {
        "block_size": 16,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "mlp_hidden_size": 96,
        "dropout": 0.0,
        "seed": 41,
    }
    payload.update(overrides)
    return DarkMindV2Config(**payload)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_candidate_configs_have_exact_counts_and_valid_shares(candidate) -> None:
    path, expected = CANDIDATES[candidate]
    config = DarkMindV2Config.from_json_file(path)
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
    assert estimate.vocab_related_percentage < 20.0
    assert estimate.transformer_body_params > 3_225_088
    assert estimate.total_params == (
        estimate.token_embedding_params
        + estimate.position_embedding_params
        + estimate.attention_params
        + estimate.mlp_params
        + estimate.layer_norm_params
        + estimate.lm_head_params
    )


def test_explicit_mlp_size_and_invalid_scaling_config_rejection() -> None:
    assert small_config().effective_mlp_hidden_size == 96
    with pytest.raises(ValueError, match="divisible"):
        small_config(n_head=3)
    with pytest.raises(ValueError, match="attention_implementation"):
        small_config(attention_implementation="unknown")
    with pytest.raises(ValueError, match="mlp_hidden_size"):
        small_config(mlp_hidden_size=0)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        small_config(gradient_checkpointing="yes")


def test_gradient_checkpointing_forward_backward_and_tied_identity() -> None:
    model = DarkMindV2ForCausalLM(small_config(gradient_checkpointing=True)).train()
    input_ids = torch.randint(8, 100, (2, 16))
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters() if parameter.grad is not None)
    assert model.embeddings_are_tied()


@pytest.mark.skipif(not hasattr(torch.nn.functional, "scaled_dot_product_attention"), reason="SDPA unavailable")
def test_sdpa_and_fallback_outputs_are_compatible() -> None:
    fallback = DarkMindV2ForCausalLM(small_config(attention_implementation="fallback")).eval()
    sdpa = DarkMindV2ForCausalLM(small_config(attention_implementation="sdpa")).eval()
    sdpa.load_state_dict(fallback.state_dict())
    input_ids = torch.randint(8, 100, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = fallback(input_ids, attention_mask=attention_mask).logits
        actual = sdpa(input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)


def test_scalable_save_reload_preserves_outputs_and_config(tmp_path) -> None:
    model = DarkMindV2ForCausalLM(
        small_config(attention_implementation="fallback", gradient_checkpointing=True)
    ).eval()
    input_ids = torch.randint(8, 100, (1, 8))
    expected = model(input_ids).logits
    output = tmp_path / "model"
    save_model_package(model, output)
    loaded = load_model_package(output)
    assert torch.equal(expected, loaded(input_ids).logits)
    assert loaded.config.gradient_checkpointing is True
    assert loaded.config.effective_mlp_hidden_size == 96


def test_bf16_safetensors_reload_preserves_stored_dtype(tmp_path) -> None:
    model = DarkMindV2ForCausalLM(small_config()).to(dtype=torch.bfloat16)
    output = tmp_path / "bf16-model"
    save_model_package(model, output)
    loaded = load_model_package(output)
    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert loaded.embeddings_are_tied()


def test_benchmark_schema_accepts_required_protocol_and_rejects_short_runs() -> None:
    profiles = [
        {
            "candidate": candidate,
            "micro_batch_size": 1,
            "attention_implementation": "sdpa",
        }
        for candidate in CANDIDATES
    ]
    payload = {
        "schema_version": "darkmind-v2-phase3a-rtx4060-benchmark-v1",
        "protocol": {"warmup_microsteps": 10, "measured_microsteps": 30},
        "profiles": profiles,
    }
    validate_benchmark_report(payload)
    payload = json.loads(json.dumps(payload))
    payload["protocol"]["warmup_microsteps"] = 9
    with pytest.raises(ValueError, match="warmup"):
        validate_benchmark_report(payload)


def test_tiny_model_remains_exactly_backward_compatible() -> None:
    config = DarkMindV2Config.from_json_file("darkmind_v2/config/model_tiny_smoke.json")
    model = DarkMindV2ForCausalLM(config)
    assert model.parameter_count() == 9_369_088
    assert config.effective_mlp_hidden_size == 1024
    assert config.attention_implementation == "auto"
    assert config.gradient_checkpointing is False
    assert model_config_hash(config) == "61ebdadb15d4fde09db9842413450f512a12fc7f7e6fcaead7dc1347478f4414"
