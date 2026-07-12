from pathlib import Path

from darkmind_v2.evaluation.validate_generation_health import validate_text_health
from darkmind_v2.export.export_huggingface import REQUIRED_EXPORT_FILES, export_fixture_package
from darkmind_v2.export.validate_huggingface_export import validate_export
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.training.train_tiny_smoke import run_fixture_smoke


def small_config() -> DarkMindV2Config:
    return DarkMindV2Config(
        vocab_size=24000,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        mlp_ratio=4,
        dropout=0.0,
        bias=True,
        seed=321,
    )


def test_fixture_overfit_checkpoint_reload_resume_and_generation(tmp_path) -> None:
    result = run_fixture_smoke(
        model_config=small_config(),
        checkpoint_dir=tmp_path / "checkpoint",
        steps=8,
        sequence_length=16,
        learning_rate=0.01,
        mixed_precision="fp32",
        device_name="cpu",
        data_manifest_hash="test-fixture-manifest",
    )
    assert result.loss_decreased
    assert result.final_loss < result.initial_loss
    assert result.checkpoint_reloaded
    assert result.resumed_step == 8
    assert result.resume_continued_to_step == 9
    assert result.generation_token_range_valid


def test_generation_health_checks() -> None:
    healthy = validate_text_health("Readable output", [10, 11, 12, 13], maximum_repetition_ratio=0.5)
    assert healthy["result"] == "PASS"
    replacement = validate_text_health("broken \ufffd text", [9, 9, 9], maximum_repetition_ratio=0.5)
    assert "decoder_defect" in replacement["failures"]
    assert "repetition" in replacement["failures"]


def test_huggingface_fixture_export_structure(tmp_path) -> None:
    output = tmp_path / "hf-export"
    export_fixture_package(DarkMindV2ForCausalLM(small_config()), output)
    assert REQUIRED_EXPORT_FILES <= {path.name for path in output.iterdir()}
    report = validate_export(output, offline_roundtrip=False)
    assert report["result"] == "PASS"
    assert (output / "model.safetensors").is_file()
    assert not (output / "pytorch_model.bin").exists()
