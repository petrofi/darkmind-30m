"""Create a local fixture-only Hugging Face package; never uploads."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import DEFAULT_FROZEN_DIR, EXPECTED_HASHES


REQUIRED_EXPORT_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "README.md",
    "training_evaluation_metadata.json",
    "provenance_manifest.json",
    "configuration_darkmind_v2.py",
    "modeling_darkmind_v2.py",
    "tokenization_darkmind_v2.py",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_fixture_package(
    model: DarkMindV2ForCausalLM,
    output_dir: Path,
    *,
    tokenizer_dir: Path = DEFAULT_FROZEN_DIR,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite export directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.auto_map = {
        "AutoConfig": "configuration_darkmind_v2.DarkMindV2Config",
        "AutoModelForCausalLM": "modeling_darkmind_v2.DarkMindV2ForCausalLM",
    }
    model.config.architectures = ["DarkMindV2ForCausalLM"]
    model.save_pretrained(output_dir, safe_serialization=True)

    source_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(source_root / "modeling" / "configuration_darkmind_v2.py", output_dir / "configuration_darkmind_v2.py")
    shutil.copyfile(source_root / "modeling" / "modeling_darkmind_v2.py", output_dir / "modeling_darkmind_v2.py")
    shutil.copyfile(Path(__file__).with_name("tokenization_darkmind_v2.py"), output_dir / "tokenization_darkmind_v2.py")
    shutil.copyfile(tokenizer_dir / "tokenizer.model", output_dir / "tokenizer.model")
    shutil.copyfile(tokenizer_dir / "tokenizer.vocab", output_dir / "tokenizer.vocab")

    tokenizer_config = {
        "auto_map": {"AutoTokenizer": ["tokenization_darkmind_v2.DarkMindV2Tokenizer", None]},
        "tokenizer_class": "DarkMindV2Tokenizer",
        "model_max_length": model.config.block_size,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": True,
        "add_eos_token": True,
    }
    special_tokens_map = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    }
    generation_config = {
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "do_sample": False,
        "max_new_tokens": 64,
        "transformers_version": "recorded_at_export",
    }
    atomic_write_json(output_dir / "tokenizer_config.json", tokenizer_config)
    atomic_write_json(output_dir / "special_tokens_map.json", special_tokens_map)
    atomic_write_json(output_dir / "generation_config.json", generation_config)
    release_metadata = {
        "schema_version": "darkmind-v2-hf-metadata-v1",
        "fixture_only": True,
        "trained_model": False,
        "instruction_tuned": False,
        "upload_performed": False,
        **(metadata or {}),
    }
    atomic_write_json(output_dir / "training_evaluation_metadata.json", release_metadata)
    (output_dir / "README.md").write_text(
        "# DarkMind v2 Tiny Base Fixture Export\n\n"
        "This local package contains untrained fixture weights used only to validate the export pipeline. "
        "It is not a released model, not instruction-tuned, and not a conversational assistant.\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "darkmind-v2-export-provenance-v1",
        "tokenizer_model_sha256": EXPECTED_HASHES["tokenizer.model"],
        "tokenizer_vocab_sha256": EXPECTED_HASHES["tokenizer.vocab"],
        "files": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "provenance_manifest.json"
        },
    }
    atomic_write_json(output_dir / "provenance_manifest.json", provenance)
    missing = sorted(REQUIRED_EXPORT_FILES - {path.name for path in output_dir.iterdir()})
    if missing:
        raise RuntimeError(f"incomplete fixture export: {missing}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fixture export only; no upload support.")
    parser.add_argument("model_package", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    from darkmind_v2.modeling.model_io import load_model_package

    model = load_model_package(args.model_package)
    result = export_fixture_package(model, args.output_dir)
    print(json.dumps({"result": "PASS", "files": sorted(result["files"])}, indent=2))


if __name__ == "__main__":
    main()
