"""Create a local-only Hugging Face export for the Phase 2C best checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash, sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import DEFAULT_FROZEN_DIR, EXPECTED_HASHES, verify_frozen_tokenizer


REQUIRED_FULL_EPOCH_EXPORT_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "configuration_darkmind_v2.py",
    "modeling_darkmind_v2.py",
    "tokenization_darkmind_v2.py",
    "README.md",
    "training_metrics.json",
    "evaluation_results.json",
    "public_audit.json",
    "corpus_attribution.json",
    "LICENSE_INFORMATION.md",
    "provenance_manifest.json",
    "file_hashes.json",
}


def export_full_epoch(config_path: Path, summary_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite local export: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "full_epoch_complete":
        raise RuntimeError("Phase 2C run is not complete")
    if summary["diagnosis"]["public_release_eligible"]:
        raise RuntimeError("unexpected public-release eligibility state")
    if summary["full_epoch"]["public_hard_failures"] != 0:
        raise RuntimeError("public-preview hard failures block local export")
    checkpoint = Path(summary["best_checkpoint"]["path"])
    actual_model_hash = sha256_file(checkpoint / "model" / "model.safetensors")
    if actual_model_hash != summary["best_checkpoint"]["model_sha256"]:
        raise RuntimeError("best checkpoint hash mismatch")
    model = load_model_package(checkpoint / "model", device="cpu")
    if model.num_parameters() != 9_369_088 or not model.embeddings_are_tied():
        raise RuntimeError("unexpected Phase 2C model architecture")
    verify_frozen_tokenizer()

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
    shutil.copyfile(DEFAULT_FROZEN_DIR / "tokenizer.model", output_dir / "tokenizer.model")
    shutil.copyfile(DEFAULT_FROZEN_DIR / "tokenizer.vocab", output_dir / "tokenizer.vocab")
    shutil.copyfile(Path(__file__).with_name("MODEL_CARD_PHASE2C.md"), output_dir / "README.md")
    shutil.copyfile(Path(__file__).with_name("LICENSE_INFORMATION.md"), output_dir / "LICENSE_INFORMATION.md")
    shutil.copyfile(source_root / "config" / "corpus_attribution_summary.json", output_dir / "corpus_attribution.json")

    atomic_write_json(output_dir / "tokenizer_config.json", {
        "auto_map": {"AutoTokenizer": ["tokenization_darkmind_v2.DarkMindV2Tokenizer", None]},
        "tokenizer_class": "DarkMindV2Tokenizer",
        "model_max_length": model.config.block_size,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": True,
        "add_eos_token": True,
    })
    atomic_write_json(output_dir / "special_tokens_map.json", {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    })
    atomic_write_json(output_dir / "generation_config.json", {
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "do_sample": False,
        "max_new_tokens": 64,
    })
    atomic_write_json(output_dir / "training_metrics.json", {
        "schema_version": "darkmind-v2-phase2c-training-metrics-v1",
        "optimizer_steps": summary["full_epoch"]["optimizer_steps"],
        "consumed_tokens": summary["full_epoch"]["trained_tokens"],
        "coverage_percent": summary["full_epoch"]["coverage_percent"],
        "excluded_tail_tokens": summary["full_epoch"]["excluded_tail_tokens"],
        "final_training_loss": summary["full_epoch"]["final_train_loss"],
        "final_smoothed_training_loss": summary["full_epoch"]["final_smoothed_train_loss"],
        "final_validation_loss": summary["full_epoch"]["validation_loss"],
        "final_eval_loss": summary["full_epoch"]["eval_loss"],
        "throughput": summary["throughput"],
    })
    atomic_write_json(output_dir / "evaluation_results.json", summary)
    final_audit_path = run_dir / "evaluations" / "public_preview_v2_step_002867" / "audit_summary.json"
    atomic_write_json(output_dir / "public_audit.json", json.loads(final_audit_path.read_text(encoding="utf-8")))

    provenance_core = {
        "schema_version": "darkmind-v2-phase2c-export-provenance-v1",
        "local_only": True,
        "upload_performed": False,
        "upload_authorized": False,
        "instruction_tuned": False,
        "chat_model": False,
        "production_ready": False,
        "public_release_eligible": False,
        "model_weight_license_resolved": False,
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "summary_sha256": sha256_file(summary_path),
        "public_audit_sha256": sha256_file(final_audit_path),
        "selected_checkpoint": summary["best_checkpoint"],
        "selected_checkpoint_model_sha256": actual_model_hash,
        "tokenizer_hashes": EXPECTED_HASHES,
        "tokenized_corpus_manifest_sha256": run_manifest["tokenized_corpus_manifest_sha256"],
        "tokenized_corpus_content_hash": run_manifest["tokenized_corpus_content_hash"],
        "source_corpus_manifest_sha256": run_manifest["source_corpus_manifest_sha256"],
    }
    atomic_write_json(
        output_dir / "provenance_manifest.json",
        {**provenance_core, "deterministic_content_hash": canonical_json_hash(provenance_core)},
    )
    file_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "file_hashes.json"
    }
    atomic_write_json(output_dir / "file_hashes.json", file_hashes)
    missing = sorted(REQUIRED_FULL_EPOCH_EXPORT_FILES - {path.name for path in output_dir.iterdir()})
    if missing:
        raise RuntimeError(f"incomplete Phase 2C export: {missing}")
    return {
        "result": "PASS",
        "output_dir": str(output_dir),
        "selected_checkpoint": summary["best_checkpoint"],
        "files": sorted(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Phase 2C export only; no upload support.")
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_full_epoch.json"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/runs/tiny_full_epoch_seed20260712_v1/evaluations/full_epoch_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/exports/darkmind-v2-tiny-full-epoch"),
    )
    args = parser.parse_args()
    print(json.dumps(export_full_epoch(args.config, args.summary, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
