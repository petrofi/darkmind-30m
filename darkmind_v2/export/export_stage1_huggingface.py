"""Export the accepted Stage-1 checkpoint as a local Hugging Face research package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash, sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import DEFAULT_FROZEN_DIR, EXPECTED_HASHES, verify_frozen_tokenizer


REQUIRED_STAGE1_EXPORT_FILES = {
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
    "corpus_attribution.json",
    "LICENSE_INFORMATION.md",
    "provenance_manifest.json",
    "file_hashes.json",
}


def export_stage1(
    config_path: Path,
    evaluation_path: Path,
    integrity_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite local export: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("result") != "PASS":
        raise RuntimeError("Stage-1 artifact integrity did not pass")
    failed_gates = [key for key, value in evaluation["acceptance_gates"].items() if value is False]
    if failed_gates:
        raise RuntimeError(f"Stage-1 evaluation gates failed: {failed_gates}")
    if not evaluation["acceptance_gates"]["generation_hard_failures_zero"]:
        raise RuntimeError("generation hard failures block export")
    best = evaluation["best_checkpoint"]
    checkpoint = Path(best["checkpoint"])
    model = load_model_package(checkpoint / "model", device="cpu")
    if model.num_parameters() != 9_369_088:
        raise RuntimeError("unexpected Stage-1 parameter count")
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
    shutil.copyfile(Path(__file__).with_name("MODEL_CARD_STAGE1.md"), output_dir / "README.md")
    shutil.copyfile(Path(__file__).with_name("LICENSE_INFORMATION.md"), output_dir / "LICENSE_INFORMATION.md")
    shutil.copyfile(
        source_root / "config" / "corpus_attribution_summary.json",
        output_dir / "corpus_attribution.json",
    )

    atomic_write_json(
        output_dir / "tokenizer_config.json",
        {
            "auto_map": {"AutoTokenizer": ["tokenization_darkmind_v2.DarkMindV2Tokenizer", None]},
            "tokenizer_class": "DarkMindV2Tokenizer",
            "model_max_length": model.config.block_size,
            "clean_up_tokenization_spaces": False,
            "add_bos_token": True,
            "add_eos_token": True,
        },
    )
    atomic_write_json(
        output_dir / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        },
    )
    atomic_write_json(
        output_dir / "generation_config.json",
        {"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0, "do_sample": False, "max_new_tokens": 64},
    )

    training_metrics = {
        "schema_version": "darkmind-v2-stage1-training-metrics-v1",
        "optimizer_steps": evaluation["summary"]["optimizer_steps"],
        "consumed_tokens": evaluation["summary"]["consumed_tokens"],
        "initial_training_loss": evaluation["summary"]["initial_training_loss"],
        "midpoint_training_loss": evaluation["summary"]["midpoint_training_loss"],
        "final_training_loss": evaluation["summary"]["final_training_loss"],
        "initial_validation_loss": evaluation["summary"]["initial_validation_loss"],
        "midpoint_validation_loss": evaluation["summary"]["midpoint_validation_loss"],
        "final_validation_loss": evaluation["summary"]["final_validation_loss"],
        "throughput": integrity["throughput"],
    }
    atomic_write_json(output_dir / "training_metrics.json", training_metrics)
    atomic_write_json(output_dir / "evaluation_results.json", evaluation)

    provenance_core = {
        "schema_version": "darkmind-v2-stage1-export-provenance-v1",
        "research_checkpoint": True,
        "upload_performed": False,
        "instruction_tuned": False,
        "conversational_assistant": False,
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "evaluation_sha256": sha256_file(evaluation_path),
        "integrity_report_sha256": sha256_file(integrity_path),
        "selected_checkpoint": best,
        "selected_checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "tokenizer_hashes": EXPECTED_HASHES,
        "tokenized_corpus_manifest_sha256": run_manifest["tokenized_corpus_manifest_sha256"],
        "tokenized_corpus_content_hash": run_manifest["tokenized_corpus_content_hash"],
        "source_corpus_manifest_sha256": run_manifest["source_corpus_manifest_sha256"],
    }
    provenance = {**provenance_core, "deterministic_content_hash": canonical_json_hash(provenance_core)}
    atomic_write_json(output_dir / "provenance_manifest.json", provenance)
    file_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "file_hashes.json"
    }
    atomic_write_json(output_dir / "file_hashes.json", file_hashes)
    missing = sorted(REQUIRED_STAGE1_EXPORT_FILES - {path.name for path in output_dir.iterdir()})
    if missing:
        raise RuntimeError(f"incomplete Stage-1 export: {missing}")
    return {
        "result": "PASS",
        "output_dir": str(output_dir),
        "selected_checkpoint": best,
        "files": sorted(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Stage-1 export only; no upload support.")
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1_r2.json"))
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/byte_trace_policy_v2/stage1_evaluation.json"
        ),
    )
    parser.add_argument(
        "--integrity",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/byte_trace_policy_v1/artifact_integrity.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("darkmind_v2/data/phase2a/exports/darkmind-v2-tiny-stage1"),
    )
    args = parser.parse_args()
    print(json.dumps(export_stage1(args.config, args.evaluation, args.integrity, args.output), indent=2))


if __name__ == "__main__":
    main()
