"""Create and offline-validate the local-only Base V1 Stage-1 export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import DEFAULT_FROZEN_DIR, EXPECTED_HASHES
from darkmind_v2.training.validate_phase4a_config import DEFAULT_CONFIG, load_and_validate_phase4a_config
from darkmind_v2.training.validate_phase4a_preflight import ROOT


REQUIRED_FILES = {
    "README.md",
    "config.json",
    "configuration_darkmind_v2.py",
    "evaluation_results.json",
    "generation_config.json",
    "model.safetensors",
    "modeling_darkmind_v2.py",
    "provenance_manifest.json",
    "special_tokens_map.json",
    "tokenization_darkmind_v2.py",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "training_metrics.jsonl",
}


def export_stage1(config_path: Path) -> dict[str, Any]:
    config = load_and_validate_phase4a_config(config_path, check_runtime_assets=True)
    run_dir = ROOT / config["run_dir"]
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    if run.get("status") != "evaluated" or summary.get("result") != "PASS":
        raise ValueError("Stage-1 integrity/evaluation gates did not pass")
    checkpoint = ROOT / summary["best_checkpoint"]
    output_dir = ROOT / config["export_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite local export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_package(checkpoint / "model", device="cpu")
    model.config.auto_map = {
        "AutoConfig": "configuration_darkmind_v2.DarkMindV2Config",
        "AutoModelForCausalLM": "modeling_darkmind_v2.DarkMindV2ForCausalLM",
    }
    model.config.architectures = ["DarkMindV2ForCausalLM"]
    model.save_pretrained(output_dir, safe_serialization=True)
    source_root = ROOT / "darkmind_v2"
    shutil.copyfile(source_root / "modeling" / "configuration_darkmind_v2.py", output_dir / "configuration_darkmind_v2.py")
    shutil.copyfile(source_root / "modeling" / "modeling_darkmind_v2.py", output_dir / "modeling_darkmind_v2.py")
    shutil.copyfile(source_root / "export" / "tokenization_darkmind_v2.py", output_dir / "tokenization_darkmind_v2.py")
    shutil.copyfile(DEFAULT_FROZEN_DIR / "tokenizer.model", output_dir / "tokenizer.model")
    shutil.copyfile(DEFAULT_FROZEN_DIR / "tokenizer.vocab", output_dir / "tokenizer.vocab")
    shutil.copyfile(run_dir / "metrics.jsonl", output_dir / "training_metrics.jsonl")
    shutil.copyfile(run_dir / "evaluation_summary.json", output_dir / "evaluation_results.json")
    atomic_write_json(
        output_dir / "tokenizer_config.json",
        {
            "auto_map": {"AutoTokenizer": ["tokenization_darkmind_v2.DarkMindV2Tokenizer", None]},
            "tokenizer_class": "DarkMindV2Tokenizer",
            "model_max_length": 512,
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
        {"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0, "do_sample": False, "max_new_tokens": 32},
    )
    (output_dir / "README.md").write_text(
        "# DarkMind v2 Base V1 Stage-1 5M\n\n"
        "This is a local-only from-scratch Base V1 Stage-1 checkpoint with 118,056,960 parameters. "
        "It consumed exactly 4,997,120 training tokens from the validated Corpus V3 aggregate 100M dataset.\n\n"
        "The model is not instruction-tuned, not a chat model, not production-ready, not publicly released, "
        "and is expected to produce incoherent, repetitive, or factually unreliable text at this early stage. "
        "The model-weight license is unresolved; redistribution and public upload are not authorized.\n",
        encoding="utf-8",
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "provenance_manifest.json"
    }
    provenance = {
        "schema_version": "darkmind-v2-base-v1-stage1-export-v1",
        "local_only": True,
        "upload_performed": False,
        "checkpoint": summary["best_checkpoint"],
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "parameters": 118_056_960,
        "training_tokens": 4_997_120,
        "optimizer_steps": 610,
        "corpus_hash": config["corpus"]["corpus_hash"],
        "tokenized_manifest_hash": config["corpus"]["tokenized_manifest_hash"],
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "instruction_tuned": False,
        "chat_model": False,
        "production_ready": False,
        "publicly_released": False,
        "model_weight_license": "unresolved",
        "files": files,
    }
    atomic_write_json(output_dir / "provenance_manifest.json", provenance)
    missing = REQUIRED_FILES - {path.name for path in output_dir.iterdir() if path.is_file()}
    if missing:
        raise ValueError(f"local export is incomplete: {sorted(missing)}")
    return {"output_dir": output_dir, "provenance": provenance}


def validate_export(output_dir: Path) -> dict[str, Any]:
    provenance = json.loads((output_dir / "provenance_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for filename, expected in provenance["files"].items():
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            failures.append(f"file integrity mismatch: {filename}")
    auto_config_loaded = False
    auto_model_loaded = False
    auto_tokenizer_loaded = False
    finite_forward = False
    greedy_generation = False
    seeded_generation = False
    safetensors_loaded = False
    if not failures:
        try:
            from safetensors import safe_open
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as weights:
                safetensors_loaded = bool(weights.keys())
            config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            auto_config_loaded = config.vocab_size == 24_000 and config.n_layer == 14
            auto_model_loaded = model.parameter_count() == 118_056_960
            auto_tokenizer_loaded = tokenizer.vocab_size == 24_000
            encoded = tokenizer("Turkiye ve bilim", return_tensors="pt")
            with torch.no_grad():
                output = model(**encoded)
            finite_forward = bool(torch.isfinite(output.logits).all())
            greedy = model.generate_tokens(encoded["input_ids"], max_new_tokens=8, do_sample=False, eos_token_id=3)
            sampled = model.generate_tokens(encoded["input_ids"], max_new_tokens=8, do_sample=True, temperature=0.8, top_p=0.9, top_k=40, seed=20260712, eos_token_id=3)
            greedy_generation = greedy.shape[1] > encoded["input_ids"].shape[1]
            seeded_generation = sampled.shape[1] > encoded["input_ids"].shape[1]
        except Exception as exc:
            failures.append(f"offline round-trip failed: {type(exc).__name__}: {exc}")
    checks = {
        "safetensors_loaded": safetensors_loaded,
        "auto_config_loaded": auto_config_loaded,
        "auto_model_loaded": auto_model_loaded,
        "auto_tokenizer_loaded": auto_tokenizer_loaded,
        "finite_forward": finite_forward,
        "greedy_generation": greedy_generation,
        "seeded_generation": seeded_generation,
    }
    if not all(checks.values()):
        failures.append(f"offline checks incomplete: {checks}")
    report = {
        "schema_version": "darkmind-v2-base-v1-stage1-export-validation-v1",
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        **checks,
        "local_files_only": True,
        "trust_remote_code": True,
        "upload_performed": False,
    }
    atomic_write_json(output_dir / "offline_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    exported = export_stage1(args.config)
    report = validate_export(exported["output_dir"])
    config = load_and_validate_phase4a_config(args.config, check_runtime_assets=True)
    run_dir = ROOT / config["run_dir"]
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run["local_export"] = {
        "path": config["export_dir"],
        "result": report["result"],
        "upload_performed": False,
    }
    atomic_write_json(run_dir / "run_manifest.json", run)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
