"""Create and offline-validate the local-only Base V1 Stage-2 25M export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES
from darkmind_v2.training.phase4c_diagnostics import TOKENIZER_INPUT
from darkmind_v2.training.phase4d_stage2 import (
    AUTHORIZATION_PATH,
    RUN_DIR,
    RUNTIME_ROOT,
    TARGET_STEP,
    TARGET_TOKENS,
    V2_CONFIG,
    atomic_write_json,
    directory_size,
    load_json,
)
from darkmind_v2.training.validate_phase4a_preflight import EXPECTED_CORPUS_HASH, EXPECTED_TOKENIZED_HASH, ROOT


OUTPUT_DIR = RUNTIME_ROOT / "exports" / "darkmind-v2-base-v1-stage2-25m"
REQUIRED_FILES = {
    "README.md",
    "config.json",
    "configuration_darkmind_v2.py",
    "evaluation_results.json",
    "generation_audit.json",
    "generation_config.json",
    "hashes.json",
    "model.safetensors",
    "modeling_darkmind_v2.py",
    "provenance_manifest.json",
    "special_tokens_map.json",
    "stage2_authorization.json",
    "tokenization_darkmind_v2.py",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "training_config_v2.json",
    "training_metrics.jsonl",
    "training_summary.json",
}


def _write_model_card(output_dir: Path, training: dict[str, Any], generation: dict[str, Any]) -> None:
    final = generation["authoritative_final"]
    greedy_repetition = final["greedy"]["quality_warning_counts"].get("repetition", 0) / final["greedy"]["generations"]
    greedy_loops = final["greedy"]["exact_repeated_ngram_loop_outputs"] / final["greedy"]["generations"]
    sampling_repetition = final["sampling"]["quality_warning_counts"].get("repetition", 0) / final["sampling"]["generations"]
    sampling_loops = final["sampling"]["exact_repeated_ngram_loop_outputs"] / final["sampling"]["generations"]
    text = f"""# DarkMind v2 Base V1 Stage-2 25M

This is a local-only research artifact for a from-scratch DarkMind v2 Base V1 causal language model with 118,056,960 parameters.

## Training identity

- Exact training tokens: {training['final_tokens']:,}
- Exact optimizer steps: {training['final_optimizer_step']:,}
- Exact checkpoint: `{training['best_checkpoint']}`
- Corpus: validated Corpus V3 aggregate, deterministic no-replacement order
- Tokenizer: frozen DarkMind v2 SentencePiece BPE 24k v1
- Stage-2 learning classification: {training['classification']}

## Intended status

This model is not instruction-tuned, not a chat model, not production-ready, and not publicly released. It is not approved for public deployment or factual reliance. The model-weight license is unresolved, so redistribution and public upload are not authorized.

## Known limitations

The checkpoint remains an early base model. Outputs can be incoherent, repetitive, empty, very short, structurally weak, or factually wrong. Turkish technical loss improved more slowly than the other primary probes.

The final 200/500 diagnostic audit measured greedy repetition warnings at {greedy_repetition:.1%} and exact loops at {greedy_loops:.1%}; seeded sampling repetition warnings at {sampling_repetition:.1%} and exact loops at {sampling_loops:.1%}. Generation quality remains diagnostic and does not imply conversational or release quality.

## Safety and provenance

No SFT or preference tuning was performed. No Qwen teacher data was generated. No runtime optimizer state, scheduler state, RNG state, corpus shard, or credential is included. See `provenance_manifest.json`, `training_summary.json`, `evaluation_results.json`, and `generation_audit.json` for exact local evidence.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def export_stage2() -> dict[str, Any]:
    training = load_json(RUN_DIR / "training_summary.json")
    generation = load_json(RUN_DIR / "generation_analysis.json")
    if training.get("classification") != "STRONG PASS" or not training.get("integrity_pass"):
        raise ValueError("Phase 4D learning/integrity gates did not pass")
    if generation.get("result") != "PASS" or generation["authoritative_final"].get("result") != "PASS":
        raise ValueError("Phase 4D generation integrity gate did not pass")
    if training["best_validation_step"] != TARGET_STEP or training["final_tokens"] != TARGET_TOKENS:
        raise ValueError("Phase 4D best checkpoint or exact token gate changed")
    checkpoint = Path(training["best_checkpoint"])
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        existing = OUTPUT_DIR / "offline_validation.json"
        if existing.is_file() and load_json(existing).get("result") == "PASS":
            return {"output_dir": OUTPUT_DIR, "reused": True}
        raise FileExistsError(f"refusing to overwrite local Phase 4D export: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model_package(checkpoint / "model", device="cpu")
    model.config.auto_map = {
        "AutoConfig": "configuration_darkmind_v2.DarkMindV2Config",
        "AutoModelForCausalLM": "modeling_darkmind_v2.DarkMindV2ForCausalLM",
    }
    model.config.architectures = ["DarkMindV2ForCausalLM"]
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    source_root = ROOT / "darkmind_v2"
    shutil.copyfile(source_root / "modeling" / "configuration_darkmind_v2.py", OUTPUT_DIR / "configuration_darkmind_v2.py")
    shutil.copyfile(source_root / "modeling" / "modeling_darkmind_v2.py", OUTPUT_DIR / "modeling_darkmind_v2.py")
    shutil.copyfile(source_root / "export" / "tokenization_darkmind_v2.py", OUTPUT_DIR / "tokenization_darkmind_v2.py")
    shutil.copyfile(TOKENIZER_INPUT / "tokenizer.model", OUTPUT_DIR / "tokenizer.model")
    shutil.copyfile(TOKENIZER_INPUT / "tokenizer.vocab", OUTPUT_DIR / "tokenizer.vocab")
    shutil.copyfile(RUN_DIR / "metrics.jsonl", OUTPUT_DIR / "training_metrics.jsonl")
    shutil.copyfile(RUN_DIR / "training_summary.json", OUTPUT_DIR / "training_summary.json")
    shutil.copyfile(RUN_DIR / "generation_analysis.json", OUTPUT_DIR / "generation_audit.json")
    shutil.copyfile(AUTHORIZATION_PATH, OUTPUT_DIR / "stage2_authorization.json")
    shutil.copyfile(V2_CONFIG, OUTPUT_DIR / "training_config_v2.json")
    atomic_write_json(
        OUTPUT_DIR / "evaluation_results.json",
        {
            "schema_version": "darkmind-v2-phase4d-export-evaluation-v1",
            "classification": training["classification"],
            "validation_progression": training["validation_progression"],
            "eval_progression": training["eval_progression"],
            "probe_regressions": training["probe_regressions"],
            "process_restart": training["process_restart"],
            "authoritative_generation_summary": generation["authoritative_final"],
        },
    )
    atomic_write_json(
        OUTPUT_DIR / "tokenizer_config.json",
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
        OUTPUT_DIR / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        },
    )
    atomic_write_json(
        OUTPUT_DIR / "generation_config.json",
        {
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 0,
            "do_sample": False,
            "max_new_tokens": 32,
            "warning": "Base model generation is diagnostic; this is not a chat preset.",
        },
    )
    _write_model_card(OUTPUT_DIR, training, generation)
    file_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file() and path.name not in {"hashes.json", "provenance_manifest.json", "offline_validation.json"}
    }
    atomic_write_json(
        OUTPUT_DIR / "hashes.json",
        {"schema_version": "darkmind-v2-phase4d-export-hashes-v1", "files": file_hashes},
    )
    provenance_files = {
        **file_hashes,
        "hashes.json": {
            "bytes": (OUTPUT_DIR / "hashes.json").stat().st_size,
            "sha256": sha256_file(OUTPUT_DIR / "hashes.json"),
        },
    }
    provenance = {
        "schema_version": "darkmind-v2-base-v1-stage2-25m-export-v1",
        "local_only": True,
        "upload_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "parameters": 118_056_960,
        "training_tokens": TARGET_TOKENS,
        "optimizer_steps": TARGET_STEP,
        "architecture": "from-scratch DarkMind v2 Base V1",
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "v2_config_sha256": training["v2_config_sha256"],
        "stage2_authorization_sha256": training["authorization_file_sha256"],
        "instruction_tuned": False,
        "chat_model": False,
        "production_ready": False,
        "publicly_released": False,
        "model_weight_license": "unresolved",
        "known_limitations": ["repetition", "incoherence", "factual unreliability", "weak technical generation"],
        "optimizer_state_included": False,
        "runtime_checkpoint_included": False,
        "files": provenance_files,
    }
    atomic_write_json(OUTPUT_DIR / "provenance_manifest.json", provenance)
    missing = REQUIRED_FILES - {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if missing:
        raise ValueError(f"local Phase 4D export is incomplete: {sorted(missing)}")
    forbidden = [path.name for path in OUTPUT_DIR.iterdir() if path.suffix in {".bin", ".pt", ".pth"}]
    if forbidden:
        raise ValueError(f"local Phase 4D export contains forbidden state: {forbidden}")
    del model
    return {"output_dir": OUTPUT_DIR, "provenance": provenance, "reused": False}


def validate_export(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    provenance = load_json(output_dir / "provenance_manifest.json")
    failures = []
    for filename, expected in provenance["files"].items():
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            failures.append(f"file integrity mismatch: {filename}")
    checks = {
        "safetensors_only_weights": not any(path.suffix in {".bin", ".pt", ".pth"} for path in output_dir.iterdir()),
        "safetensors_loaded": False,
        "auto_config_loaded": False,
        "auto_model_loaded": False,
        "auto_tokenizer_loaded": False,
        "finite_forward": False,
        "greedy_generation": False,
        "seeded_generation": False,
        "exact_training_tokens": provenance.get("training_tokens") == TARGET_TOKENS,
        "exact_optimizer_steps": provenance.get("optimizer_steps") == TARGET_STEP,
        "required_disclosures": False,
    }
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    disclosures = (
        "not instruction-tuned",
        "not a chat model",
        "not production-ready",
        "not publicly released",
        "model-weight license is unresolved",
    )
    checks["required_disclosures"] = all(text in readme for text in disclosures)
    if not failures:
        try:
            from safetensors import safe_open
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as weights:
                checks["safetensors_loaded"] = bool(weights.keys())
            config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
            checks["auto_config_loaded"] = config.vocab_size == 24_000 and config.n_layer == 14
            checks["auto_model_loaded"] = model.parameter_count() == 118_056_960
            checks["auto_tokenizer_loaded"] = tokenizer.vocab_size == 24_000
            encoded = tokenizer("Turkiye ve bilim", return_tensors="pt")
            with torch.no_grad():
                output = model(**encoded)
            checks["finite_forward"] = bool(torch.isfinite(output.logits).all())
            greedy = model.generate_tokens(encoded["input_ids"], max_new_tokens=8, do_sample=False, eos_token_id=3)
            seeded = model.generate_tokens(
                encoded["input_ids"],
                max_new_tokens=8,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                seed=20260712,
                eos_token_id=3,
            )
            checks["greedy_generation"] = greedy.shape[1] > encoded["input_ids"].shape[1]
            checks["seeded_generation"] = seeded.shape[1] > encoded["input_ids"].shape[1]
        except Exception as exc:
            failures.append(f"offline round-trip failed: {type(exc).__name__}: {exc}")
    if not all(checks.values()):
        failures.append(f"offline checks incomplete: {checks}")
    report = {
        "schema_version": "darkmind-v2-base-v1-stage2-25m-export-validation-v1",
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        **checks,
        "local_files_only": True,
        "trust_remote_code": True,
        "upload_performed": False,
        "export_bytes": directory_size(output_dir),
    }
    atomic_write_json(output_dir / "offline_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "validate", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if args.command == "export":
        payload = export_stage2()
        print(json.dumps({"output_dir": str(payload["output_dir"]), "reused": payload["reused"]}, indent=2))
        return
    if args.command == "validate":
        report = validate_export()
    else:
        exported = export_stage2()
        report = validate_export(exported["output_dir"])
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
