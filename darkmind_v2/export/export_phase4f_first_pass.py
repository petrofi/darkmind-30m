"""Create and offline-validate the local-only Phase 4F first-pass export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import sha256_file
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES
from darkmind_v2.training.phase4f_completion import (
    AUTHORIZATION_PATH,
    FINAL_STEP,
    FINAL_TOKENS,
    ROOT,
    RUN_DIR,
    RUNTIME_ROOT,
    TOKENIZER_INPUT,
    V2_CONFIG,
    atomic_write_json,
    directory_size,
    load_json,
)
from darkmind_v2.training.validate_phase4a_preflight import EXPECTED_CORPUS_HASH, EXPECTED_TOKENIZED_HASH


OUTPUT_DIR = RUNTIME_ROOT / "exports" / "darkmind-v2-base-v1-first-pass-98m"
FORBIDDEN_EXPORT_PARTS = {
    "corpus_v3_tokenized",
    "document_boundaries.jsonl",
    "attribution_manifest.jsonl",
    "split_manifest.jsonl",
    "shard_checksums.json",
    "resume_state.pt",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
}
REQUIRED_FILES = {
    "README.md",
    "config.json",
    "configuration_darkmind_v2.py",
    "evaluation_results.json",
    "generation_audit.json",
    "generation_config.json",
    "hashes.json",
    "memorization_audit.json",
    "model.safetensors",
    "modeling_darkmind_v2.py",
    "provenance_manifest.json",
    "special_tokens_map.json",
    "tokenization_darkmind_v2.py",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "training_authorization.json",
    "training_config_v2.json",
    "training_summary.json",
}


def validate_export_file_list(paths: Iterable[str | Path]) -> None:
    for value in paths:
        path = Path(value)
        lowered = {part.lower() for part in path.parts}
        if lowered & FORBIDDEN_EXPORT_PARTS or path.suffix.lower() in {".bin", ".pt", ".pth"}:
            raise ValueError(f"forbidden corpus/runtime file in Phase 4F export: {path}")


def require_export_preconditions(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    progress = load_json(run_dir / "progress.json")
    if int(progress["optimizer_step"]) != FINAL_STEP:
        raise PermissionError("Phase 4F local export requires the final no-wrap checkpoint")
    summary = load_json(run_dir / "final_assessment.json")
    if summary.get("classification") == "FAIL" or not summary.get("integrity_pass"):
        raise PermissionError("Phase 4F learning/integrity gates do not permit export")
    memorization = load_json(run_dir / "memorization_audit.json")
    if memorization.get("result") != "PASS" or memorization.get("hard_release_blockers"):
        raise PermissionError("Phase 4F memorization/PII hard blockers are not cleared")
    generation = load_json(run_dir / "generation_analysis.json")
    if generation.get("hard_failure_total") != 0:
        raise PermissionError("Phase 4F generation hard failures block export")
    return summary


def _write_model_card(output_dir: Path, training: dict[str, Any], generation: dict[str, Any], memorization: dict[str, Any]) -> None:
    final = generation["final"]
    text = f"""# DarkMind v2 Base V1 First Corpus V3 Pass

This is a local-only research artifact for a from-scratch DarkMind v2 Base V1 causal language model with 118,056,960 parameters.

## Training identity

- Exact consumed training tokens: {training['final_tokens']:,}
- Exact optimizer steps: {training['final_optimizer_step']:,}
- Deterministic no-wrap stop: 191,552 complete sequences consumed; 14 incomplete-batch tail sequences excluded
- Selected best-validation checkpoint: `{training['best_checkpoint']}`
- Corpus: validated Corpus V3, first deterministic no-replacement pass
- Tokenizer: frozen DarkMind v2 SentencePiece BPE 24k v1
- First-pass classification: {training['classification']}

## Intended status

This model is not instruction-tuned, not a chat model, not production-ready, and not publicly uploaded. It is not approved for factual reliance or user-facing deployment. The model-weight license is unresolved, so redistribution and public upload are not authorized.

## Known limitations

Outputs can be incoherent, repetitive, empty, short, structurally weak, or factually wrong. The final audit measured greedy repetition at {final['greedy']['repetition_warning_rate']:.1%}, greedy exact loops at {final['greedy']['exact_loop_rate']:.1%}, seeded-sampling repetition at {final['sampling']['repetition_warning_rate']:.1%}, and seeded-sampling loops at {final['sampling']['exact_loop_rate']:.1%}.

The controlled memorization audit used {memorization['train_prefix_count']} train and {memorization['heldout_prefix_count']} held-out prefixes. It found {memorization['material_personal_data_reproduction_count']} material personal-data reproductions and {len(memorization['hard_release_blockers'])} hard blockers. Extraction risk is not claimed to be zero.

## Safety and provenance

No second epoch, SFT, preference tuning, or Qwen teacher-data generation was performed. No corpus shard, tokenized dataset, optimizer state, scheduler state, RNG state, runtime checkpoint, or credential is included. This package was validated offline and was not uploaded.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def export_first_pass() -> dict[str, Any]:
    training = require_export_preconditions()
    generation = load_json(RUN_DIR / "generation_analysis.json")
    memorization = load_json(RUN_DIR / "memorization_audit.json")
    checkpoint = Path(training["best_checkpoint"])
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        existing = OUTPUT_DIR / "offline_validation.json"
        if existing.is_file() and load_json(existing).get("result") == "PASS":
            return {"output_dir": OUTPUT_DIR, "reused": True}
        raise FileExistsError(f"refusing to overwrite local Phase 4F export: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model_package(checkpoint / "model", device="cpu")
    model.config.auto_map = {
        "AutoConfig": "configuration_darkmind_v2.DarkMindV2Config",
        "AutoModelForCausalLM": "modeling_darkmind_v2.DarkMindV2ForCausalLM",
    }
    model.config.architectures = ["DarkMindV2ForCausalLM"]
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    source_root = ROOT / "darkmind_v2"
    copies = {
        source_root / "modeling" / "configuration_darkmind_v2.py": OUTPUT_DIR / "configuration_darkmind_v2.py",
        source_root / "modeling" / "modeling_darkmind_v2.py": OUTPUT_DIR / "modeling_darkmind_v2.py",
        source_root / "export" / "tokenization_darkmind_v2.py": OUTPUT_DIR / "tokenization_darkmind_v2.py",
        TOKENIZER_INPUT / "tokenizer.model": OUTPUT_DIR / "tokenizer.model",
        TOKENIZER_INPUT / "tokenizer.vocab": OUTPUT_DIR / "tokenizer.vocab",
        RUN_DIR / "final_assessment.json": OUTPUT_DIR / "training_summary.json",
        RUN_DIR / "generation_analysis.json": OUTPUT_DIR / "generation_audit.json",
        RUN_DIR / "memorization_audit.json": OUTPUT_DIR / "memorization_audit.json",
        AUTHORIZATION_PATH: OUTPUT_DIR / "training_authorization.json",
        V2_CONFIG: OUTPUT_DIR / "training_config_v2.json",
    }
    for source, target in copies.items():
        shutil.copyfile(source, target)
    atomic_write_json(
        OUTPUT_DIR / "evaluation_results.json",
        {
            "schema_version": "darkmind-v2-phase4f-export-evaluation-v1",
            "classification": training["classification"],
            "validation_progression": training["validation_progression"],
            "eval_progression": training["eval_progression"],
            "probe_trends": training["probe_trends"],
            "final_eval_perplexity": training["final_eval_perplexity"],
            "integrity_pass": training["integrity_pass"],
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
            "warning": "Base-model generation is diagnostic; this is not a chat preset.",
        },
    )
    _write_model_card(OUTPUT_DIR, training, generation, memorization)
    validate_export_file_list(path.relative_to(OUTPUT_DIR) for path in OUTPUT_DIR.rglob("*") if path.is_file())
    file_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(OUTPUT_DIR.iterdir())
        if path.is_file() and path.name not in {"hashes.json", "provenance_manifest.json", "offline_validation.json"}
    }
    atomic_write_json(OUTPUT_DIR / "hashes.json", {"schema_version": "darkmind-v2-phase4f-export-hashes-v1", "files": file_hashes})
    provenance_files = {
        **file_hashes,
        "hashes.json": {"bytes": (OUTPUT_DIR / "hashes.json").stat().st_size, "sha256": sha256_file(OUTPUT_DIR / "hashes.json")},
    }
    provenance = {
        "schema_version": "darkmind-v2-base-v1-first-pass-98m-export-v1",
        "local_only": True,
        "upload_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        "parameters": 118_056_960,
        "training_tokens": FINAL_TOKENS,
        "optimizer_steps": FINAL_STEP,
        "consumed_sequences": 191_552,
        "excluded_tail_sequences": 14,
        "architecture": "from-scratch DarkMind v2 Base V1",
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "instruction_tuned": False,
        "chat_model": False,
        "production_ready": False,
        "publicly_uploaded": False,
        "model_weight_license": "unresolved",
        "known_limitations": ["repetition", "incoherence", "factual unreliability", "weak structured generation"],
        "memorization_audit_result": memorization["result"],
        "extraction_risk_zero_claimed": False,
        "optimizer_state_included": False,
        "runtime_checkpoint_included": False,
        "corpus_files_included": False,
        "files": provenance_files,
    }
    atomic_write_json(OUTPUT_DIR / "provenance_manifest.json", provenance)
    missing = REQUIRED_FILES - {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    if missing:
        raise ValueError(f"local Phase 4F export is incomplete: {sorted(missing)}")
    del model
    return {"output_dir": OUTPUT_DIR, "provenance": provenance, "reused": False}


def validate_export(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    provenance = load_json(output_dir / "provenance_manifest.json")
    failures = []
    for filename, expected in provenance["files"].items():
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            failures.append(f"file integrity mismatch: {filename}")
    all_files = [path for path in output_dir.rglob("*") if path.is_file()]
    checks = {
        "safetensors_only_weights": not any(path.suffix.lower() in {".bin", ".pt", ".pth"} for path in all_files),
        "corpus_files_excluded": not any(part.lower() in FORBIDDEN_EXPORT_PARTS for path in all_files for part in path.parts),
        "safetensors_loaded": False,
        "auto_config_loaded": False,
        "auto_model_loaded": False,
        "auto_tokenizer_loaded": False,
        "finite_forward": False,
        "greedy_generation": False,
        "seeded_generation": False,
        "exact_training_tokens": provenance.get("training_tokens") == FINAL_TOKENS,
        "exact_optimizer_steps": provenance.get("optimizer_steps") == FINAL_STEP,
        "required_disclosures": False,
    }
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    disclosures = (
        "not instruction-tuned",
        "not a chat model",
        "not production-ready",
        "not publicly uploaded",
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
                encoded["input_ids"], max_new_tokens=8, do_sample=True, temperature=0.8, top_p=0.9, top_k=40, seed=20260712, eos_token_id=3
            )
            checks["greedy_generation"] = greedy.shape[1] > encoded["input_ids"].shape[1]
            checks["seeded_generation"] = seeded.shape[1] > encoded["input_ids"].shape[1]
        except Exception as exc:
            failures.append(f"offline round-trip failed: {type(exc).__name__}: {exc}")
    if not all(checks.values()):
        failures.append(f"offline checks incomplete: {checks}")
    report = {
        "schema_version": "darkmind-v2-base-v1-first-pass-98m-export-validation-v1",
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
        payload = export_first_pass()
        print(json.dumps({"output_dir": str(payload["output_dir"]), "reused": payload["reused"]}, indent=2))
        return
    report = validate_export() if args.command == "validate" else validate_export(export_first_pass()["output_dir"])
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
