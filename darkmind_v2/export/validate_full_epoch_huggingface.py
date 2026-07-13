"""Offline-only validation for the local Phase 2C Hugging Face export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.export.export_full_epoch_huggingface import REQUIRED_FULL_EPOCH_EXPORT_FILES
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES


def validate(export_dir: Path) -> dict[str, Any]:
    failures = []
    present = {path.name for path in export_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FULL_EPOCH_EXPORT_FILES - present)
    if missing:
        failures.append(f"missing export files: {missing}")
    file_hashes = json.loads((export_dir / "file_hashes.json").read_text(encoding="utf-8"))
    for filename, expected in file_hashes.items():
        if sha256_file(export_dir / filename) != expected:
            failures.append(f"file hash mismatch: {filename}")
    if sha256_file(export_dir / "tokenizer.model") != EXPECTED_HASHES["tokenizer.model"]:
        failures.append("exported tokenizer.model differs from frozen hash")
    if sha256_file(export_dir / "tokenizer.vocab") != EXPECTED_HASHES["tokenizer.vocab"]:
        failures.append("exported tokenizer.vocab differs from frozen hash")
    model_card = (export_dir / "README.md").read_text(encoding="utf-8").lower()
    for disclosure in (
        "tiny capacity diagnostic",
        "not instruction-tuned",
        "not a chat model",
        "not production-ready",
        "not publicly released",
        "model-weight distribution license is unresolved",
    ):
        if disclosure not in model_card:
            failures.append(f"missing model-card disclosure: {disclosure}")

    checks = {
        "auto_config_loaded": False,
        "auto_model_loaded": False,
        "auto_tokenizer_loaded": False,
        "safetensors_loaded": False,
        "finite_forward": False,
        "greedy_generation": False,
        "seeded_generation": False,
        "seeded_generation_reproducible": False,
    }
    generations: dict[str, Any] = {}
    try:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True, local_files_only=True)
        checks["auto_config_loaded"] = config.model_type == "darkmind_v2"
        model = AutoModelForCausalLM.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        )
        checks["auto_model_loaded"] = model.num_parameters() == 9_369_088
        checks["auto_tokenizer_loaded"] = tokenizer.vocab_size == 24_000
        with safe_open(export_dir / "model.safetensors", framework="pt", device="cpu") as handle:
            checks["safetensors_loaded"] = bool(handle.keys())
        sample = tokenizer("DarkMind pipeline check", return_tensors="pt")
        with torch.no_grad():
            output = model(**sample)
        checks["finite_forward"] = bool(torch.isfinite(output.logits).all()) and math.isfinite(float(output.logits.mean()))
        prompt_length = sample["input_ids"].shape[1]
        greedy_ids = model.generate_tokens(
            sample["input_ids"], max_new_tokens=8, do_sample=False, eos_token_id=tokenizer.eos_token_id
        )[0].tolist()
        seeded_a = model.generate_tokens(
            sample["input_ids"], max_new_tokens=8, do_sample=True, temperature=0.8,
            top_p=0.9, top_k=40, seed=20260712, eos_token_id=tokenizer.eos_token_id,
        )[0].tolist()
        seeded_b = model.generate_tokens(
            sample["input_ids"], max_new_tokens=8, do_sample=True, temperature=0.8,
            top_p=0.9, top_k=40, seed=20260712, eos_token_id=tokenizer.eos_token_id,
        )[0].tolist()
        checks["greedy_generation"] = len(greedy_ids) > prompt_length
        checks["seeded_generation"] = len(seeded_a) > prompt_length
        checks["seeded_generation_reproducible"] = seeded_a == seeded_b
        generations = {
            "greedy_token_ids": greedy_ids,
            "greedy_decoded": tokenizer.decode(greedy_ids, skip_special_tokens=False),
            "seeded_token_ids": seeded_a,
            "seeded_decoded": tokenizer.decode(seeded_a, skip_special_tokens=False),
            "seeded_sha256": hashlib.sha256(json.dumps(seeded_a).encode("ascii")).hexdigest(),
        }
    except Exception as exc:
        failures.append(f"offline reload failed: {type(exc).__name__}: {exc}")
    for name, passed in checks.items():
        if not passed:
            failures.append(f"{name} validation failed")
    return {
        "schema_version": "darkmind-v2-phase2c-offline-export-validation-v1",
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "checks": checks,
        "generations": generations,
        "verified_file_hashes": len(file_hashes),
        "local_files_only": True,
        "trust_remote_code": True,
        "upload_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "export_dir",
        type=Path,
        nargs="?",
        default=Path("darkmind_v2/data/phase2c/exports/darkmind-v2-tiny-full-epoch"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2c/runs/tiny_full_epoch_seed20260712_v1/"
            "evaluations/huggingface_offline_validation.json"
        ),
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite offline validation: {args.report}")
    report = validate(args.export_dir)
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
