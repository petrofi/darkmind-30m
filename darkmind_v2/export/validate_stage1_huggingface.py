"""Offline validation for the local DarkMind v2 Stage-1 Hugging Face export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.export.export_stage1_huggingface import REQUIRED_STAGE1_EXPORT_FILES
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES


def validate(export_dir: Path) -> dict[str, Any]:
    failures = []
    present = {path.name for path in export_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_STAGE1_EXPORT_FILES - present)
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

    auto_config_loaded = False
    auto_model_loaded = False
    auto_tokenizer_loaded = False
    safetensors_loaded = False
    forward_finite = False
    greedy_generation = None
    try:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True, local_files_only=True)
        auto_config_loaded = config.model_type == "darkmind_v2"
        model = AutoModelForCausalLM.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        )
        auto_model_loaded = model.num_parameters() == 9_369_088
        auto_tokenizer_loaded = tokenizer.vocab_size == 24_000
        with safe_open(export_dir / "model.safetensors", framework="pt", device="cpu") as handle:
            safetensors_loaded = bool(handle.keys())
        sample = tokenizer("DarkMind pipeline check", return_tensors="pt")
        with torch.no_grad():
            output = model(**sample)
        forward_finite = bool(torch.isfinite(output.logits).all()) and math.isfinite(float(output.logits.mean()))
        generated_ids = model.generate_tokens(sample["input_ids"], max_new_tokens=8, do_sample=False)[0].tolist()
        greedy_generation = {
            "token_ids": generated_ids,
            "decoded": tokenizer.decode(generated_ids, skip_special_tokens=False),
        }
    except Exception as exc:
        failures.append(f"offline reload failed: {type(exc).__name__}: {exc}")
    for name, passed in {
        "AutoConfig": auto_config_loaded,
        "AutoModelForCausalLM": auto_model_loaded,
        "AutoTokenizer": auto_tokenizer_loaded,
        "safetensors": safetensors_loaded,
        "finite forward": forward_finite,
    }.items():
        if not passed:
            failures.append(f"{name} validation failed")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "auto_config_loaded": auto_config_loaded,
        "auto_model_loaded": auto_model_loaded,
        "auto_tokenizer_loaded": auto_tokenizer_loaded,
        "safetensors_loaded": safetensors_loaded,
        "forward_finite": forward_finite,
        "greedy_generation": greedy_generation,
        "verified_file_hashes": len(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "export_dir",
        type=Path,
        nargs="?",
        default=Path("darkmind_v2/data/phase2a/exports/darkmind-v2-tiny-stage1"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/byte_trace_policy_v2/huggingface_offline_validation.json"
        ),
    )
    args = parser.parse_args()
    report = validate(args.export_dir)
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite offline validation: {args.report}")
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
