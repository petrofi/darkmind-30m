"""Validate local Hugging Face package structure and offline loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.export.export_huggingface import REQUIRED_EXPORT_FILES, sha256_file


def validate_export(path: Path, *, offline_roundtrip: bool = True) -> dict[str, Any]:
    failures = []
    present = {item.name for item in path.iterdir() if item.is_file()}
    missing = sorted(REQUIRED_EXPORT_FILES - present)
    if missing:
        failures.append(f"missing files: {missing}")
    metadata = json.loads((path / "training_evaluation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("fixture_only") is not True or metadata.get("trained_model") is not False:
        failures.append("fixture export is not labeled safely")
    provenance = json.loads((path / "provenance_manifest.json").read_text(encoding="utf-8"))
    for filename, expected in provenance.get("files", {}).items():
        if sha256_file(path / filename) != expected:
            failures.append(f"provenance hash mismatch: {filename}")

    auto_model_loaded = False
    auto_tokenizer_loaded = False
    if offline_roundtrip and not failures:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=True,
            )
            sample = tokenizer("pipeline check", return_tensors="pt")
            with torch.no_grad():
                output = model(**sample)
            auto_model_loaded = output.logits.shape[-1] == 24000
            auto_tokenizer_loaded = tokenizer.vocab_size == 24000
        except Exception as exc:  # explicit report rather than a silent skip
            failures.append(f"offline AutoClass round-trip failed: {exc}")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "auto_model_loaded": auto_model_loaded,
        "auto_tokenizer_loaded": auto_tokenizer_loaded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    report = validate_export(args.export_dir, offline_roundtrip=not args.structure_only)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
