"""Validate the complete local public-preview package without network access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.evaluation.trace_byte_fallback import trace_generated_tokens
from darkmind_v2.evaluation.validate_generation_health import classify_generation_health
from darkmind_v2.export.validate_stage1_huggingface import validate as validate_export
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


REQUIRED_MODEL_CARD_DISCLOSURES = [
    "DarkMind v2 Tiny Stage-1",
    "decoder-only causal base language model",
    "9,369,088",
    "SentencePiece BPE, 24,000",
    "1,048,576",
    "8.9%",
    "not instruction-tuned",
    "not a chat model",
    "not production-ready",
    "repetition",
    "Factual statements are unverified and unreliable",
    "Script mixing",
    "corpus_attribution.json",
    "No standalone distribution license",
    "trust_remote_code=True",
    "Public research-preview audit result: FAIL",
]


def validate_model_card(text: str) -> list[str]:
    return [item for item in REQUIRED_MODEL_CARD_DISCLOSURES if item not in text]


def validate_public_preview_package(export_dir: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    base = validate_export(export_dir)
    failures = list(base["failures"])
    warnings = []
    present = {path.name for path in export_dir.iterdir() if path.is_file()}
    forbidden_weights = sorted(
        name for name in present if name.endswith((".bin", ".pt", ".pth", ".pkl", ".pickle"))
    )
    if forbidden_weights:
        failures.append(f"pickle/non-safetensors weight files present: {forbidden_weights}")
    if "model.safetensors" not in present:
        failures.append("model.safetensors is missing")

    model_card = (export_dir / "README.md").read_text(encoding="utf-8")
    missing_disclosures = validate_model_card(model_card)
    if missing_disclosures:
        failures.append(f"missing model-card disclosures: {missing_disclosures}")
    attribution = json.loads((export_dir / "corpus_attribution.json").read_text(encoding="utf-8"))
    if len(attribution.get("sources", [])) != 7:
        failures.append("corpus attribution must list all seven source groups")
    if any(not source.get("license_id") or not source.get("license_url") for source in attribution.get("sources", [])):
        failures.append("corpus source license reference is incomplete")
    license_text = (export_dir / "LICENSE_INFORMATION.md").read_text(encoding="utf-8")
    if "Repository source code: MIT License" not in license_text or "no standalone distribution license" not in license_text:
        failures.append("license information is incomplete")
    if "no standalone distribution license" in license_text:
        warnings.append("model_weight_distribution_license_not_designated")

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    expected_config = {"vocab_size": 24000, "block_size": 256, "n_layer": 4, "n_head": 4, "n_embd": 256}
    if any(config.get(key) != value for key, value in expected_config.items()):
        failures.append("model config does not match the selected checkpoint architecture")
    special = json.loads((export_dir / "special_tokens_map.json").read_text(encoding="utf-8"))
    if special.get("pad_token") != "<pad>" or special.get("eos_token") != "</s>":
        failures.append("special-token map mismatch")
    provenance = json.loads((export_dir / "provenance_manifest.json").read_text(encoding="utf-8"))
    if provenance.get("selected_checkpoint_model_sha256") != "5e2fd69d4775940629926a7bf659e36beafe4c1cd544feb8d97beeae6537b097":
        failures.append("selected checkpoint provenance hash mismatch")

    seeded = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        ).eval()
        hf_tokenizer = AutoTokenizer.from_pretrained(
            export_dir, trust_remote_code=True, local_files_only=True
        )
        frozen = FrozenTokenizer()
        prompt_ids = frozen.encode("A short local package check", add_bos=True)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            generated = model.generate_tokens(
                input_ids,
                max_new_tokens=16,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                seed=20260712,
                eos_token_id=frozen.eos_token_id,
            )[0].tolist()
        continuation = generated[len(prompt_ids) :]
        output = frozen.decode(continuation)
        trace = trace_generated_tokens(frozen, continuation, output)
        policy = classify_generation_health(
            output,
            continuation,
            checkpoint_stage="research_preview",
            token_trace=trace,
        )
        seeded = {
            "output": output,
            "token_ids": continuation,
            "hard_failures": policy["hard_failures"],
            "warnings": policy["warnings"],
            "hf_tokenizer_vocab_size": hf_tokenizer.vocab_size,
        }
        if policy["hard_failures"]:
            failures.append(f"offline seeded generation hard failures: {policy['hard_failures']}")
    except Exception as exc:
        failures.append(f"offline seeded generation failed: {type(exc).__name__}: {exc}")

    file_hashes = json.loads((export_dir / "file_hashes.json").read_text(encoding="utf-8"))
    if any(sha256_file(export_dir / filename) != expected for filename, expected in file_hashes.items()):
        failures.append("one or more final package file hashes differ")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "base_offline_validation": base,
        "safetensors_only": not forbidden_weights and "model.safetensors" in present,
        "missing_model_card_disclosures": missing_disclosures,
        "attribution_sources": len(attribution.get("sources", [])),
        "license_information_present": not any("license information" in item for item in failures),
        "model_weight_distribution_license_designated": False,
        "offline_seeded_generation": seeded,
        "network_required": False,
        "verified_file_hashes": len(file_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "export_dir", type=Path, nargs="?",
        default=Path("darkmind_v2/data/phase2a/exports/darkmind-v2-tiny-stage1"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/public_preview_v2/release_package_validation.json"
        ),
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite package validation report: {args.report}")
    report = validate_public_preview_package(args.export_dir)
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
