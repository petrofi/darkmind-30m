"""Diagnose deterministic zero-token generations before changing gate policy."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.corpus.detect_mojibake import detect_text
from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.evaluation.generate_fixed_prompts import load_prompts
from darkmind_v2.evaluation.validate_generation_health import validate_text_health
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import (
    EXPECTED_HASHES,
    FrozenTokenizer,
    verify_frozen_tokenizer,
)
from darkmind_v2.training.train_tiny_smoke import set_deterministic_seed


TARGET_PROMPT_IDS = {"tr_technical_002", "en_technical_001"}


def script_for_character(character: str) -> str:
    if character.isspace():
        return "whitespace"
    category = unicodedata.category(character)
    if category.startswith(("P", "N", "S")):
        return "common"
    name = unicodedata.name(character, "UNNAMED")
    for script in ("LATIN", "CYRILLIC", "ARABIC", "HEBREW", "GREEK"):
        if script in name:
            return script.lower()
    return "other"


def diagnose(
    model_config_path: Path,
    output_path: Path,
    *,
    max_new_tokens: int = 16,
    seed: int = 20260712,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnosis: {output_path}")
    manifest = verify_frozen_tokenizer()
    tokenizer = FrozenTokenizer()
    config = DarkMindV2Config.from_json_file(model_config_path)
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DarkMindV2ForCausalLM(config).to(device).eval()
    prompts = [
        item
        for item in load_prompts(Path("darkmind_v2/eval/fixed_base_prompts.jsonl"))
        if item["id"] in TARGET_PROMPT_IDS
    ]
    records = []
    hard_failures = []
    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt["prompt"], add_bos=True)
            input_ids = torch.tensor([prompt_ids[-config.block_size :]], dtype=torch.long, device=device)
            logits = model(input_ids).logits
            logits_finite = bool(torch.all(torch.isfinite(logits)))
            generated = model.generate_tokens(input_ids, max_new_tokens=max_new_tokens)[0].tolist()
            generated_ids = generated[len(input_ids[0]) :]
            token_range_violations = sum(
                int(token_id < 0 or token_id >= tokenizer.vocab_size)
                for token_id in generated_ids
            )
            decode_exception = None
            try:
                decoded = tokenizer.decode(generated_ids)
            except Exception as exc:  # recorded as a hard pipeline failure
                decoded = ""
                decode_exception = f"{type(exc).__name__}: {exc}"
            pieces = [
                {
                    "id": token_id,
                    "piece": tokenizer.id_to_piece(token_id),
                    "kind": "byte_fallback" if tokenizer.is_byte_fallback_id(token_id) else "normal_vocabulary",
                }
                for token_id in generated_ids
                if 0 <= token_id < tokenizer.vocab_size
            ]
            code_points = [
                {
                    "character": character,
                    "code_point": f"U+{ord(character):04X}",
                    "name": unicodedata.name(character, "UNNAMED"),
                    "script": script_for_character(character),
                }
                for character in decoded
            ]
            invalid_unicode = 0
            try:
                decoded.encode("utf-8", errors="strict")
            except UnicodeError:
                invalid_unicode = 1
            reencoded_ids = tokenizer.encode(decoded)
            roundtrip_text = tokenizer.decode(reencoded_ids)
            health = validate_text_health(decoded, generated_ids)
            record = {
                "id": prompt["id"],
                "language": prompt["language"],
                "prompt": prompt["prompt"],
                "decoded_output": decoded,
                "generated_token_ids": generated_ids,
                "sentencepiece_pieces": pieces,
                "unicode_code_points": code_points,
                "detected_scripts": sorted(
                    {item["script"] for item in code_points if item["script"] not in {"whitespace", "common"}}
                ),
                "byte_fallback_pieces": sum(item["kind"] == "byte_fallback" for item in pieces),
                "normal_vocabulary_pieces": sum(item["kind"] == "normal_vocabulary" for item in pieces),
                "roundtrip": {
                    "reencoded_token_ids": reencoded_ids,
                    "token_ids_equal": reencoded_ids == generated_ids,
                    "decoded_text_equal": roundtrip_text == decoded,
                },
                "invalid_unicode_count": invalid_unicode,
                "replacement_character_count": decoded.count("\ufffd"),
                "mojibake_count": len(detect_text(decoded)),
                "token_range_violations": token_range_violations,
                "decode_exception": decode_exception,
                "tokenizer_hash_result": "PASS",
                "model_logits_finite": logits_finite,
                "generation_health": health,
            }
            failures = []
            if invalid_unicode:
                failures.append("invalid_unicode")
            if record["replacement_character_count"]:
                failures.append("replacement_character")
            if record["mojibake_count"]:
                failures.append("mojibake")
            if token_range_violations:
                failures.append("token_range")
            if decode_exception:
                failures.append("decode_exception")
            if not logits_finite:
                failures.append("non_finite_logits")
            record["hard_failures"] = failures
            hard_failures.extend({"id": prompt["id"], "failure": item} for item in failures)
            records.append(record)
    payload = {
        "result": "FAIL" if hard_failures else "PASS",
        "seed": seed,
        "model_config": str(model_config_path),
        "max_new_tokens": max_new_tokens,
        "tokenizer_name": manifest["tokenizer_name"],
        "tokenizer_hashes": {
            "model": EXPECTED_HASHES["tokenizer.model"],
            "vocab": EXPECTED_HASHES["tokenizer.vocab"],
            "freeze_manifest": EXPECTED_HASHES["tokenizer_freeze_manifest.json"],
        },
        "hard_failures": hard_failures,
        "records": sorted(records, key=lambda item: item["id"]),
    }
    atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("darkmind_v2/config/model_tiny_smoke.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("darkmind_v2/reports/phase2a_runtime/phase2b1_initial_generation_diagnosis.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    report = diagnose(args.model_config, args.output, max_new_tokens=args.max_new_tokens)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    raise SystemExit(1 if report["hard_failures"] else 0)


if __name__ == "__main__":
    main()
