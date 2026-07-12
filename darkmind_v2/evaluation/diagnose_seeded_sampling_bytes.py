"""Reproduce and explain Stage-1 seeded-sampling byte-fallback findings."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.corpus.detect_mojibake import detect_text
from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.evaluation.generate_fixed_prompts import load_prompts
from darkmind_v2.evaluation.trace_byte_fallback import (
    audit_normal_vocabulary,
    script_for_character,
    trace_generated_tokens,
)
from darkmind_v2.modeling.model_io import load_model_package, model_config_hash
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer, verify_frozen_tokenizer


TARGET_PROMPT_IDS = {
    "tr_ordinary_001",
    "tr_ordinary_002",
    "tr_ordinary_003",
    "tr_ordinary_004",
    "tr_factual_001",
}


def code_points(text: str) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "character": character,
            "code_point": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNNAMED"),
            "script": script_for_character(character),
        }
        for position, character in enumerate(text)
    ]


def diagnose(config_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable diagnosis: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run"]["output_dir"])
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(run_manifest["initial_checkpoint"])
    checkpoint_metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    tokenizer_manifest = verify_frozen_tokenizer()
    tokenizer = FrozenTokenizer()
    model = load_model_package(checkpoint / "model", device="cuda")
    prompts = [
        item
        for item in load_prompts(Path("darkmind_v2/eval/fixed_base_prompts.jsonl"))
        if item["id"] in TARGET_PROMPT_IDS
    ]
    records = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt["prompt"], add_bos=True)
        input_ids = torch.tensor(
            [prompt_ids[-model.config.block_size :]], dtype=torch.long, device="cuda"
        )
        with torch.no_grad():
            logits = model(input_ids).logits
            generated = model.generate_tokens(
                input_ids,
                max_new_tokens=config["evaluation"]["generation_max_new_tokens"],
                do_sample=True,
                seed=config["seed"],
            )[0].tolist()
        generated_ids = generated[len(input_ids[0]) :]
        decoded = tokenizer.decode(generated_ids)
        trace = trace_generated_tokens(tokenizer, generated_ids, decoded)
        reencoded_ids = tokenizer.encode(decoded)
        roundtrip_decoded = tokenizer.decode(reencoded_ids)
        points = code_points(decoded)
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt": prompt["prompt"],
                "prompt_escaped": ascii(prompt["prompt"]),
                "generated_token_ids": generated_ids,
                "token_trace": trace,
                "raw_decoded_output": decoded,
                "escaped_output": ascii(decoded),
                "unicode_code_points": points,
                "detected_scripts": sorted(
                    {item["script"] for item in points if item["script"] not in {"common", "whitespace"}}
                ),
                "replacement_character_positions": [
                    position for position, character in enumerate(decoded) if character == "\ufffd"
                ],
                "mojibake_detector_matches": [
                    {
                        "substring": item.suspicious_substring,
                        "severity": item.severity,
                        "automatic_repair_safe": item.automatic_repair_safe,
                    }
                    for item in detect_text(decoded)
                ],
                "logits_finite": bool(torch.isfinite(logits).all()),
                "token_range_status": "PASS" if trace["token_range_ok"] else "FAIL",
                "tokenizer_hash_status": "PASS",
                "roundtrip": {
                    "reencoded_token_ids": reencoded_ids,
                    "token_ids_equal": reencoded_ids == generated_ids,
                    "decoded_text_equal": roundtrip_decoded == decoded,
                },
            }
        )
    normal_vocabulary = audit_normal_vocabulary(tokenizer)
    exact_reproduction = all(
        record["generated_token_ids"][-1:] == [155]
        and record["token_trace"]["invalid_utf8_byte_runs"][0]["raw_bytes_hex"] == "93"
        for record in records
    )
    hard_integrity_failures = []
    if not exact_reproduction:
        hard_integrity_failures.append("recorded_failure_not_reproduced")
    if normal_vocabulary["result"] != "PASS":
        hard_integrity_failures.append("normal_vocabulary_piece_corruption")
    if not all(record["logits_finite"] for record in records):
        hard_integrity_failures.append("non_finite_logits")
    if not all(record["token_range_status"] == "PASS" for record in records):
        hard_integrity_failures.append("token_range_violation")
    payload = {
        "schema_version": "darkmind-v2-phase2b1-byte-diagnosis-v1",
        "result": "PASS" if not hard_integrity_failures else "FAIL",
        "failure_category": "5. valid byte-fallback tokens sampled in an invalid UTF-8 order",
        "root_cause": (
            "The initial model sampled valid token 155 (<0x93>) as a one-byte run. "
            "0x93 is not a valid UTF-8 start byte, so SentencePiece emitted U+FFFD."
        ),
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "seed": config["seed"],
        "generation": {
            "do_sample": True,
            "max_new_tokens": config["evaluation"]["generation_max_new_tokens"],
        },
        "model_config_hash": model_config_hash(model.config),
        "checkpoint_model_config_hash": checkpoint_metadata["model_config_hash"],
        "tokenizer_name": tokenizer_manifest["tokenizer_name"],
        "tokenizer_hashes": EXPECTED_HASHES,
        "normal_vocabulary_audit": normal_vocabulary,
        "tokenizer_corruption": False,
        "decoder_defect": False,
        "hard_integrity_failures": hard_integrity_failures,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 2B.1 Seeded-Sampling Byte Diagnosis",
        "",
        f"- Result: **{payload['result']}**",
        f"- Failure category: **{payload['failure_category']}**",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Tokenizer corruption: **{payload['tokenizer_corruption']}**",
        f"- Decoder defect: **{payload['decoder_defect']}**",
        "",
        "## Proven Root Cause",
        "",
        payload["root_cause"],
        "",
        "The raw generated characters are preserved. No token was masked, removed, replaced, or sanitized.",
        "",
        "## Token And Byte Evidence",
        "",
    ]
    for record in payload["records"]:
        run = record["token_trace"]["invalid_utf8_byte_runs"][0]
        lines.extend(
            [
                f"### {record['prompt_id']}",
                "",
                f"- Prompt: `{record['prompt_escaped']}`",
                f"- Generated token IDs: `{record['generated_token_ids']}`",
                f"- Token pieces: `{[item['piece'] for item in record['token_trace']['tokens']]}`",
                f"- Raw decoded output: `{record['escaped_output']}`",
                f"- Replacement positions: `{record['replacement_character_positions']}`",
                f"- Byte run: `{run['raw_bytes_hex']}` from token `{run['token_ids']}` / `{run['pieces']}`",
                f"- Strict UTF-8: **FAIL** at byte offset `{run['strict_utf8']['error_offset']}` ({run['strict_utf8']['reason']})",
                f"- SentencePiece byte-run decode: `{run['sentencepiece_decoded_escaped']}`",
                f"- Logits finite / token range / tokenizer hash: **{record['logits_finite']} / {record['token_range_status']} / {record['tokenizer_hash_status']}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact Findings",
            "",
            f"- Normal vocabulary pieces scanned: `{payload['normal_vocabulary_audit']['normal_piece_count']}`",
            f"- Normal-piece U+FFFD or mojibake issues: `{len(payload['normal_vocabulary_audit']['issues'])}`",
            "- Tokenizer files are hash-identical to the frozen manifest.",
            "- The event is a generated-byte-sequence quality warning for this Stage-1 checkpoint, not tokenizer corruption.",
            "- The same event remains release-blocking for any future public-release candidate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1_r2.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/"
            "evaluations/byte_trace_policy_v1/seeded_sampling_byte_diagnosis.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("darkmind_v2/reports/phase2b1_seeded_sampling_byte_diagnosis.md"),
    )
    args = parser.parse_args()
    result = diagnose(args.config, args.output, args.report)
    print(json.dumps({"result": result["result"], "category": result["failure_category"]}, indent=2))
    raise SystemExit(1 if result["hard_integrity_failures"] else 0)


if __name__ == "__main__":
    main()
