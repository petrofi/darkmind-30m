"""Deterministic public research-preview audit for the selected Stage-1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from darkmind_v2.corpus.detect_mojibake import detect_text
from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash, sha256_file
from darkmind_v2.evaluation.trace_byte_fallback import script_for_character, trace_generated_tokens
from darkmind_v2.evaluation.validate_generation_health import classify_generation_health
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer, verify_frozen_tokenizer


EXPECTED_CHECKPOINT_SHA256 = "5e2fd69d4775940629926a7bf659e36beafe4c1cd544feb8d97beeae6537b097"
EXPECTED_CATEGORY_COUNTS = {
    "tr:ordinary_text": 60,
    "tr:factual_encyclopedic": 30,
    "tr:technical": 20,
    "en:ordinary_text": 50,
    "en:factual_encyclopedic": 20,
    "en:technical": 10,
    "code:code_structured": 10,
}
SAMPLING_SUBSET_COUNTS = {
    "tr:ordinary_text": 15,
    "tr:factual_encyclopedic": 8,
    "tr:technical": 5,
    "en:ordinary_text": 12,
    "en:factual_encyclopedic": 5,
    "en:technical": 3,
    "code:code_structured": 2,
}


def load_audit_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(f"{item['language']}:{item['category']}" for item in prompts)
    if len(prompts) != 200 or dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"public-preview prompt distribution mismatch: {dict(counts)}")
    if len({item["id"] for item in prompts}) != len(prompts):
        raise ValueError("public-preview prompt IDs must be unique")
    return prompts


def sampling_subset(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    used: Counter[str] = Counter()
    for prompt in prompts:
        key = f"{prompt['language']}:{prompt['category']}"
        if used[key] < SAMPLING_SUBSET_COUNTS[key]:
            selected.append(prompt)
            used[key] += 1
    if len(selected) != 50 or dict(used) != SAMPLING_SUBSET_COUNTS:
        raise ValueError("sampling subset distribution mismatch")
    return selected


def exact_repeated_ngram_loops(token_ids: list[int], minimum_n: int = 2, maximum_n: int = 4) -> list[dict[str, Any]]:
    loops = []
    for n in range(minimum_n, maximum_n + 1):
        for start in range(0, len(token_ids) - 2 * n + 1):
            first = token_ids[start : start + n]
            if first == token_ids[start + n : start + 2 * n]:
                loops.append({"start": start, "n": n, "token_ids": first})
    return loops


def longest_repeated_token_run(token_ids: list[int]) -> int:
    longest = 0
    current = 0
    previous = None
    for token_id in token_ids:
        current = current + 1 if token_id == previous else 1
        previous = token_id
        longest = max(longest, current)
    return longest


def percentile(values: list[int | float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def unicode_trace(text: str) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "code_point": f"U+{ord(character):04X}",
            "character_escaped": ascii(character),
            "script": script_for_character(character),
        }
        for position, character in enumerate(text)
    ]


def generate_record(
    model: torch.nn.Module,
    tokenizer: FrozenTokenizer,
    prompt: dict[str, Any],
    *,
    max_new_tokens: int,
    do_sample: bool,
    profile_name: str,
    seed: int | None,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt["prompt"], add_bos=True)
    input_ids = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
    with torch.no_grad():
        initial_logits = model(input_ids).logits
        generated = model.generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            eos_token_id=tokenizer.eos_token_id,
        )[0].tolist()
    token_ids = generated[len(input_ids[0]) :]
    decode_exception = None
    try:
        output = tokenizer.decode(token_ids)
    except Exception as exc:
        output = ""
        decode_exception = f"{type(exc).__name__}: {exc}"
    token_trace = trace_generated_tokens(tokenizer, token_ids, output) if decode_exception is None else {
        "tokens": [], "byte_runs": [], "invalid_utf8_byte_runs": [], "normal_piece_issues": [],
        "generated_invalid_utf8_byte_sequence": False,
        "replacement_character_from_generated_bytes": False,
        "token_range_ok": all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids),
    }
    policy = classify_generation_health(
        output,
        token_ids,
        checkpoint_stage="research_preview",
        maximum_repetition_ratio=float(prompt.get("maximum_repetition_ratio", 0.35)),
        decode_exception=decode_exception,
        logits_finite=bool(torch.isfinite(initial_logits).all()),
        token_trace=token_trace,
    )
    loops = exact_repeated_ngram_loops(token_ids)
    warnings = set(policy["warnings"])
    if loops:
        warnings.add("repetition")
    if len(token_ids) < 4:
        warnings.add("very_short_output")
    if prompt["category"] == "factual_encyclopedic":
        warnings.add("factual_unreliability")
    if prompt["category"] == "code_structured" and not any(symbol in output for symbol in "(){}[]:=;\n"):
        warnings.add("code_generation_failure")
    policy["warnings"] = sorted(warnings)
    policy["findings"] = sorted(set(policy["findings"]) | warnings)
    points = unicode_trace(output)
    return {
        "prompt_id": prompt["id"],
        "language": prompt["language"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "settings": {
            "profile": profile_name,
            "do_sample": do_sample,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "eos_aware": True,
        },
        "output": output,
        "output_escaped": ascii(output),
        "token_ids": token_ids,
        "token_trace": token_trace,
        "generated_token_count": len(token_ids),
        "unicode_code_points": points,
        "detected_scripts": sorted({item["script"] for item in points if item["script"] not in {"common", "whitespace"}}),
        "replacement_character_count": output.count("\ufffd"),
        "mojibake_matches": [item.suspicious_substring for item in detect_text(output)],
        "exact_repeated_ngram_loops": loops,
        "longest_repeated_token_run": longest_repeated_token_run(token_ids),
        "unique_token_ratio": len(set(token_ids)) / len(token_ids) if token_ids else 0.0,
        "eos_completed": bool(token_ids and token_ids[-1] == tokenizer.eos_token_id),
        "empty_output": not output.strip(),
        "decode_exception": decode_exception,
        "policy": policy,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    hard = Counter(item for record in records for item in record["policy"]["hard_failures"])
    warnings = Counter(item for record in records for item in record["policy"]["warnings"])
    lengths = [record["generated_token_count"] for record in records]
    unique_ratios = [record["unique_token_ratio"] for record in records]
    return {
        "generations": len(records),
        "hard_failure_counts": dict(sorted(hard.items())),
        "hard_failure_total": sum(hard.values()),
        "quality_warning_counts": dict(sorted(warnings.items())),
        "quality_warning_total": sum(warnings.values()),
        "invalid_utf8_sequence_count": sum(len(record["token_trace"]["invalid_utf8_byte_runs"]) for record in records),
        "replacement_character_count": sum(record["replacement_character_count"] for record in records),
        "mojibake_output_count": sum(bool(record["mojibake_matches"]) for record in records),
        "unexpected_script_output_count": sum("unexpected_script" in record["policy"]["warnings"] for record in records),
        "mixed_script_output_count": sum("mixed_script" in record["policy"]["warnings"] for record in records),
        "special_token_leakage_count": sum("special_token_leakage" in record["policy"]["warnings"] for record in records),
        "empty_output_count": sum(record["empty_output"] for record in records),
        "exact_repeated_ngram_loop_outputs": sum(bool(record["exact_repeated_ngram_loops"]) for record in records),
        "longest_repeated_token_run": max((record["longest_repeated_token_run"] for record in records), default=0),
        "mean_unique_token_ratio": sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0.0,
        "p50_unique_token_ratio": percentile(unique_ratios, 0.50),
        "p50_output_tokens": percentile(lengths, 0.50),
        "p90_output_tokens": percentile(lengths, 0.90),
        "p95_output_tokens": percentile(lengths, 0.95),
        "eos_completion_rate": sum(record["eos_completed"] for record in records) / len(records) if records else 0.0,
    }


def write_manifest(path: Path, *, settings: dict[str, Any], prompt_hash: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit manifest: {path}")
    core = {"settings": settings, "prompt_manifest_sha256": prompt_hash, "results": records}
    payload = {**core, "summary": summarize(records), "deterministic_content_hash": canonical_json_hash(core)}
    atomic_write_json(path, payload)
    return payload


def preserve_hard_failure(output_dir: Path, mode: str, record: dict[str, Any], attempted: int) -> None:
    evidence = output_dir / f"{mode}_hard_failure_{attempted:06d}.json"
    atomic_write_json(evidence, record)
    atomic_write_json(
        output_dir / "audit_stopped.json",
        {
            "result": "FAIL",
            "mode": mode,
            "attempted_generations": attempted,
            "prompt_id": record["prompt_id"],
            "hard_failures": record["policy"]["hard_failures"],
            "evidence": str(evidence),
        },
    )


def audit(config_path: Path, gates_path: Path, prompts_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse public-preview audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run"]["output_dir"])
    checkpoint = run_dir / "checkpoints" / "step_000256_tokens_001048576"
    actual_checkpoint_hash = sha256_file(checkpoint / "model" / "model.safetensors")
    if actual_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("selected checkpoint hash mismatch")
    verify_frozen_tokenizer()
    prompts = load_audit_prompts(prompts_path)
    prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer()

    greedy_records = []
    for prompt in prompts:
        record = generate_record(
            model, tokenizer, prompt,
            max_new_tokens=gates["generation"]["greedy_max_new_tokens"],
            do_sample=False, profile_name="greedy", seed=None,
        )
        greedy_records.append(record)
        if record["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "greedy", record, len(greedy_records))
            raise RuntimeError(f"public-preview greedy hard failure: {record['prompt_id']} {record['policy']['hard_failures']}")
    greedy = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={"mode": "greedy", "max_new_tokens": gates["generation"]["greedy_max_new_tokens"], "eos_aware": True},
        prompt_hash=prompt_hash,
        records=greedy_records,
    )

    sample_prompts = sampling_subset(prompts)
    sampling_records = []
    for profile_name, profile in gates["generation"]["sampling_profiles"].items():
        for seed in gates["generation"]["sampling_seeds"]:
            for prompt in sample_prompts:
                record = generate_record(
                    model, tokenizer, prompt,
                    max_new_tokens=gates["generation"]["greedy_max_new_tokens"],
                    do_sample=True, profile_name=profile_name, seed=seed,
                    temperature=profile["temperature"], top_p=profile["top_p"], top_k=profile["top_k"],
                )
                sampling_records.append(record)
                if record["policy"]["hard_failures"]:
                    preserve_hard_failure(output_dir, "sampling", record, len(sampling_records))
                    raise RuntimeError(
                        f"public-preview sampling hard failure: {record['prompt_id']} "
                        f"profile={profile_name} seed={seed} {record['policy']['hard_failures']}"
                    )
    sampling = write_manifest(
        output_dir / "sampling_manifest.json",
        settings={
            "mode": "seeded_sampling_matrix",
            "max_new_tokens": gates["generation"]["greedy_max_new_tokens"],
            "eos_aware": True,
            "profiles": gates["generation"]["sampling_profiles"],
            "seeds": gates["generation"]["sampling_seeds"],
            "prompt_subset_count": len(sample_prompts),
            "prompt_subset_distribution": SAMPLING_SUBSET_COUNTS,
        },
        prompt_hash=prompt_hash,
        records=sampling_records,
    )
    report = {
        "schema_version": "darkmind-v2-public-preview-audit-v1",
        "result": "PASS" if not greedy["summary"]["hard_failure_total"] and not sampling["summary"]["hard_failure_total"] else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": actual_checkpoint_hash,
        "prompt_manifest": str(prompts_path),
        "prompt_manifest_sha256": prompt_hash,
        "greedy": greedy["summary"],
        "sampling": sampling["summary"],
        "greedy_manifest_sha256": sha256_file(output_dir / "greedy_manifest.json"),
        "sampling_manifest_sha256": sha256_file(output_dir / "sampling_manifest.json"),
    }
    atomic_write_json(output_dir / "audit_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("darkmind_v2/config/train_tiny_stage1_r2.json"))
    parser.add_argument("--gates", type=Path, default=Path("darkmind_v2/config/public_research_preview_gates.json"))
    parser.add_argument("--prompts", type=Path, default=Path("darkmind_v2/eval/public_preview_prompts.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/evaluations/public_preview_v1"),
    )
    args = parser.parse_args()
    report = audit(args.config, args.gates, args.prompts, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
