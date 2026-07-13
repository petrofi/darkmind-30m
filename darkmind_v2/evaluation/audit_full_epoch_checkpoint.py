"""Run the corrected public-preview audit against an explicit checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, sha256_file
from darkmind_v2.evaluation.audit_public_preview import (
    SAMPLING_SUBSET_COUNTS,
    generate_record,
    load_audit_prompts,
    preserve_hard_failure,
    sampling_subset,
    write_manifest,
)
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer, verify_frozen_tokenizer


def audit_checkpoint(
    checkpoint: Path,
    gates_path: Path,
    prompts_path: Path,
    output_dir: Path,
    *,
    checkpoint_stage: str,
    expected_model_hash: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse immutable audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(checkpoint / "model" / "model.safetensors")
    if expected_model_hash is not None and actual_hash != expected_model_hash:
        raise ValueError("explicit checkpoint hash mismatch")
    verify_frozen_tokenizer()
    prompts = load_audit_prompts(prompts_path)
    prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    model = load_model_package(checkpoint / "model", device="cuda")
    tokenizer = FrozenTokenizer()
    max_new_tokens = gates["generation"]["greedy_max_new_tokens"]

    greedy_records = []
    for prompt in prompts:
        record = generate_record(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            profile_name="greedy",
            seed=None,
            checkpoint_stage=checkpoint_stage,
        )
        greedy_records.append(record)
        if record["policy"]["hard_failures"]:
            preserve_hard_failure(output_dir, "greedy", record, len(greedy_records))
            raise RuntimeError(
                f"checkpoint audit greedy hard failure: {record['prompt_id']} "
                f"{record['policy']['hard_failures']}"
            )
    greedy = write_manifest(
        output_dir / "greedy_manifest.json",
        settings={
            "mode": "greedy",
            "max_new_tokens": max_new_tokens,
            "eos_aware": True,
            "terminal_eos_is_not_special_token_leakage": True,
            "checkpoint_stage": checkpoint_stage,
        },
        prompt_hash=prompt_hash,
        records=greedy_records,
    )

    sample_prompts = sampling_subset(prompts)
    sampling_records = []
    for profile_name, profile in gates["generation"]["sampling_profiles"].items():
        for seed in gates["generation"]["sampling_seeds"]:
            for prompt in sample_prompts:
                record = generate_record(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    profile_name=profile_name,
                    seed=seed,
                    temperature=profile["temperature"],
                    top_p=profile["top_p"],
                    top_k=profile["top_k"],
                    checkpoint_stage=checkpoint_stage,
                )
                sampling_records.append(record)
                if record["policy"]["hard_failures"]:
                    preserve_hard_failure(output_dir, "sampling", record, len(sampling_records))
                    raise RuntimeError(
                        f"checkpoint audit sampling hard failure: {record['prompt_id']} "
                        f"profile={profile_name} seed={seed} {record['policy']['hard_failures']}"
                    )
    sampling = write_manifest(
        output_dir / "sampling_manifest.json",
        settings={
            "mode": "seeded_sampling_matrix",
            "max_new_tokens": max_new_tokens,
            "eos_aware": True,
            "terminal_eos_is_not_special_token_leakage": True,
            "checkpoint_stage": checkpoint_stage,
            "profiles": gates["generation"]["sampling_profiles"],
            "seeds": gates["generation"]["sampling_seeds"],
            "prompt_subset_count": len(sample_prompts),
            "prompt_subset_distribution": SAMPLING_SUBSET_COUNTS,
        },
        prompt_hash=prompt_hash,
        records=sampling_records,
    )
    report = {
        "schema_version": "darkmind-v2-public-preview-audit-v2",
        "policy": "public_preview_v2",
        "result": "PASS" if not greedy["summary"]["hard_failure_total"] and not sampling["summary"]["hard_failure_total"] else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_model_sha256": actual_hash,
        "prompt_manifest": str(prompts_path),
        "prompt_manifest_sha256": prompt_hash,
        "raw_outputs_retained": True,
        "sanitization_performed": False,
        "greedy": greedy["summary"],
        "sampling": sampling["summary"],
        "greedy_manifest_sha256": sha256_file(output_dir / "greedy_manifest.json"),
        "sampling_manifest_sha256": sha256_file(output_dir / "sampling_manifest.json"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_dir / "audit_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gates", type=Path, default=Path("darkmind_v2/config/public_research_preview_gates.json"))
    parser.add_argument("--prompts", type=Path, default=Path("darkmind_v2/eval/public_preview_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-stage",
        choices=("initialization", "stage1", "midpoint", "stage1_final", "best_validation", "research_preview"),
        required=True,
    )
    parser.add_argument("--expected-model-hash")
    args = parser.parse_args()
    report = audit_checkpoint(
        args.checkpoint,
        args.gates,
        args.prompts,
        args.output_dir,
        checkpoint_stage=args.checkpoint_stage,
        expected_model_hash=args.expected_model_hash,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
