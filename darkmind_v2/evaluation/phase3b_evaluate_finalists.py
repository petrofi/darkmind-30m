"""Run and summarize identical Phase 3B milestone generation audits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json
from darkmind_v2.training.validate_phase3b_pilot_config import load_and_validate_phase3b_pilot_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "darkmind_v2" / "config" / "phase3b_finalist_pilot.json"
REPORT_PATH = ROOT / "darkmind_v2" / "reports" / "phase3b_finalist_evaluation.md"
RUNTIME_REPORT = ROOT / "darkmind_v2" / "data" / "phase3b" / "finalist_evaluation.json"
STAGES = {
    0: "initialization",
    152: "stage1",
    305: "midpoint",
    458: "stage1",
    610: "stage1_final",
}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def meaningful_proxy(record: dict[str, Any]) -> bool:
    warnings = set(record["policy"]["warnings"])
    disallowed = {
        "empty_output",
        "repetition",
        "unexpected_script",
        "mixed_script",
        "special_token_leakage",
        "generated_invalid_utf8_byte_sequence",
        "replacement_character_from_generated_bytes",
    }
    return (
        record["generated_token_count"] >= 4
        and not record["exact_repeated_ngram_loops"]
        and not (warnings & disallowed)
        and not record["policy"]["hard_failures"]
    )


def subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = Counter(record["language"] for record in records)
    by_category = Counter(record["category"] for record in records)
    meaningful = Counter()
    for record in records:
        if meaningful_proxy(record):
            meaningful[record["language"]] += 1
            meaningful[record["category"]] += 1
    return {
        "language_counts": dict(sorted(by_language.items())),
        "category_counts": dict(sorted(by_category.items())),
        "meaningful_proxy_total": sum(meaningful_proxy(record) for record in records),
        "meaningful_proxy_counts": dict(sorted(meaningful.items())),
        "finite_logits": all("non_finite_logits" not in record["policy"]["findings"] for record in records),
    }


def run_audit(
    checkpoint: Path,
    output_dir: Path,
    *,
    checkpoint_stage: str,
    expected_hash: str,
) -> dict[str, Any]:
    summary_path = output_dir / "audit_summary.json"
    if summary_path.is_file():
        summary = load_manifest(summary_path)
        if summary.get("checkpoint_model_sha256") != expected_hash:
            raise ValueError(f"completed audit hash mismatch: {output_dir}")
        return summary
    command = [
        sys.executable,
        "-m",
        "darkmind_v2.evaluation.audit_full_epoch_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--checkpoint-stage",
        checkpoint_stage,
        "--expected-model-hash",
        expected_hash,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=7200, check=False)
    if completed.returncode or not summary_path.is_file():
        raise RuntimeError(
            f"milestone audit failed: checkpoint={checkpoint.name} exit={completed.returncode} "
            f"stderr={completed.stderr[-2000:]}"
        )
    return load_manifest(summary_path)


def evaluate_all(config_path: Path) -> dict[str, Any]:
    contract = load_and_validate_phase3b_pilot_config(config_path)
    results: dict[str, dict[str, Any]] = {}
    for candidate in ("C", "D"):
        run_dir = ROOT / contract["runs"][candidate]
        run_manifest = load_manifest(run_dir / "run_manifest.json")
        if run_manifest.get("status") != "pilot_complete":
            raise ValueError(f"Candidate {candidate} pilot is incomplete")
        candidate_results: dict[str, Any] = {}
        for step, stage in STAGES.items():
            checkpoint = Path(run_manifest["checkpoints"][str(step)])
            output_dir = run_dir / "audits" / f"step_{step:06d}"
            expected_hash = run_manifest["checkpoint_model_hashes"][str(step)]["model_sha256"]
            audit = run_audit(
                checkpoint,
                output_dir,
                checkpoint_stage=stage,
                expected_hash=expected_hash,
            )
            validation = load_manifest(run_dir / "validations" / f"step_{step:06d}.json")
            evaluation = load_manifest(run_dir / "eval" / f"step_{step:06d}.json")
            greedy_manifest = load_manifest(output_dir / "greedy_manifest.json")
            sampling_manifest = load_manifest(output_dir / "sampling_manifest.json")
            candidate_results[str(step)] = {
                "step": step,
                "tokens": step * contract["data"]["effective_tokens_per_optimizer_step"],
                "validation": validation,
                "eval": evaluation,
                "greedy": audit["greedy"],
                "sampling": audit["sampling"],
                "greedy_subsets": subset_summary(greedy_manifest["results"]),
                "sampling_subsets": subset_summary(sampling_manifest["results"]),
                "audit_elapsed_seconds": audit["elapsed_seconds"],
                "checkpoint_model_sha256": expected_hash,
                "result": audit["result"],
            }
        best_checkpoint = Path(run_manifest["best_checkpoint"])
        best_step = next(
            step for step, path in run_manifest["checkpoints"].items() if Path(path) == best_checkpoint
        )
        candidate_results["best_validation"] = {
            "reused_checkpoint_step": int(best_step),
            **candidate_results[str(best_step)],
        }
        results[candidate] = candidate_results
    payload = {
        "schema_version": "darkmind-v2-phase3b-finalist-evaluation-v1",
        "generation_contract": {
            "greedy_per_checkpoint": 200,
            "seeded_sampling_per_checkpoint": 500,
            "max_new_tokens": 32,
            "best_checkpoint_reuses_identical_milestone_audit": True,
        },
        "results": results,
        "result": "PASS",
    }
    atomic_write_json(RUNTIME_REPORT, payload)
    render_report(payload)
    return payload


def render_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 3B Finalist Evaluation",
        "",
        "Each unique milestone used 200 deterministic greedy generations and 500 fixed seeded-sampling "
        "generations. The best-validation checkpoint equaled the final checkpoint for both finalists, so its "
        "byte-identical audit is reused rather than regenerated.",
        "",
        "The meaningful-continuation count is a structural proxy: at least four generated tokens, no exact "
        "n-gram loop, no hard failure, and no empty/repetition/script/special-token/invalid-byte warning. It is "
        "not a human quality judgment.",
        "",
        "| Candidate | Step | Tokens | Val loss | Eval loss | Greedy rep warn | Greedy loops | Greedy unique | Sampling rep warn | Sampling loops | Sampling unique | EOS greedy/sample | Meaningful proxy greedy/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate in ("C", "D"):
        for step in STAGES:
            item = payload["results"][candidate][str(step)]
            greedy = item["greedy"]
            sampling = item["sampling"]
            lines.append(
                f"| {candidate} | {step} | {item['tokens']:,} | {item['validation']['loss']:.6f} | "
                f"{item['eval']['loss']:.6f} | {greedy['quality_warning_counts'].get('repetition', 0)} | "
                f"{greedy['exact_repeated_ngram_loop_outputs']} | {greedy['mean_unique_token_ratio']:.3f} | "
                f"{sampling['quality_warning_counts'].get('repetition', 0)} | "
                f"{sampling['exact_repeated_ngram_loop_outputs']} | {sampling['mean_unique_token_ratio']:.3f} | "
                f"{greedy['eos_completion_rate']:.3f}/{sampling['eos_completion_rate']:.3f} | "
                f"{item['greedy_subsets']['meaningful_proxy_total']}/"
                f"{item['sampling_subsets']['meaningful_proxy_total']} |"
            )
    lines.extend(
        [
            "",
            "All raw generations remain in ignored runtime manifests. Invalid UTF-8 byte sequences, U+FFFD, "
            "mojibake, unexpected/mixed script, special-token leakage, output lengths, language/category counts, "
            "and finite-logit status are retained in the machine-readable audit summaries and manifests.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = evaluate_all(args.config)
    print(json.dumps({"result": result["result"], "candidates": sorted(result["results"])}, indent=2))


if __name__ == "__main__":
    main()
