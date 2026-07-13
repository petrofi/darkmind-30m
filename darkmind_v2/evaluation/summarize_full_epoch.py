"""Build Phase 2C metrics, balanced samples, diagnosis, and next-model reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash


MILESTONES = (0, 256, 717, 1434, 2150, 2867)
CHECKPOINT_TOKENS = {
    0: 0,
    256: 1_048_576,
    717: 2_936_832,
    1434: 5_873_664,
    2150: 8_806_400,
    2867: 11_743_232,
}
AUDIT_DIRS = {
    step: f"public_preview_v2_step_{step:06d}" for step in MILESTONES
}
STAGE1_RUN = Path("darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def warning_rate(summary: dict[str, Any], warning: str) -> float:
    return summary["quality_warning_counts"].get(warning, 0) / summary["generations"]


def build_summary(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest.json")
    metrics = [item for item in load_metrics(run_dir / "metrics.jsonl") if item.get("event") == "optimizer_step"]
    by_step = {int(item["optimizer_step"]): item for item in metrics}
    milestones: dict[str, Any] = {}
    for step in MILESTONES:
        validation = load_json(run_dir / "validations" / f"step_{step:06d}.json")
        evaluation = load_json(run_dir / "eval" / f"step_{step:06d}.json")
        audit = load_json(run_dir / "evaluations" / AUDIT_DIRS[step] / "audit_summary.json")
        training = by_step.get(step)
        milestones[str(step)] = {
            "optimizer_step": step,
            "consumed_tokens": CHECKPOINT_TOKENS[step],
            "corpus_coverage_percent": 100 * CHECKPOINT_TOKENS[step] / manifest["train_corpus_tokens"],
            "raw_training_loss": training.get("raw_training_loss") if training else None,
            "smoothed_training_loss": training.get("smoothed_training_loss") if training else None,
            "learning_rate": training.get("learning_rate") if training else None,
            "gradient_norm": training.get("gradient_norm") if training else None,
            "validation_loss": validation["loss"],
            "eval_loss": evaluation["loss"],
            "perplexity": validation["perplexity"],
            "greedy": audit["greedy"],
            "sampling": audit["sampling"],
            "checkpoint_model_sha256": audit["checkpoint_model_sha256"],
        }

    stage1_eval = load_json(STAGE1_RUN / "evaluations" / "byte_trace_policy_v2" / "stage1_evaluation.json")
    stage1_audit = load_json(STAGE1_RUN / "evaluations" / "public_preview_v2" / "audit_summary.json")
    throughputs = [float(item["tokens_per_second"]) for item in metrics]
    segment_wall = sum(
        float(load_json(run_dir / name)["elapsed_seconds"])
        for name in ("segment_to_step_001434.json", "segment_to_step_002867.json")
    )
    final = milestones["2867"]
    stage1 = {
        "trained_tokens": 1_048_576,
        "optimizer_steps": 256,
        "validation_loss": stage1_eval["summary"]["final_validation_loss"],
        "eval_loss": stage1_eval["summary"]["final_eval_loss"],
        "perplexity": stage1_eval["summary"]["final_perplexity"],
        "greedy_repetition_warning_rate": warning_rate(stage1_audit["greedy"], "repetition"),
        "greedy_exact_loop_rate": stage1_audit["greedy"]["exact_repeated_ngram_loop_outputs"] / 200,
        "greedy_eos_completion_rate": stage1_audit["greedy"]["eos_completion_rate"],
        "sampling_repetition_warning_rate": warning_rate(stage1_audit["sampling"], "repetition"),
        "sampling_exact_loop_rate": stage1_audit["sampling"]["exact_repeated_ngram_loop_outputs"] / 500,
        "sampling_eos_completion_rate": stage1_audit["sampling"]["eos_completion_rate"],
    }
    full_epoch = {
        "trained_tokens": 11_743_232,
        "optimizer_steps": 2867,
        "coverage_percent": manifest["coverage_percent"],
        "excluded_tail_tokens": manifest["excluded_tail_tokens"],
        "final_train_loss": final["raw_training_loss"],
        "final_smoothed_train_loss": final["smoothed_training_loss"],
        "validation_loss": final["validation_loss"],
        "eval_loss": final["eval_loss"],
        "perplexity": final["perplexity"],
        "greedy_repetition_warning_rate": warning_rate(final["greedy"], "repetition"),
        "greedy_exact_loop_rate": final["greedy"]["exact_repeated_ngram_loop_outputs"] / 200,
        "greedy_eos_completion_rate": final["greedy"]["eos_completion_rate"],
        "sampling_repetition_warning_rate": warning_rate(final["sampling"], "repetition"),
        "sampling_exact_loop_rate": final["sampling"]["exact_repeated_ngram_loop_outputs"] / 500,
        "sampling_eos_completion_rate": final["sampling"]["eos_completion_rate"],
        "meaningful_turkish_rate_manual": 2 / 60,
        "meaningful_english_rate_manual": 0.0,
        "factual_success_rate_manual": 0.0,
        "technical_code_success_rate_manual": 0.0,
        "public_hard_failures": final["greedy"]["hard_failure_total"] + final["sampling"]["hard_failure_total"],
    }
    core = {
        "schema_version": "darkmind-v2-phase2c-capacity-summary-v1",
        "run": manifest,
        "milestones": milestones,
        "stage1_baseline": stage1,
        "full_epoch": full_epoch,
        "throughput": {
            "active_mean_tokens_per_second": statistics.mean(throughputs),
            "active_median_tokens_per_second": statistics.median(throughputs),
            "training_segment_wall_seconds": segment_wall,
            "canonical_full_wall_tokens_per_second": manifest["target_training_tokens"] / segment_wall,
            "peak_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in metrics),
            "peak_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in metrics),
        },
        "resume": load_json(run_dir / "midpoint_process_restart_validation.json"),
        "best_checkpoint": {
            "path": manifest["best_checkpoint"],
            "validation_loss": manifest["best_validation_loss"],
            "model_sha256": final["checkpoint_model_sha256"],
        },
        "diagnosis": {
            "dominant": "capacity-limited",
            "secondary": "data-scale-and-composition-limited",
            "overfitting_detected": False,
            "further_tiny_training_justified": False,
            "public_release_eligible": False,
            "model_weight_license_resolved": False,
        },
    }
    return {**core, "deterministic_content_hash": canonical_json_hash(core)}


def sample_score(record: dict[str, Any]) -> float:
    warnings = set(record["policy"]["warnings"])
    return (
        float(record["unique_token_ratio"])
        + 0.35 * int(record["eos_completed"])
        + 0.5 * int(not record["exact_repeated_ngram_loops"])
        - 1.0 * int("repetition" in warnings)
        - 0.4 * int(record["generated_token_count"] < 4)
    )


def balanced_band(records: list[dict[str, Any]], count: int) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(records, key=lambda item: (-sample_score(item), item["prompt_id"]))
    stronger_count = max(1, math.ceil(count / 3))
    weak_count = max(1, math.floor(count / 3))
    average_count = count - stronger_count - weak_count
    selected: list[tuple[str, dict[str, Any]]] = []
    selected.extend(("relative stronger", item) for item in ordered[:stronger_count])
    middle_start = max(stronger_count, (len(ordered) - average_count) // 2)
    selected.extend(("average", item) for item in ordered[middle_start : middle_start + average_count])
    selected.extend(("weak", item) for item in ordered[-weak_count:])
    return selected


def fenced(text: str) -> str:
    return text.replace("```", "` ` `")


def write_samples_report(run_dir: Path, output: Path) -> None:
    manifest = load_json(run_dir / "evaluations" / AUDIT_DIRS[2867] / "greedy_manifest.json")
    records = manifest["results"]
    groups = [
        ("Turkish ordinary continuations", 15, [item for item in records if item["language"] == "tr" and item["category"] == "ordinary_text"]),
        ("English ordinary continuations", 15, [item for item in records if item["language"] == "en" and item["category"] == "ordinary_text"]),
        ("Factual continuations", 5, [item for item in records if item["category"] == "factual_encyclopedic"]),
        ("Technical and code continuations", 5, [item for item in records if item["category"] in {"technical", "code_structured"}]),
    ]
    lines = [
        "# DarkMind v2 Phase 2C Tiny Full-Epoch Samples",
        "",
        "These 40 examples are selected deterministically across quality bands from the final 200-prompt greedy audit. ",
        "`Relative stronger` means stronger only within this weak checkpoint; it is not a fluency claim. No output was edited or sanitized.",
        "",
        "## Review Summary",
        "",
        "- Basic grammatical structure: occasional short fragments, especially in English; not reliable.",
        "- Topical continuation: generally absent.",
        "- Language consistency: strong for greedy decoding, but content quality remains weak.",
        "- Lexical diversity: poor under greedy decoding; seeded sampling is more diverse but still unreliable.",
        "- Sentence completion: two short Turkish completions are minimally plausible; most outputs loop or terminate as fragments.",
        "- Factual reliability: no reviewed factual continuation is usable.",
        "- Code usefulness: no reviewed technical/code continuation is usable.",
        "",
        "Checkpoint: `step_002867_tokens_011743232`; decoding: greedy, EOS-aware, max 32 new tokens.",
        "",
    ]
    index = 0
    for title, count, candidates in groups:
        lines.extend([f"## {title}", ""])
        for band, item in balanced_band(candidates, count):
            index += 1
            warnings = ", ".join(item["policy"]["warnings"]) or "none"
            lines.extend([
                f"### {index}. {item['prompt_id']} - {band}",
                "",
                "Prompt:",
                "```text",
                fenced(item["prompt"]),
                "```",
                "Raw continuation:",
                "```text",
                fenced(item["output"]),
                "```",
                f"Warnings: `{warnings}`; exact loop: `{bool(item['exact_repeated_ngram_loops'])}`; "
                f"unique-token ratio: `{item['unique_token_ratio']:.4f}`; EOS: `{item['eos_completed']}`.",
                "",
            ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_diagnosis_report(summary: dict[str, Any], output: Path) -> None:
    stage1 = summary["stage1_baseline"]
    full = summary["full_epoch"]
    lines = [
        "# DarkMind v2 Phase 2C Tiny Capacity Diagnosis",
        "",
        "## Experiment",
        "",
        "This is one deterministic epoch-equivalent with 99.9915% train-token coverage. It restarted from the same seed and did not reuse Stage-1 weights.",
        "The run consumed 11,743,232 of 11,744,226 train tokens in 2,867 optimizer steps; the final 994-token tail was excluded without wraparound.",
        "",
        "## Loss Progression",
        "",
        "| Step | Tokens | Validation loss | Eval loss | Perplexity |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in MILESTONES:
        item = summary["milestones"][str(step)]
        lines.append(
            f"| {step:,} | {item['consumed_tokens']:,} | {item['validation_loss']:.6f} | "
            f"{item['eval_loss']:.6f} | {item['perplexity']:.3f} |"
        )
    lines.extend([
        "",
        "Validation and eval loss improve throughout the run. There is no measured train/validation divergence, so classic overfitting is not the dominant failure.",
        "",
        "## Stage-1 Comparison",
        "",
        "| Metric | Stage-1 | Full epoch-equivalent | Change |",
        "|---|---:|---:|---:|",
        f"| Validation loss | {stage1['validation_loss']:.6f} | {full['validation_loss']:.6f} | {full['validation_loss'] - stage1['validation_loss']:+.6f} |",
        f"| Eval loss | {stage1['eval_loss']:.6f} | {full['eval_loss']:.6f} | {full['eval_loss'] - stage1['eval_loss']:+.6f} |",
        f"| Greedy repetition warning | {pct(stage1['greedy_repetition_warning_rate'])} | {pct(full['greedy_repetition_warning_rate'])} | {pct(full['greedy_repetition_warning_rate'] - stage1['greedy_repetition_warning_rate'])} points |",
        f"| Greedy exact n-gram loop | {pct(stage1['greedy_exact_loop_rate'])} | {pct(full['greedy_exact_loop_rate'])} | {pct(full['greedy_exact_loop_rate'] - stage1['greedy_exact_loop_rate'])} points |",
        f"| Greedy EOS completion | {pct(stage1['greedy_eos_completion_rate'])} | {pct(full['greedy_eos_completion_rate'])} | {pct(full['greedy_eos_completion_rate'] - stage1['greedy_eos_completion_rate'])} points |",
        f"| Sampling repetition warning | {pct(stage1['sampling_repetition_warning_rate'])} | {pct(full['sampling_repetition_warning_rate'])} | {pct(full['sampling_repetition_warning_rate'] - stage1['sampling_repetition_warning_rate'])} points |",
        f"| Sampling exact n-gram loop | {pct(stage1['sampling_exact_loop_rate'])} | {pct(full['sampling_exact_loop_rate'])} | {pct(full['sampling_exact_loop_rate'] - stage1['sampling_exact_loop_rate'])} points |",
        f"| Sampling EOS completion | {pct(stage1['sampling_eos_completion_rate'])} | {pct(full['sampling_eos_completion_rate'])} | {pct(full['sampling_eos_completion_rate'] - stage1['sampling_eos_completion_rate'])} points |",
        "",
        "## Human Quality Review",
        "",
        "- Meaningful Turkish ordinary continuations: 2/60 (3.3%), both short and generic.",
        "- Meaningful English ordinary continuations: 0/50 under a strict topical-continuation criterion.",
        "- Factual success: 0%; outputs are unreliable and often numeric loops.",
        "- Technical/code success: 0%; repeated symbols, fragments, or irrelevant continuations dominate.",
        "- Greedy language consistency improved mechanically, but this did not produce useful semantics.",
        "",
        "## Diagnosis",
        "",
        "**Outcome B - capacity-limited, with data scale/composition as a secondary constraint.**",
        "",
        "The model has 9,369,088 parameters, but 6,144,000 (65.58%) belong to the tied 24k embedding table. Only 3,225,088 parameters remain for attention, MLP, normalization, and positions.",
        "Loss improvement proves the pipeline learns, but generation remains largely repetitive and incoherent. Greedy repetition falls by 14 points, while exact loops worsen by 9 points and EOS completion falls by 16 points.",
        "The current corpus also imprints numeric and Python/C-symbol patterns on weak generations, so a larger model should not be trained on this 13.05M-token corpus alone.",
        "",
        "A second tiny epoch is not technically justified from this evidence. It could lower loss further, but the dominant representational bottleneck and unstable generation metrics are unlikely to be solved by repeating the same data.",
        "",
        "## Release Decision",
        "",
        "Public research-preview eligibility remains **FAIL**. Pipeline hard failures are zero, but quality is inadequate and the model-weight distribution license remains unresolved.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_next_model_report(output: Path) -> None:
    lines = [
        "# DarkMind v2 Phase 2C Next-Model Requirements",
        "",
        "## Evidence-Derived Floor",
        "",
        "- Minimum non-vocabulary parameters: 45M.",
        "- Maximum vocabulary/embedding share: 25%; prefer 20% or lower.",
        "- Frozen vocabulary: 24,000 with tied input/output embeddings.",
        "- Initial context length: 1,024; extend only after throughput and data-length analysis.",
        "- Corpus: at least 250M-500M high-quality deduplicated tokens before architecture selection; target training tokens should follow the table below.",
        "- Turkish/English balance, factual prose, and code/technical sources must be measured separately.",
        "",
        "## Candidate Classes",
        "",
        "These are parameter-count anchors, not a selected winner.",
        "",
        "| Class | d_model | Layers | Heads | Exact params | Non-vocab params | Vocab share | Est. BF16 train VRAM | Est. active tok/s | Target train tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| 60M | 512 | 15 | 8 | 60,099,072 | 47,811,072 | 20.45% | 2.0-3.0 GiB | 9k-13k | 1.2B+ |",
        "| 80M | 640 | 13 | 10 | 80,022,400 | 64,662,400 | 19.19% | 2.6-3.8 GiB | 7k-10k | 1.6B+ |",
        "| 100M | 640 | 17 | 10 | 99,716,480 | 84,356,480 | 15.40% | 3.2-4.8 GiB | 5.5k-8k | 2.0B+ |",
        "| 120M | 768 | 15 | 12 | 125,538,048 | 107,106,048 | 14.68% | 4.0-6.0 GiB | 4.5k-7k | 2.5B+ |",
        "",
        "## Training-System Requirements",
        "",
        "- Preserve deterministic uint16 shards, exact resume state, no hidden wraparound, and milestone public-preview audits.",
        "- Gradient checkpointing: optional for 60M/80M, recommended for 100M, required for the 120M class on an 8 GiB GPU unless calibration proves otherwise.",
        "- Use BF16, fused AdamW only after deterministic equivalence testing, and activation-memory measurements at context 1,024.",
        "- Benchmark each class with the same effective token batch before selecting by throughput, VRAM, validation scaling, and generation quality.",
        "- Do not train any candidate until corpus licensing, model-weight licensing, and a larger balanced corpus are ready.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/runs/tiny_full_epoch_seed20260712_v1"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("darkmind_v2/data/phase2c/runs/tiny_full_epoch_seed20260712_v1/evaluations/full_epoch_summary.json"),
    )
    parser.add_argument(
        "--samples-report",
        type=Path,
        default=Path("darkmind_v2/reports/phase2c_tiny_full_epoch_samples.md"),
    )
    parser.add_argument(
        "--diagnosis-report",
        type=Path,
        default=Path("darkmind_v2/reports/phase2c_tiny_capacity_diagnosis.md"),
    )
    parser.add_argument(
        "--next-model-report",
        type=Path,
        default=Path("darkmind_v2/reports/phase2c_next_model_requirements.md"),
    )
    args = parser.parse_args()
    for output in (args.summary, args.samples_report, args.diagnosis_report, args.next_model_report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Phase 2C report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.run_dir)
    atomic_write_json(args.summary, summary)
    write_samples_report(args.run_dir, args.samples_report)
    write_diagnosis_report(summary, args.diagnosis_report)
    write_next_model_report(args.next_model_report)
    print(json.dumps({
        "result": "PASS",
        "summary": str(args.summary),
        "samples_report": str(args.samples_report),
        "diagnosis_report": str(args.diagnosis_report),
        "next_model_report": str(args.next_model_report),
        "diagnosis": summary["diagnosis"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
