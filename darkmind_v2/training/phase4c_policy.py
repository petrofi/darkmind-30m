"""Freeze and report the selected Phase 4C Base V1 production policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES
from darkmind_v2.training.phase4c_diagnostics import (
    INITIALIZATION_SEED,
    MODEL_INPUT,
    ORDER_INPUT,
    RUNTIME_ROOT,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    learning_rate_for_policy,
    sha256_file,
)
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
    ROOT,
)


V2_CONFIG = ROOT / "darkmind_v2" / "config" / "train_base_v1_production_100m_v2.json"
POLICY_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4c_corrected_training_policy.md"
STORAGE_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4b_runtime_storage_inventory.md"
ANALYSIS_PATH = RUNTIME_ROOT / "diagnostics" / "policy_analysis.json"
CONFIRMATION_RUN = RUNTIME_ROOT / "runs" / "base_v1_stage1_5m_v2_confirmation"
PHASE4B_RUNTIME = Path(r"C:\DarkMindRuntime\phase4b")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_v2_config(policy: dict[str, Any]) -> dict[str, Any]:
    schedule = dict(policy["schedule"])
    if policy != {
        "optimizer_grouping": "corrected_adamw_v1",
        "initialization_policy": "base_v1_standard_v1",
        "schedule": schedule,
        "sequence_order": "deterministic_stratified_v1",
    }:
        raise ValueError("selected Phase 4C policy escaped the approved dimensions")
    if schedule["name"] != "warmup_cosine_global" or schedule["peak_learning_rate"] != 0.0001:
        raise ValueError("selected Phase 4C schedule is not the approved global 1e-4 policy")
    lr_milestones = {
        "5m_step_610": learning_rate_for_policy(610, schedule),
        "25m_step_3051": learning_rate_for_policy(3051, schedule),
        "100m_step_12207": learning_rate_for_policy(12207, schedule),
    }
    payload = {
        "schema_version": "darkmind-v2-base-v1-production-100m-v2",
        "model_name": "darkmind-v2-base-v1",
        "model_config": str(MODEL_INPUT),
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_dir": str(TOKENIZER_INPUT),
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "corpus": {
            "tokenized_dir": str(TOKENIZED_INPUT),
            "corpus_hash": EXPECTED_CORPUS_HASH,
            "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
            "train_complete_sequence_tokens": 98_081_792,
            "train_tail_tokens": 328,
        },
        "runtime_root": str(RUNTIME_ROOT),
        "run_dir": str(CONFIRMATION_RUN),
        "sequence_order_manifest": str(ORDER_INPUT),
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_policy": "base_v1_standard_v1",
        "data": {
            "data_order_seed": INITIALIZATION_SEED,
            "sequence_order": "deterministic_stratified_v1",
            "sequence_length": 512,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_tokens_per_optimizer_step": 8192,
            "no_replacement": True,
            "no_wrap": True,
        },
        "optimizer": {
            "name": "AdamW",
            "grouping": "corrected_adamw_v1",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "gradient_clipping": 1.0,
        },
        "schedule": {**schedule, "learning_rate_milestones": lr_milestones},
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "gradient_checkpointing": False,
        "training_compile": False,
        "fused_optimizer": False,
        "evaluation_steps": [0, 64, 128, 192, 256, 384, 512, 610],
        "confirmation": {
            "optimizer_steps": 610,
            "tokens": 4_997_120,
            "forced_midpoint_restart_step": 305,
            "fresh_process_resume_required": True,
            "reuse_exploratory_run": False,
        },
        "continuation": {
            "resume_exact_checkpoint_state": True,
            "scheduler_restart": False,
            "next_optimizer_step_after_5m": 611,
            "next_learning_rate_after_5m": learning_rate_for_policy(611, schedule),
        },
        "stage_gates": {
            "5m": {"optimizer_steps": 610, "tokens": 4_997_120, "authorized": True},
            "25m": {"optimizer_steps": 3_051, "tokens": 24_993_792, "authorized": False},
            "100m": {"optimizer_steps": 12_207, "tokens": 99_999_744, "authorized": False},
        },
        "authorization": {
            "maximum_optimizer_steps": 610,
            "maximum_training_tokens": 4_997_120,
            "phase_25m_authorized": False,
            "phase_100m_authorized": False,
        },
    }
    payload["deterministic_content_hash"] = canonical_json_hash(payload)
    return payload


def validate_v2_config(config: dict[str, Any]) -> None:
    core = {key: value for key, value in config.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != config.get("deterministic_content_hash"):
        raise ValueError("V2 deterministic config hash mismatch")
    if config["model_config_sha256"] != EXPECTED_CONFIG_SHA256 or config["architecture_hash"] != EXPECTED_ARCHITECTURE_HASH:
        raise ValueError("V2 Base V1 identity changed")
    if config["tokenizer_hashes"] != EXPECTED_HASHES:
        raise ValueError("V2 frozen tokenizer identity changed")
    if config["corpus"]["corpus_hash"] != EXPECTED_CORPUS_HASH or config["corpus"]["tokenized_manifest_hash"] != EXPECTED_TOKENIZED_HASH:
        raise ValueError("V2 Corpus V3 identity changed")
    if config["optimizer"]["grouping"] != "corrected_adamw_v1":
        raise ValueError("V2 corrected AdamW grouping changed")
    schedule = config["schedule"]
    if schedule["name"] != "warmup_cosine_global" or schedule["peak_learning_rate"] != 0.0001:
        raise ValueError("V2 selected scheduler changed")
    if schedule["scheduler_horizon_optimizer_steps"] != 12_207 or schedule["scheduler_horizon_tokens"] != 99_999_744:
        raise ValueError("V2 global scheduler horizon changed")
    if schedule["learning_rate_milestones"] != {
        "5m_step_610": learning_rate_for_policy(610, schedule),
        "25m_step_3051": learning_rate_for_policy(3051, schedule),
        "100m_step_12207": learning_rate_for_policy(12207, schedule),
    }:
        raise ValueError("V2 LR milestones changed")
    if config["confirmation"] != {
        "optimizer_steps": 610,
        "tokens": 4_997_120,
        "forced_midpoint_restart_step": 305,
        "fresh_process_resume_required": True,
        "reuse_exploratory_run": False,
    }:
        raise ValueError("V2 confirmation contract changed")
    if Path(config["run_dir"]).resolve() != CONFIRMATION_RUN.resolve():
        raise ValueError("V2 confirmation run directory changed")
    if config["authorization"] != {
        "maximum_optimizer_steps": 610,
        "maximum_training_tokens": 4_997_120,
        "phase_25m_authorized": False,
        "phase_100m_authorized": False,
    }:
        raise ValueError("V2 authorization exceeds the 5M gate")
    if config["stage_gates"]["25m"]["authorized"] or config["stage_gates"]["100m"]["authorized"]:
        raise ValueError("V2 future stage authorization changed")


def prepare() -> dict[str, Any]:
    analysis = _load(ANALYSIS_PATH)
    if not analysis["stable_policy_found"] or not analysis["selected_authoritative_generation_complete"]:
        raise RuntimeError("stable exploratory policy and authoritative generation are required")
    config = build_v2_config(analysis["selected_policy"])
    validate_v2_config(config)
    if V2_CONFIG.exists():
        existing = _load(V2_CONFIG)
        if existing != config:
            raise FileExistsError("existing V2 config differs from selected Phase 4C policy")
    else:
        V2_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": "darkmind-v2-phase4c-policy-freeze-v1",
        "result": "PASS",
        "selected_arm": analysis["selected_arm"],
        "v2_config": str(V2_CONFIG),
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "confirmation_authorized": True,
        "phase_25m_authorized": False,
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _generation_text(audit: dict[str, Any]) -> str:
    greedy = audit["greedy"]
    sampling = audit["sampling"]
    return (
        f"greedy repetition {greedy['quality_warning_counts'].get('repetition', 0)}/200, "
        f"loops {greedy['exact_repeated_ngram_loop_outputs']}/200; "
        f"sampling repetition {sampling['quality_warning_counts'].get('repetition', 0)}/500, "
        f"loops {sampling['exact_repeated_ngram_loop_outputs']}/500"
    )


def finalize() -> dict[str, Any]:
    config = _load(V2_CONFIG)
    validate_v2_config(config)
    analysis = _load(ANALYSIS_PATH)
    confirmation = _load(CONFIRMATION_RUN / "evaluation_summary.json")
    resume = _load(CONFIRMATION_RUN / "process_restart_validation.json")
    if confirmation["stability"] != "stable" or not resume["all_checks_pass"]:
        raise RuntimeError("immutable Phase 4C confirmation did not pass")
    authoritative = confirmation["authoritative_final"]
    if authoritative is None or authoritative["result"] != "PASS":
        raise RuntimeError("immutable confirmation authoritative generation is incomplete")
    lines = [
        "# DarkMind v2 Phase 4C Corrected Training Policy",
        "",
        f"Final classification: **PASS**",
        f"Selected exploratory arm: `{analysis['selected_arm']}`",
        f"V2 config SHA-256: `{sha256_file(V2_CONFIG)}`",
        "",
        "## Diagnosis",
        "",
        "The Phase 3B pilot used a 5M-local cosine schedule, while Phase 4A/4B used a 100M-global schedule that stayed near peak LR through step 610. Phase 4B peak LR 1.5e-4 remained too high. At 1e-4, the Base V1 loss trajectory is stable without rebound.",
        "",
        "The prior optimizer placed bias, LayerNorm, token/position embeddings, and the tied LM-head weight in the decay group. The tied tensor appeared once, so there was no duplicate optimizer state. V2 uses decay only for approved matrix weights and no decay for bias, normalization, and embeddings.",
        "",
        "The staged first-5M decay remained stable but underfit the global 1e-4 schedule. Depth-scaled residual initialization improved loss but failed the diagnostic-health gate because residual growth reached 8.30x its initialization ratio and 96.7% of steps clipped. Standard Base V1 initialization is retained.",
        "",
        "## Exploratory arms",
        "",
        "| Arm | Final val | Final eval | Val/eval rebound | Clip fraction | Residual growth | Stability |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in analysis["arms"].items():
        lines.append(
            f"| {name} | {item['final_validation_loss']:.6f} | {item['final_eval_loss']:.6f} | "
            f"{item['validation_rebound_percent']:.3f}% / {item['eval_rebound_percent']:.3f}% | "
            f"{item['clipped_step_fraction']:.3f} | {item['maximum_residual_ratio_growth_multiple_from_step0']:.3f}x | {item['stability']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen V2 policy",
            "",
            "- Initialization: `base_v1_standard_v1`, seed `20260712`.",
            "- Optimizer: AdamW, corrected decay/no-decay groups, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.1, gradient clip 1.0.",
            "- Schedule: 100M-global warmup cosine, peak `0.0001`, minimum `0.00003`, warmup 100 steps, no restart.",
            f"- Applied LR at 5M/25M/100M: `{config['schedule']['learning_rate_milestones']['5m_step_610']:.12f}` / `{config['schedule']['learning_rate_milestones']['25m_step_3051']:.12f}` / `{config['schedule']['learning_rate_milestones']['100m_step_12207']:.12f}`.",
            "- Sequence order: `deterministic_stratified_v1`, no replacement and no wrap.",
            "",
            "## Immutable confirmation",
            "",
            f"- Initialization hash: `{confirmation['initialization_hash']}`.",
            f"- Steps/tokens: `{confirmation['optimizer_steps']}` / `{confirmation['training_tokens']}`.",
            f"- Validation: `{confirmation['initial_validation_loss']:.6f}` to `{confirmation['final_validation_loss']:.6f}` ({confirmation['validation_improvement_percent']:.3f}% improvement), rebound `{confirmation['validation_rebound_percent']:.3f}%`.",
            f"- Eval: `{confirmation['initial_eval_loss']:.6f}` to `{confirmation['final_eval_loss']:.6f}` ({confirmation['eval_improvement_percent']:.3f}% improvement), rebound `{confirmation['eval_rebound_percent']:.3f}%`.",
            f"- Fresh process restart: `PASS`, step 305 to 306 continuity; PIDs `{resume['process_ids']}`.",
            f"- Authoritative generation: {_generation_text(authoritative)}.",
            "- Generation quality remains diagnostic at 5M and is not a chatbot-quality claim.",
            "",
            "The exact step-610 confirmation checkpoint is resume-capable. A 25M continuation is recommended only after explicit user approval; it was not started here.",
            "",
        ]
    )
    POLICY_REPORT.write_text("\n".join(lines), encoding="utf-8")

    arm_sizes = {
        path.name: directory_size(path)
        for path in sorted((RUNTIME_ROOT / "runs").iterdir())
        if path.is_dir()
    }
    input_reference = _load(RUNTIME_ROOT / "inputs" / "shared_input_reference.json")
    total_phase4c = directory_size(RUNTIME_ROOT)
    storage_lines = [
        "# DarkMind v2 Phase 4B and Phase 4C Runtime Storage Inventory",
        "",
        f"Phase 4B runtime preserved at `C:\\DarkMindRuntime\\phase4b`: {directory_size(PHASE4B_RUNTIME):,} bytes.",
        f"Phase 4C runtime at `C:\\DarkMindRuntime\\phase4c`: {total_phase4c:,} bytes.",
        f"Immutable input bytes shared from Phase 4B rather than duplicated: {input_reference['phase4b_input_bytes_reused']:,} bytes.",
        f"Phase 4C copied immutable input bytes: {input_reference['copied_input_bytes']:,} bytes.",
        "",
        "| Phase 4C run | Bytes |",
        "|---|---:|",
    ]
    storage_lines.extend(f"| {name} | {size:,} |" for name, size in arm_sizes.items())
    storage_lines.extend(
        [
            "",
            f"Resume-capable confirmation checkpoint: `{confirmation['checkpoints']['610']}` ({directory_size(Path(confirmation['checkpoints']['610'])):,} bytes).",
            "",
            "Required for a future 25M continuation: the frozen V2 config, Phase 4B immutable inputs, deterministic order manifest, final step-610 model, optimizer/scheduler/RNG resume state, checkpoint metadata, validation manifests, and hash reports.",
            "",
            "Safe to move to an external SSD only after a user-approved copy and hash verification: completed exploratory model-only checkpoints, raw generation manifests, diagnostics, and older Phase 4A/4B evidence. Keep summaries and hashes beside archives.",
            "",
            "Safe to delete only after hash-verified archival and explicit approval: redundant exploratory model-only checkpoint payloads. Never delete the frozen inputs, final confirmation resume checkpoint, manifests, or reports before a validated archive exists.",
            "",
            "No file was deleted, moved, or archived in Phase 4C.",
            "",
        ]
    )
    STORAGE_REPORT.write_text("\n".join(storage_lines), encoding="utf-8")
    return {
        "schema_version": "darkmind-v2-phase4c-finalization-v1",
        "result": "PASS",
        "classification": "PASS",
        "v2_config_sha256": sha256_file(V2_CONFIG),
        "confirmation_checkpoint": confirmation["checkpoints"]["610"],
        "phase4c_runtime_bytes": total_phase4c,
        "phase_25m_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finalize"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else finalize(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
