"""Finalize Phase 4B policy and storage reports without exceeding the 5M gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.training.phase4b_factorial import (
    INITIALIZATION_SEED,
    RUNTIME_ROOT,
    STAGE1_STEPS,
    STAGE1_TOKENS,
)
from darkmind_v2.training.phase4b_runtime import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_HASHES,
    EXPECTED_TOKENIZED_HASH,
    INPUT_ROOT,
    ORDER_ROOT,
    ROOT,
    TOKENIZED_INPUT,
    TOKENIZER_INPUT,
    atomic_write_json,
    sha256_file,
)


V2_CONFIG = ROOT / "darkmind_v2" / "config" / "train_base_v1_production_100m_v2.json"
POLICY_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4b_corrected_training_policy.md"
STORAGE_REPORT = ROOT / "darkmind_v2" / "reports" / "phase4b_runtime_storage_inventory.md"
ANALYSIS_PATH = RUNTIME_ROOT / "factorial_analysis.json"
CONFIRMATION_RUN = RUNTIME_ROOT / "runs" / "base_v1_stage1_5m_v2_confirmation"


def build_v2_config(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("peak_learning_rate") not in {0.00015, 0.0002, 0.0003}:
        raise ValueError("V2 policy has an unapproved peak learning rate")
    if policy.get("sequence_order") not in {"legacy_order_v1", "deterministic_stratified_v1"}:
        raise ValueError("V2 policy has an unapproved sequence order")
    payload = {
        "schema_version": "darkmind-v2-base-v1-production-100m-v2",
        "model_name": "darkmind-v2-base-v1",
        "model_config": str(INPUT_ROOT / "model" / "model_base_v1.json"),
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_dir": str(TOKENIZER_INPUT),
        "tokenizer_hashes": EXPECTED_HASHES,
        "corpus": {
            "tokenized_dir": str(TOKENIZED_INPUT),
            "corpus_hash": EXPECTED_CORPUS_HASH,
            "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
            "train_complete_sequence_tokens": 98_081_792,
            "train_tail_tokens": 328,
        },
        "runtime_root": str(RUNTIME_ROOT),
        "run_dir": str(CONFIRMATION_RUN),
        "sequence_order_manifest": str(ORDER_ROOT / f"{policy['sequence_order']}.json"),
        "initialization_seed": INITIALIZATION_SEED,
        "data": {
            "data_order_seed": 20260712,
            "sequence_order": policy["sequence_order"],
            "sequence_length": 512,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_tokens_per_optimizer_step": 8192,
            "no_replacement": True,
            "no_wrap": True,
        },
        "optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "gradient_clipping": 1.0,
        },
        "schedule": {
            "name": "warmup_cosine",
            "peak_learning_rate": policy["peak_learning_rate"],
            "minimum_learning_rate": 0.00003,
            "warmup_optimizer_steps": 100,
            "scheduler_horizon_optimizer_steps": 12_207,
            "scheduler_horizon_tokens": 99_999_744,
            "scheduler_restart": False,
        },
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "gradient_checkpointing": False,
        "training_compile": False,
        "fused_optimizer": False,
        "confirmation": {
            "optimizer_steps": STAGE1_STEPS,
            "tokens": STAGE1_TOKENS,
            "forced_midpoint_restart_step": 305,
            "fresh_process_resume_required": True,
            "reuse_factorial_run": False,
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


def validate_confirmation_requirements(config: dict[str, Any]) -> None:
    confirmation = config.get("confirmation", {})
    if config.get("schema_version") != "darkmind-v2-base-v1-production-100m-v2":
        raise ValueError("confirmation requires a frozen V2 production config")
    if confirmation != {
        "optimizer_steps": 610,
        "tokens": 4_997_120,
        "forced_midpoint_restart_step": 305,
        "fresh_process_resume_required": True,
        "reuse_factorial_run": False,
    }:
        raise ValueError("confirmation run contract changed")
    if Path(config["run_dir"]).resolve() != CONFIRMATION_RUN.resolve():
        raise ValueError("confirmation run must use its dedicated external directory")
    if config["authorization"]["maximum_optimizer_steps"] != 610 or config["authorization"]["phase_25m_authorized"]:
        raise ValueError("confirmation authorization exceeds the 5M gate")


def validate_25m_resume_compatibility(config: dict[str, Any]) -> None:
    if config["model_config_sha256"] != EXPECTED_CONFIG_SHA256 or config["architecture_hash"] != EXPECTED_ARCHITECTURE_HASH:
        raise ValueError("V2 Base V1 identity changed")
    if config["tokenizer_hashes"] != EXPECTED_HASHES:
        raise ValueError("V2 frozen tokenizer identity changed")
    if config["corpus"]["corpus_hash"] != EXPECTED_CORPUS_HASH or config["corpus"]["tokenized_manifest_hash"] != EXPECTED_TOKENIZED_HASH:
        raise ValueError("V2 Corpus V3 identity changed")
    if config["schedule"]["scheduler_horizon_optimizer_steps"] != 12_207 or config["schedule"]["scheduler_horizon_tokens"] != 99_999_744:
        raise ValueError("V2 100M scheduler horizon changed")
    if config["stage_gates"]["25m"] != {"optimizer_steps": 3_051, "tokens": 24_993_792, "authorized": False}:
        raise ValueError("V2 25M resume gate changed")
    if config["stage_gates"]["100m"] != {"optimizer_steps": 12_207, "tokens": 99_999_744, "authorized": False}:
        raise ValueError("V2 100M resume gate changed")


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _generation_row(summary: dict[str, Any]) -> str:
    authoritative = summary["authoritative_final"]
    return (
        f"{authoritative['greedy']['quality_warning_counts'].get('repetition', 0)}/200 repeated, "
        f"{authoritative['greedy']['exact_repeated_ngram_loop_outputs']}/200 loops; "
        f"{authoritative['sampling']['quality_warning_counts'].get('repetition', 0)}/500 sampled repeated, "
        f"{authoritative['sampling']['exact_repeated_ngram_loop_outputs']}/500 sampled loops"
    )


def write_policy_report(analysis: dict[str, Any]) -> None:
    lines = [
        "# DarkMind v2 Phase 4B Corrected Training Policy",
        "",
        f"Stable factorial policy found: **{'YES' if analysis['stable_policy_found'] else 'NO'}**",
        "",
        "No policy is frozen unless it passes integrity, final validation/eval improvement, rebound, sustained-divergence, and generation-health requirements.",
        "",
        "| Arm | Order | Peak LR | Final validation | Final eval | Rebound val/eval | Stability | Final generation |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for name, item in analysis["arms"].items():
        lines.append(
            f"| {name} | {item['sequence_order']} | {item['peak_learning_rate']:.5f} | "
            f"{item['final_validation_loss']:.6f} | {item['final_eval_loss']:.6f} | "
            f"{item['validation_rebound_percent']:.3f}% / {item['eval_rebound_percent']:.3f}% | "
            f"{item['stability']} | {_generation_row(item)} |"
        )
    lines.extend(
        [
            "",
            "The lower learning rate and stratified order both improve final loss, but every arm remains unstable. The combined arm has only a 3-4% loss rebound, yet it shows three consecutive worsening evaluations and complete greedy loop collapse.",
            "",
            "The predeclared optional 0.0002 trigger did not fire because no 0.00015 arm was stable and neither lower-LR arm had its best checkpoint at step 610 with monotonic final improvement.",
            "",
            "`train_base_v1_production_100m_v2.json` was not created. No final confirmation run or future resume checkpoint is authorized.",
            "",
            "Recommended next diagnosis: optimizer dynamics, schedule shape/warmup transition, precision behavior, initialization scale, and architecture/training-policy assumptions. Frozen Base V1, tokenizer, and Corpus V3 remain unmodified.",
            "",
        ]
    )
    POLICY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_storage_report() -> dict[str, Any]:
    phase3c = directory_size(ROOT / "darkmind_v2" / "data" / "phase3c")
    phase3c1 = directory_size(ROOT / "darkmind_v2" / "data" / "phase3c1")
    phase4a_checkpoints = directory_size(
        ROOT / "darkmind_v2" / "data" / "phase4a" / "runs" / "base_v1_stage1_5m_seed20260712_v1" / "checkpoints"
    )
    relocation = json.loads((INPUT_ROOT / "runtime_relocation_manifest.json").read_text(encoding="utf-8"))
    arm_sizes = {
        name: directory_size(RUNTIME_ROOT / "runs" / name)
        for name in (
            "arm_a_legacy_lr3e4",
            "arm_b_legacy_lr15e5",
            "arm_c_stratified_lr3e4",
            "arm_d_stratified_lr15e5",
        )
    }
    total_runtime = directory_size(RUNTIME_ROOT)
    payload = {
        "schema_version": "darkmind-v2-phase4b-runtime-storage-inventory-v1",
        "result": "PASS",
        "original_phase3c_bytes": phase3c + phase3c1,
        "original_phase3c_components": {"phase3c": phase3c, "phase3c1": phase3c1},
        "minimal_copied_input_bytes": relocation["total_bytes"],
        "phase4a_checkpoint_bytes": phase4a_checkpoints,
        "phase4b_arm_bytes": arm_sizes,
        "total_new_runtime_bytes": total_runtime,
        "future_25m_resume_checkpoint": None,
        "stable_policy_found": False,
    }
    atomic_write_json(RUNTIME_ROOT / "storage_inventory.json", payload)
    lines = [
        "# DarkMind v2 Phase 4B Runtime Storage Inventory",
        "",
        f"Original Phase 3C runtime: {payload['original_phase3c_bytes']:,} bytes",
        f"Minimal copied immutable inputs: {payload['minimal_copied_input_bytes']:,} bytes",
        f"Phase 4A checkpoints: {payload['phase4a_checkpoint_bytes']:,} bytes",
        f"Total new Phase 4B runtime: {payload['total_new_runtime_bytes']:,} bytes",
        "",
        "| Phase 4B arm | Bytes |",
        "|---|---:|",
    ]
    for name, size in arm_sizes.items():
        lines.append(f"| {name} | {size:,} |")
    lines.extend(
        [
            "",
            "Must not be deleted: original Phase 4A evidence, original Corpus V3 runtime, relocated immutable inputs and validation manifests, factorial summaries, order manifests, raw audits, and storage/factor analysis manifests.",
            "",
            "Safe to archive later after an explicit user decision: model-only checkpoints from the four failed exploratory arms. Retain their hashes, summaries, metrics, and raw audit manifests with any archive.",
            "",
            "Required for further diagnosis: `inputs/corpus_v3_tokenized`, frozen tokenizer package, Base V1 config, both order manifests, validation passes, factorial configs, and all arm summaries/audits. No checkpoint is approved for 25M continuation.",
            "",
            "Recommended external SSD layout: `DarkMindRuntime/phase4b/inputs`, `runs/failed_factorial`, `reports`, `exports`, `logs`, and `archive/phase4a_phase4b_evidence`. Preserve manifests beside every moved archive and verify hashes after any future copy.",
            "",
            "No file was deleted, moved, or archived in this task.",
            "",
        ]
    )
    STORAGE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return payload


def finalize() -> dict[str, Any]:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    write_policy_report(analysis)
    storage = write_storage_report()
    if analysis["stable_policy_found"]:
        config = build_v2_config(analysis["selected_policy"])
        validate_confirmation_requirements(config)
        validate_25m_resume_compatibility(config)
        V2_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config_hash = sha256_file(V2_CONFIG)
    else:
        if V2_CONFIG.exists():
            raise FileExistsError("V2 config exists even though no stable policy was selected")
        config_hash = None
    return {
        "schema_version": "darkmind-v2-phase4b-finalization-v1",
        "result": "PASS",
        "stable_policy_found": analysis["stable_policy_found"],
        "v2_config_created": config_hash is not None,
        "v2_config_sha256": config_hash,
        "confirmation_run_authorized": config_hash is not None,
        "optional_0_0002_triggered": analysis["optional_midpoint_lr"]["triggered"],
        "storage": storage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize",))
    args = parser.parse_args()
    print(json.dumps(finalize(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
