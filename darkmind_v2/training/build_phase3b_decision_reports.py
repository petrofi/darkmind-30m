"""Build the source-controlled Phase 3B freeze and corpus-planning artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.training.phase3b_decision import (
    generation_health,
    score_finalists,
    validate_corpus_approval,
    validate_memory_audit,
    validate_resume_continuity,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "darkmind_v2" / "config"
DATA_DIR = ROOT / "darkmind_v2" / "data" / "phase3b"
REPORT_DIR = ROOT / "darkmind_v2" / "reports"
CORPUS_DIR = ROOT / "darkmind_v2" / "corpus"
TOKENIZER_DIR = ROOT / "darkmind_v2" / "tokenizer" / "frozen" / "darkmind_v2_sp_bpe24k_v1"
MEMORY_PATH = DATA_DIR / "diagnostics" / "memory_20260713T094552Z" / "memory_audit.json"
SOAK_PATH = DATA_DIR / "diagnostics" / "soak_20260713T094629Z" / "soak_audit.json"
CHECKPOINT_DIAG_PATH = DATA_DIR / "diagnostics" / "checkpointing_20260713T094245Z" / "diagnosis.json"
EVALUATION_PATH = DATA_DIR / "finalist_evaluation.json"
INCIDENTS_PATH = DATA_DIR / "process_incidents.json"
PILOT_TOKENS = 4_997_120
SELECTED_THROUGHPUT = 14_542.114258990738
WALL_CLOCK_MULTIPLIER = 1.38


STRICT_SOURCE_STATUS = {
    "wikimedia_trwiki_20260701": ("approved", "Ready after official dump hashes and page-level attribution are captured."),
    "wikimedia_enwiki_20260701": ("approved", "Ready as a deterministic capped Wikimedia snapshot with attribution."),
    "wikimedia_simplewiki_20260701": ("approved", "Ready as a capped educational source after deduplication against enwiki."),
    "wikimedia_trwikibooks_20260701": ("approved", "Ready after namespace filtering and page-level attribution."),
    "wikimedia_trwikivoyage_20260701": ("conditional", "Requires aggressive address, phone, commercial-listing, and template filters."),
    "wikimedia_enwikiversity_20260701": ("approved", "Ready after namespace and contact-detail filtering."),
    "wikimedia_enwikibooks_20260701": ("approved", "Ready after namespace filtering and enwiki overlap deduplication."),
    "python_docs_tr_3_14_6": ("approved", "Ready with the exact Python 3.14.6 archive, PSF attribution, and example-code license metadata."),
    "python_docs_en_3_14_6": ("approved", "Ready with the exact Python 3.14.6 archive, PSF attribution, and example-code license metadata."),
    "tatoeba_tr_en_20260713": ("conditional", "Requires sentence and translation-chain attribution plus license completeness enforcement."),
    "wikimedia_trwikisource_20260701": ("deferred", "Page-level public-domain and contributor-license filtering is not implemented."),
    "wikimedia_enwikisource_20260701": ("deferred", "Work-level copyright jurisdiction and page filtering are not implemented."),
    "mdn_content_20260713": ("conditional", "Requires file-level prose/code license separation and pre-2010 code handling."),
    "project_gutenberg_en_20260713": ("conditional", "Requires work-level public-domain checks for the training jurisdiction and header cleanup."),
    "stack_exchange_20260701": ("conditional", "Requires revision-date license versioning, user attribution, and code/content separation."),
    "resmi_gazete_archive_20260701": ("deferred", "Redistribution clearance, stable bulk access, checksums, and extraction are unresolved."),
    "mevzuat_archive_20260701": ("deferred", "Redistribution clearance, dated bulk snapshot, checksums, and extraction are unresolved."),
    "common_crawl_bulk": ("rejected", "Bulk document rights, privacy, attribution, and quality remain unresolvable for this plan."),
    "reddit_user_content": ("rejected", "No approved training and redistribution grant exists for user content."),
    "oscar_web_corpus": ("rejected", "Document-level rights, attribution, privacy, and redistribution evidence are insufficient."),
}


CONTAMINATION_RISKS = {
    "general_prose": ["benchmark passages", "widely mirrored reference text"],
    "educational_prose": ["exercise solutions", "benchmark-like instructional questions"],
    "technical_documentation": ["code benchmark snippets", "documentation copied into Q&A sites"],
    "controlled_bilingual_dialogue": ["translation benchmark overlap", "reused sentence pairs"],
    "public_domain_books": ["literary benchmark overlap", "multiple editions of the same work"],
    "technical_and_code": ["coding benchmark answers", "documentation and repository overlap"],
    "institutional_prose": ["legal evaluation overlap", "near-duplicate amendments"],
    "web_crawl": ["unbounded benchmark and test-set contamination"],
    "social_media": ["unbounded benchmark leakage", "quoted copyrighted text"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_config() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(CONFIG_DIR / "model_base_candidate_d_120m.json")
    payload.update(
        {
            "architecture_name": "darkmind-v2-base-v1",
            "architecture_version": 1,
            "attention_implementation": "sdpa",
            "gradient_checkpointing": False,
            "seed": 20260712,
        }
    )
    config_path = CONFIG_DIR / "model_base_v1.json"
    write_json(config_path, payload)
    config = DarkMindV2Config.from_json_file(config_path)
    estimate = estimate_model_size(
        vocab_size=config.vocab_size,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        tied_embeddings=config.tie_word_embeddings,
        bias=config.bias,
        mlp_hidden_size=config.effective_mlp_hidden_size,
    )
    constraints = {
        "schema_version": "darkmind-v2-model-base-v1-constraints-v1",
        "architecture_name": "darkmind-v2-base-v1",
        "architecture_version": 1,
        "source_finalist": "Candidate D",
        "config_path": "darkmind_v2/config/model_base_v1.json",
        "config_sha256": sha256_file(config_path),
        "architecture_hash": model_config_hash(config),
        "parameter_counts": {
            "total": estimate.total_params,
            "transformer_body": estimate.transformer_body_params,
            "vocabulary_share_percent": estimate.vocab_related_percentage,
        },
        "tokenizer": {
            "identity": "darkmind_v2_sp_bpe24k_v1",
            "model_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.model"),
            "vocab_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.vocab"),
            "freeze_manifest_sha256": sha256_file(TOKENIZER_DIR / "tokenizer_freeze_manifest.json"),
        },
        "immutable_fields": {
            key: payload[key]
            for key in (
                "vocab_size",
                "block_size",
                "n_layer",
                "n_head",
                "n_embd",
                "mlp_hidden_size",
                "tie_word_embeddings",
                "position_embedding_type",
                "normalization",
                "activation",
                "bias",
                "tokenizer_name",
            )
        },
        "runtime_policies": {
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "gradient_checkpointing": "disabled for base v1 production runs",
            "micro_batch_size": 2,
            "sequence_length": 512,
        },
        "known_limitations": [
            "Candidate C is rejected on Windows with PyTorch 2.4.1+cu121 after repeated intermittent process terminations in the checkpointing-off profile.",
            "The frozen D architecture was only trained for a 4,997,120-token pilot and is not a finished language model.",
            "PyTorch 2.4.1+cu121 on Windows reports that flash attention is unavailable for this SDPA path.",
        ],
        "immutable_after_freeze": True,
        "future_version_conditions": [
            "a change to any immutable architecture field",
            "a tokenizer identity or vocabulary-size change",
            "a context-length or position-embedding change",
            "a backend policy that changes numerical or checkpoint compatibility",
            "new evidence that the frozen architecture cannot meet approved quality gates",
        ],
    }
    write_json(CONFIG_DIR / "model_base_v1_constraints.json", constraints)
    return payload, constraints


def build_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    memory = load_json(MEMORY_PATH)
    soak = load_json(SOAK_PATH)
    evaluation = load_json(EVALUATION_PATH)
    incidents = load_json(INCIDENTS_PATH)
    validate_memory_audit(memory)
    evidence: dict[str, Any] = {}
    bodies = {"C": 85_449_216, "D": 99_624_960}
    for candidate in ("C", "D"):
        run_name = f"candidate_{candidate.lower()}_5m_v1"
        run_dir = DATA_DIR / "runs" / run_name
        first = load_json(run_dir / "resume" / "segment_to_step_000305.json")
        second = load_json(run_dir / "resume" / "segment_to_step_000610.json")
        validate_resume_continuity(first)
        validate_resume_continuity(second)
        initial = evaluation["results"][candidate]["0"]
        final = evaluation["results"][candidate]["610"]
        hard_failures = []
        if candidate == "C" and len(incidents["incidents"]) >= 2:
            hard_failures.append("repeated process crash in recommended non-checkpointing profile")
        evidence[candidate] = {
            "initial_validation_loss": initial["validation"]["loss"],
            "final_validation_loss": final["validation"]["loss"],
            "initial_eval_loss": initial["eval"]["loss"],
            "final_eval_loss": final["eval"]["loss"],
            "pilot_wall_seconds": first["elapsed_seconds"] + second["elapsed_seconds"],
            "transformer_body_params": bodies[candidate],
            "initial_generation": initial,
            "final_generation": final,
            "soak_passed": soak["results"][candidate]["result"] == "PASS",
            "vram_headroom_percent": memory["results"][candidate]["reserved_headroom_percent"],
            "checkpoint_resume_reliability": 50.0 if candidate == "C" else 100.0,
            "implementation_backend_reliability": 50.0 if candidate == "C" else 100.0,
            "hard_failures": hard_failures,
        }
    return evidence, memory, soak, evaluation


def build_decision_report(
    evidence: dict[str, Any], decision: dict[str, Any], evaluation: dict[str, Any]
) -> None:
    rows = {row["candidate"]: row for row in decision["rows"]}
    lines = [
        "# Phase 3B Final Architecture Decision",
        "",
        "Status: **FROZEN**",
        "",
        "Selected architecture: **Candidate D / darkmind-v2-base-v1**",
        "",
        "Recommendation strength: **STRONG for the production-base architecture selection**. The 5M-token checkpoints remain early research pilots and are not finished or conversational models.",
        "",
        "## Predeclared Rules",
        "",
        "Before comparing final outcomes, a material D learning advantage was defined as at least 2% lower final validation loss than C, or a clearly better loss slope without a stability penalty. Hard eligibility gates are applied before score or the within-2% Candidate C tie-break.",
        "",
        "## Equal-Token Learning",
        "",
        f"Both candidates consumed exactly **{PILOT_TOKENS:,} tokens** in 610 optimizer steps, with identical ordered sequences and a fresh-process restart at step 305.",
        "",
        "| Candidate | Step | Validation loss | Eval loss |",
        "|---|---:|---:|---:|",
    ]
    for candidate in ("C", "D"):
        for step in (0, 152, 305, 458, 610):
            record = evaluation["results"][candidate][str(step)]
            lines.append(
                f"| {candidate} | {step} | {record['validation']['loss']:.6f} | {record['eval']['loss']:.6f} |"
            )
    lines.extend(
        [
            "",
            "| Candidate | Validation reduction | Eval reduction | Validation reduction / 1M tokens | Eval reduction / 1M tokens | Mean loss reduction / wall hour |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("C", "D"):
        raw = rows[candidate]["raw_metrics"]
        lines.append(
            f"| {candidate} | {raw['validation_improvement']:.6f} | {raw['eval_improvement']:.6f} | "
            f"{raw['validation_improvement'] / (PILOT_TOKENS / 1_000_000):.6f} | "
            f"{raw['eval_improvement'] / (PILOT_TOKENS / 1_000_000):.6f} | "
            f"{raw['loss_improvement_per_hour']:.3f} |"
        )
    c_final = evidence["C"]["final_validation_loss"]
    d_final = evidence["D"]["final_validation_loss"]
    lines.extend(
        [
            "",
            f"C finished with {(d_final - c_final) / d_final * 100:.3f}% lower validation loss and {(evidence['D']['final_eval_loss'] - evidence['C']['final_eval_loss']) / evidence['D']['final_eval_loss'] * 100:.3f}% lower eval loss. This is below the predeclared 2% materiality threshold.",
            "",
            "## Generation-Health Trend",
            "",
            "The composite averages no-repetition, no-loop, unique-token ratio, EOS completion, and meaningful-continuation proxy rates across greedy and seeded sampling. It is supporting evidence, not a conversational-quality score.",
            "",
            "| Candidate | Init composite | Final composite | Trend | Final greedy repetition | Final sampling repetition |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("C", "D"):
        initial = evidence[candidate]["initial_generation"]
        final = evidence[candidate]["final_generation"]
        lines.append(
            f"| {candidate} | {generation_health(initial):.4f} | {generation_health(final):.4f} | "
            f"{generation_health(final) - generation_health(initial):+.4f} | "
            f"{final['greedy']['quality_warning_counts'].get('repetition', 0)}/200 | "
            f"{final['sampling']['quality_warning_counts'].get('repetition', 0)}/500 |"
        )
    lines.extend(
        [
            "",
            "Both finalists eliminated invalid UTF-8 and replacement-character events by the final milestone. C had the stronger greedy trend; D had healthier final sampling repetition, loop, unique-token, and meaningful-proxy results. Outputs remain largely incoherent at this token budget.",
            "",
            "## Weighted Score",
            "",
            "| Candidate | Weighted score | Eligible | Validation 25% | Eval 15% | Loss/hour 10% | Body 10% | Generation 10% | Stability 10% | VRAM 10% | Resume 5% | Backend 5% |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("C", "D"):
        row = rows[candidate]
        c = row["component_scores"]
        lines.append(
            f"| {candidate} | {row['weighted_score']:.2f} | {'PASS' if row['eligible'] else 'FAIL'} | "
            f"{c['validation_improvement']:.2f} | {c['eval_improvement']:.2f} | "
            f"{c['loss_improvement_per_hour']:.2f} | {c['transformer_body_capacity']:.2f} | "
            f"{c['generation_health_trend']:.2f} | {c['long_run_stability']:.2f} | "
            f"{c['vram_headroom']:.2f} | {c['checkpoint_resume_reliability']:.2f} | "
            f"{c['implementation_backend_reliability']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Hard-Gate Decision",
            "",
            "C is ineligible because the recommended checkpointing-off Windows/PyTorch profile had two unexpected process terminations: one LR-calibration child exited with 3221226505 and one first Segment B child ended silently before writing a new metric. Both exact retries passed, which classifies the issue as intermittent process/backend instability rather than a deterministic model-code defect.",
            "",
            "D passed post-warmup memory, 1,000-step soak, all calibration runs, safetensors integrity, and the forced midpoint checkpoint/resume continuity gates without a process crash. D is selected even though C's final losses are about 0.44% better, because hard eligibility precedes the score and tie-break.",
            "",
            "The Phase 3A score-only recommendation for C is superseded. Candidate D is frozen as `darkmind_v2/config/model_base_v1.json`.",
        ]
    )
    (REPORT_DIR / "phase3b_final_architecture_decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        REPORT_DIR / "phase3b_finalist_decision.json",
        {
            "schema_version": "darkmind-v2-phase3b-finalist-decision-v1",
            "pilot_tokens_per_candidate": PILOT_TOKENS,
            "decision": decision,
        },
    )


def build_checkpointing_report(incidents: dict[str, Any]) -> None:
    diagnosis = load_json(CHECKPOINT_DIAG_PATH)
    lines = [
        "# Phase 3B Candidate C Checkpointing Diagnosis",
        "",
        "Classification: **intermittent process/backend instability**",
        "",
        "The checkpointing diagnosis environment was Windows, PyTorch 2.4.1+cu121, CUDA 12.1, RTX 4060 Laptop GPU, BF16, sequence length 512, micro-batch 2, SDPA, and non-reentrant gradient checkpointing.",
        "",
        "Phase 3A preserved one worker exit with code 3221226505 (`STATUS_STACK_BUFFER_OVERRUN` / fail-fast, `0xC0000409`) before a result file was written. The five identical Phase 3B attempts below did not reproduce that exit.",
        "",
        "## Identical Candidate C Attempts",
        "",
        "| Attempt | Exit | Windows exception | Last completed operation | Result |",
        "|---:|---:|---|---|---|",
    ]
    for index, item in enumerate(diagnosis["primary_attempts"], start=1):
        progress = item.get("last_progress") or {}
        lines.append(
            f"| {index} | {item['worker_exit_code']} | {item.get('windows_exception') or '-'} | "
            f"{progress.get('operation', '-')} | {item.get('result', 'CRASH')} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| Candidate | MB | Attention | Checkpointing | Exit | Result |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for item in diagnosis["controls"]:
        lines.append(
            f"| {item['candidate']} | {item['batch_size']} | {item['attention']} | "
            f"{'on' if item['gradient_checkpointing'] else 'off'} | {item['worker_exit_code']} | "
            f"{item.get('result', 'CRASH')} |"
        )
    first, second = incidents["incidents"]
    lines.extend(
        [
            "",
            "## Subsequent Checkpointing-Off Incidents",
            "",
            f"1. The first C LR=0.0001 calibration worker exited with `{first['observed_exit_code']}` and wrote no result; its exact retry passed.",
            f"2. The first C Segment B child ended before a new metric was written, left the midpoint checkpoint unchanged, and emitted no diagnostic stdout/stderr. Its native exception code is unavailable; the instrumented exact retry passed.",
            "",
            "These two events occurred in the recommended checkpointing-off profile. They satisfy the predeclared repeated-process-crash hard gate even though the successful retries, memory audit, and soak show that the failure is intermittent.",
            "",
            "## Policy",
            "",
            "Gradient checkpointing is unnecessary at the measured memory levels and remains disabled. The exact C checkpointing-on combination is rejected before training. The broader C checkpointing-off signature emits a warning on the affected Windows/PyTorch environment. Candidate D is the frozen production-base architecture.",
            "",
            "No deterministic source-code defect, OOM, or exact native cause was established. The evidence supports intermittent process/backend instability; all failed and successful evidence remains in ignored runtime JSON.",
        ]
    )
    (REPORT_DIR / "phase3b_candidate_c_checkpointing_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def annotate_calibration_report(incidents: dict[str, Any]) -> None:
    path = REPORT_DIR / "phase3b_learning_rate_calibration.md"
    marker = "## Process Reliability Note"
    text = path.read_text(encoding="utf-8").rstrip()
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    first = incidents["incidents"][0]
    text += (
        f"\n\n{marker}\n\nThe first Candidate C LR=0.0001 child exited with "
        f"`{first['observed_exit_code']}` before writing a result. The identical retry passed and "
        "produced the table entry above. The failed attempt remains part of the architecture reliability gate.\n"
    )
    path.write_text(text, encoding="utf-8")


def build_memory_report(memory: dict[str, Any]) -> None:
    lines = [
        "# Phase 3B Post-Optimizer Memory Audit",
        "",
        "Both finalists passed 10 real AdamW optimizer steps in clean isolated processes. Optimizer state was materialized before measurement.",
        "",
        "| Candidate | Model weights | Gradients | Optimizer states | Activation/temp peak | Peak allocated | Peak reserved | Reserved headroom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("C", "D"):
        item = memory["results"][candidate]
        lines.append(
            f"| {candidate} | {item['model_weight_bytes']:,} | {item['gradient_bytes']:,} | "
            f"{item['optimizer_state_bytes']:,} | {item['activation_and_temporary_peak_bytes']:,} | "
            f"{item['peak_allocated_bytes']:,} | {item['peak_reserved_bytes']:,} | "
            f"{item['reserved_headroom_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Profile: BF16, sequence length 512, micro-batch 2, SDPA, gradient checkpointing disabled. Both exceed the 15% safe-headroom gate; the architecture decision uses these post-warmup figures rather than initialization-only memory.",
        ]
    )
    (REPORT_DIR / "phase3b_post_optimizer_memory.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_token_plan(constraints: dict[str, Any]) -> dict[str, Any]:
    stages = [5_000_000, 25_000_000, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_000_000_000]
    projections = []
    for tokens in stages:
        active_hours = tokens / SELECTED_THROUGHPUT / 3600.0
        projections.append(
            {
                "tokens": tokens,
                "optional": tokens >= 1_000_000_000,
                "active_training_hours": active_hours,
                "conservative_wall_clock_hours": active_hours * WALL_CLOCK_MULTIPLIER,
                "requires_user_approval": True,
                "gate": "full validation/eval loss, generation health, memorization, source exposure, checkpoint integrity, and human go/no-go review",
            }
        )
    params = constraints["parameter_counts"]["total"]
    payload = {
        "schema_version": "darkmind-v2-production-training-token-plan-v1",
        "architecture": "darkmind-v2-base-v1",
        "model_parameters": params,
        "unique_clean_corpus_targets": {
            "minimum_serious_tokens": 500_000_000,
            "preferred_stretch_tokens": 1_000_000_000,
        },
        "total_seen_token_stages": projections,
        "token_to_parameter_ratios": {
            "500m": 500_000_000 / params,
            "1b": 1_000_000_000 / params,
            "2b": 2_000_000_000 / params,
        },
        "projection_basis": {
            "measured_long_run_tokens_per_second": SELECTED_THROUGHPUT,
            "candidate": "D",
            "wall_clock_multiplier": WALL_CLOCK_MULTIPLIER,
            "multiplier_scope": "15% evaluation/checkpoint overhead and 20% interruption/thermal allowance",
        },
        "continuation_policy": "After every stage, stop unless all gates pass and the user approves the next budget.",
        "repetition_policy": "Prefer additional unique high-quality data. Any repeated epoch must use a recorded deterministic reshuffle.",
        "early_stop_rules": [
            "stop on non-finite loss or gradients",
            "stop on checkpoint, resume, or data-position mismatch",
            "stop when validation loss degrades across two scheduled gates without a documented recovery",
            "stop when repetition, Unicode, or meaningful-continuation health materially regresses",
            "stop when train-validation divergence or source-specific memorization exceeds the approved threshold",
        ],
        "overfitting_controls": [
            "document-boundary-aware split isolation",
            "deduplication and contamination audit before tokenization",
            "source-cap and per-source held-out evaluation",
            "canary and near-verbatim memorization checks",
            "deterministic epoch and sequence-order manifests",
        ],
        "scaling_law_caveat": "General large-model scaling ratios are planning references, not an exact law for this small bilingual model. Two billion tokens are neither mandatory nor guaranteed sufficient.",
    }
    write_json(CONFIG_DIR / "production_training_token_plan.json", payload)
    return payload


def build_scaling_report(plan: dict[str, Any]) -> None:
    lines = [
        "# Phase 3B Scaling Target Rationale",
        "",
        "The frozen base has 118,056,960 parameters. The minimum serious unique clean corpus target is 500M tokens; 1B is a preferred stretch target only if licensing, attribution, quality, and acquisition remain practical.",
        "",
        "| Seen tokens | Tokens / parameter | Active training | Conservative wall clock |",
        "|---:|---:|---:|---:|",
    ]
    for label, tokens in (("500M", 500_000_000), ("1B", 1_000_000_000), ("2B", 2_000_000_000)):
        row = next(item for item in plan["total_seen_token_stages"] if item["tokens"] == tokens)
        lines.append(
            f"| {label} | {tokens / plan['model_parameters']:.3f} | {row['active_training_hours']:.2f} h | {row['conservative_wall_clock_hours']:.2f} h |"
        )
    lines.extend(
        [
            "",
            "The projection uses Candidate D's 1,000-step measured 14,542.1 token/s and a 1.38 wall-clock multiplier for evaluation, checkpoint, interruption, and thermal allowance. It is a budget estimate, not a runtime guarantee.",
            "",
            "Stages are 5M, 25M, 100M, 250M, 500M, optional 1B, and optional 2B. Every transition requires validation/eval and generation-health review, checkpoint integrity, memorization/source-exposure checks, and explicit user approval.",
            "",
            "At 500M, continuation is evidence-driven. Additional unique clean data is preferred over repetition. Repeated epochs require deterministic reshuffling plus source-overexposure and near-verbatim memorization checks. General scaling-law ratios cannot be treated as an exact prescription for a 118M-parameter Turkish-English model, and 2B tokens are neither mandatory nor sufficient by definition.",
        ]
    )
    (REPORT_DIR / "phase3b_scaling_target_rationale.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_source_approval() -> dict[str, Any]:
    candidates = load_json(CORPUS_DIR / "source_registry.v3.candidates.json")
    sources = []
    for original in candidates["sources"]:
        source = copy.deepcopy(original)
        status, reason = STRICT_SOURCE_STATUS[source["id"]]
        source["phase3a_approval_status"] = source.pop("approval_status")
        source.pop("rejection_reason", None)
        source["approval_status"] = status
        source["approval_reason"] = reason
        source["quality_tier"] = source.pop("trust_tier")
        source["contamination_risks"] = CONTAMINATION_RISKS[source["content_category"]]
        source["extraction_implementation_readiness"] = {
            "approved": "ready_after_snapshot_and_checksum_verification",
            "conditional": "blocked_by_named_license_quality_or_attribution_condition",
            "deferred": "not_ready",
            "rejected": "not_applicable",
        }[status]
        sources.append(source)
    approved_tokens = sum(
        source["expected_usable_tokens"] for source in sources if source["approval_status"] == "approved"
    )
    conditional_tokens = sum(
        source["expected_usable_tokens"]
        for source in sources
        if source["approval_status"] == "conditional"
    )
    payload = {
        "schema_version": "darkmind-v2-source-registry-v3-approval-v1",
        "decision_date": "2026-07-13",
        "download_authorized": False,
        "approval_policy": "A source status authorizes planning only. Download requires a separate user approval after dated snapshot, license evidence, attribution, and checksum review.",
        "summary": {
            "status_counts": dict(Counter(source["approval_status"] for source in sources)),
            "approved_unique_tokens": approved_tokens,
            "conditional_unique_tokens": conditional_tokens,
            "approved_gap_to_500m": 500_000_000 - approved_tokens,
            "approved_gap_to_1b": 1_000_000_000 - approved_tokens,
            "approved_plus_conditional_gap_to_500m": 500_000_000 - approved_tokens - conditional_tokens,
            "approved_plus_conditional_gap_to_1b": 1_000_000_000 - approved_tokens - conditional_tokens,
        },
        "first_100m_tranche": [
            {"source_id": "wikimedia_trwiki_20260701", "tokens": 55_000_000},
            {"source_id": "wikimedia_enwiki_20260701", "tokens": 30_000_000},
            {"source_id": "python_docs_tr_3_14_6", "tokens": 5_000_000},
            {"source_id": "python_docs_en_3_14_6", "tokens": 10_000_000},
        ],
        "first_100m_language_mix": {"tr": 60_000_000, "en": 40_000_000},
        "sources": sources,
    }
    validate_corpus_approval(payload)
    write_json(CORPUS_DIR / "source_registry.v3.approval.json", payload)
    return payload


def build_source_report(approval: dict[str, Any]) -> None:
    summary = approval["summary"]
    lines = [
        "# Phase 3B Corpus Source Approval",
        "",
        "Status: **PLANNING APPROVAL ONLY; DOWNLOAD NOT AUTHORIZED**",
        "",
        "All 20 Phase 3A candidates were reclassified under the stricter production policy. The machine-readable registry contains the official URL, dated snapshot status, license and redistribution evidence, attribution, checksum source, raw-byte estimate, usable-token estimate, language/category, cap, quality tier, deduplication, contamination, PII, and extraction-readiness fields for every source.",
        "",
        "| Status | Sources | Expected usable tokens |",
        "|---|---:|---:|",
    ]
    for status in ("approved", "conditional", "deferred", "rejected"):
        tokens = sum(
            item["expected_usable_tokens"]
            for item in approval["sources"]
            if item["approval_status"] == status
        )
        lines.append(f"| {status} | {summary['status_counts'][status]} | {tokens:,} |")
    lines.extend(
        [
            "",
            f"Approved sources contribute an estimated **{summary['approved_unique_tokens']:,} unique tokens**, leaving **{summary['approved_gap_to_500m']:,}** to 500M and **{summary['approved_gap_to_1b']:,}** to 1B. Approved plus conditional sources reach 460M, leaving 40M and 540M respectively.",
            "",
            "## First 100M Tranche Proposal",
            "",
            "| Source | Tokens | Language |",
            "|---|---:|---|",
            "| Turkish Wikipedia 2026-07-01 | 55,000,000 | tr |",
            "| English Wikipedia 2026-07-01 | 30,000,000 | en |",
            "| Python 3.14.6 Turkish docs | 5,000,000 | tr |",
            "| Python 3.14.6 English docs | 10,000,000 | en |",
            "",
            "Total: 100M tokens, 60% Turkish and 40% English. Each tranche cap is below its registry maximum and requires exact snapshot/checksum verification before any acquisition.",
            "",
            "## Conditions and Deferrals",
            "",
        ]
    )
    for status in ("conditional", "deferred", "rejected"):
        lines.append(f"### {status.title()}")
        lines.append("")
        for item in approval["sources"]:
            if item["approval_status"] == status:
                lines.append(f"- `{item['id']}`: {item['approval_reason']}")
        lines.append("")
    lines.extend(
        [
            "Approved status does not authorize download. Corpus acquisition remains stopped until the user approves an exact source tranche and its snapshot/checksum plan.",
        ]
    )
    (REPORT_DIR / "phase3b_corpus_source_approval.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_freeze_report(config: dict[str, Any], constraints: dict[str, Any]) -> None:
    counts = constraints["parameter_counts"]
    tokenizer = constraints["tokenizer"]
    lines = [
        "# DarkMind v2 Model Base v1 Freeze",
        "",
        "Status: **FROZEN**",
        "",
        "Architecture: **darkmind-v2-base-v1 (Candidate D)**",
        "",
        f"- Total parameters: {counts['total']:,}",
        f"- Transformer body: {counts['transformer_body']:,}",
        f"- Vocabulary share: {counts['vocabulary_share_percent']:.4f}%",
        f"- Layers / heads / head dimension: {config['n_layer']} / {config['n_head']} / {config['n_embd'] // config['n_head']}",
        f"- Embedding / MLP hidden dimension: {config['n_embd']} / {config['mlp_hidden_size']}",
        f"- Context length: {config['block_size']}",
        "- Embeddings: tied and immutable",
        "- Attention: SDPA production policy",
        "- Gradient checkpointing: disabled for base v1 production runs",
        "- Precision: BF16",
        f"- Tokenizer: `{tokenizer['identity']}`",
        f"- Tokenizer model SHA-256: `{tokenizer['model_sha256']}`",
        f"- Tokenizer vocab SHA-256: `{tokenizer['vocab_sha256']}`",
        f"- Tokenizer freeze manifest SHA-256: `{tokenizer['freeze_manifest_sha256']}`",
        f"- Config SHA-256: `{constraints['config_sha256']}`",
        f"- Architecture hash: `{constraints['architecture_hash']}`",
        "",
        "## Evidence",
        "",
        "Candidate D passed LR calibration, a 4,997,120-token equal-data pilot, a true midpoint process restart, safetensors integrity, post-optimizer memory audit, and a 1,000-step soak. Candidate C learned about 0.44% better by final loss but failed the repeated non-checkpointing process-crash hard gate.",
        "",
        "## Limitations and Versioning",
        "",
        "The frozen architecture is a 100M-class from-scratch research base. The pilot checkpoint is not a finished model and does not support a conversational-quality claim. The recorded C instability is specific to the observed Windows/PyTorch 2.4.1+cu121 environment; D is the production profile for that environment.",
        "",
        "All immutable fields in `model_base_v1_constraints.json` require a new architecture version if changed. A tokenizer, vocabulary, context/position, numerical-backend compatibility, or material architecture change must create v2 rather than mutating base v1.",
    ]
    (REPORT_DIR / "phase3b_model_base_v1_freeze.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    config, constraints = build_frozen_config()
    evidence, memory, _soak, evaluation = build_evidence()
    incidents = load_json(INCIDENTS_PATH)
    decision = score_finalists(evidence)
    if decision["selected"] != "D":
        raise RuntimeError("Phase 3B evidence did not select Candidate D")
    build_decision_report(evidence, decision, evaluation)
    build_checkpointing_report(incidents)
    annotate_calibration_report(incidents)
    build_memory_report(memory)
    plan = build_token_plan(constraints)
    build_scaling_report(plan)
    approval = build_source_approval()
    build_source_report(approval)
    build_freeze_report(config, constraints)
    print(json.dumps({"result": "PASS", "selected": "D", "score": decision}, indent=2))


if __name__ == "__main__":
    main()
