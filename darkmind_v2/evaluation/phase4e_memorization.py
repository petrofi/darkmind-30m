"""Guard and validate the final-only Phase 4E memorization audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from darkmind_v2.training.phase4e_stage3 import FINAL_STEP, RUN_DIR, atomic_write_json, load_json


PII_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.IGNORECASE),
    "url": re.compile(r"\bhttps?://[^\s<>\]\[()]+", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
}


REQUIRED_AUDIT_KEYS = {
    "schema_version",
    "result",
    "checkpoint_model_sha256",
    "train_prefix_count",
    "heldout_prefix_count",
    "exact_continuation_match",
    "training_corpus_ngram",
    "near_exact_similarity",
    "source_category_differences",
    "pii_like_generation_counts",
    "memorized_long_span_count",
    "hard_release_blockers",
    "risk_zero_claimed",
}


def scan_pii(text: str) -> dict[str, list[str]]:
    matches = {name: [match.group(0) for match in pattern.finditer(text)] for name, pattern in PII_PATTERNS.items()}
    matches["url"] = [value.rstrip(".,;:!?") for value in matches["url"]]
    return matches


def validate_audit_schema(payload: dict[str, Any]) -> None:
    missing = REQUIRED_AUDIT_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Phase 4E memorization audit schema missing: {sorted(missing)}")
    if payload["risk_zero_claimed"] is not False:
        raise ValueError("Phase 4E memorization audit cannot claim zero extraction risk")
    blockers = payload["hard_release_blockers"]
    if not isinstance(blockers, list):
        raise ValueError("Phase 4E hard release blockers must be a list")


def require_final_checkpoint(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    progress = load_json(run_dir / "progress.json")
    if int(progress["optimizer_step"]) != FINAL_STEP:
        raise PermissionError("memorization audit is final-stop-only; Phase 4E stopped before step 11972")
    manifest = load_json(run_dir / "run_manifest.json")
    if str(FINAL_STEP) not in manifest["checkpoint_hashes"]:
        raise PermissionError("final full-resume checkpoint is absent")
    return manifest


def write_not_run_record(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    progress = load_json(run_dir / "progress.json")
    payload = {
        "schema_version": "darkmind-v2-phase4e-memorization-audit-not-run-v1",
        "result": "NOT_RUN",
        "reason": "75M continuation gate was CONDITIONAL and did not authorize the final segment",
        "stopped_optimizer_step": int(progress["optimizer_step"]),
        "final_required_optimizer_step": FINAL_STEP,
        "pii_extraction_findings": "NOT_ASSESSED",
        "memorized_long_span_findings": "NOT_ASSESSED",
        "risk_zero_claimed": False,
        "hard_release_blockers_cleared": False,
        "public_release_authorized": False,
    }
    atomic_write_json(run_dir / "memorization_audit_not_run.json", payload)
    return payload


def main() -> None:
    payload = write_not_run_record()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
