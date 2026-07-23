"""Validate a planning-only Corpus V4 acquisition plan; this module never downloads."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from darkmind_v2.corpus.phase5b_source_lock import assert_planning_only, validate_concentration


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "darkmind_v2" / "config" / "corpus_v4_acquisition_plan.json"
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
HEX_PATTERNS = {"sha256": re.compile(r"^[0-9a-f]{64}$"), "git_commit": re.compile(r"^[0-9a-f]{40}$")}
ENTRY_FIELDS = {
    "source_id", "source_family", "official_url", "snapshot_version", "expected_filename",
    "expected_size", "expected_checksum", "license_identity", "license_evidence_url",
    "attribution_record", "download_command_template", "retry_policy", "rate_limit",
    "destination_path", "classification", "approval", "post_filter_cap_tokens",
    "allowed_redirect_hosts", "content_execution_allowed",
}


def validate_acquisition_plan(plan: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != "darkmind-v2-corpus-v4-acquisition-plan-v2":
        raise ValueError("unexpected acquisition-plan schema")
    assert_planning_only(plan)
    if plan.get("execution_authorized") is not False:
        raise ValueError("acquisition execution is not authorized")
    if plan.get("acquisition_enabled") is not False or plan.get("conditional_sources_allowed") is not False:
        raise ValueError("Phase 5C acquisition must remain blocked for conditional sources")
    if plan.get("allowed_source_states") != ["approved"]:
        raise ValueError("only approved sources may appear in the acquisition plan")
    controls = plan.get("execution_controls", {})
    if controls.get("requires_separate_human_authorization") is not True:
        raise ValueError("separate human authorization is required")
    if controls.get("extract_or_execute_downloaded_content") is not False:
        raise ValueError("downloaded content execution must be disabled")
    if not isinstance(controls.get("maximum_total_bytes"), int) or controls["maximum_total_bytes"] <= 0:
        raise ValueError("a positive acquisition byte ceiling is required")
    root = plan.get("authorized_root_template", "")
    if not root or "EXTERNAL_SSD_ROOT" not in root:
        raise ValueError("authorized external-SSD root template is required")
    sources = {source["id"]: source for source in registry.get("sources", [])}
    entries = plan.get("entries", [])
    if not entries:
        raise ValueError("acquisition plan has no entries")
    seen: set[str] = set()
    for entry in entries:
        missing = ENTRY_FIELDS - set(entry)
        if missing:
            raise ValueError(f"acquisition entry missing fields: {sorted(missing)}")
        source_id = entry["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate acquisition source: {source_id}")
        seen.add(source_id)
        source = sources.get(source_id)
        if not source or source["approval_state"] != "approved":
            raise ValueError(f"unapproved source ID: {source_id}")
        if entry["official_url"] != source["official_evidence"]["official_dataset_url"]:
            raise ValueError(f"official URL changed without registry revision: {source_id}")
        if not entry["license_evidence_url"].startswith("https://") or not entry["license_identity"]:
            raise ValueError(f"missing license evidence: {source_id}")
        if not entry["expected_filename"] or Path(entry["expected_filename"]).name != entry["expected_filename"]:
            raise ValueError(f"invalid expected filename: {source_id}")
        size = entry["expected_size"]
        if not isinstance(size.get("min_bytes"), int) or not isinstance(size.get("max_bytes"), int):
            raise ValueError(f"invalid expected size: {source_id}")
        if size["min_bytes"] <= 0 or size["max_bytes"] < size["min_bytes"]:
            raise ValueError(f"invalid expected size range: {source_id}")
        checksum = entry["expected_checksum"]
        pattern = HEX_PATTERNS.get(checksum.get("algorithm"))
        if not pattern or not pattern.fullmatch(checksum.get("value", "")):
            raise ValueError(f"missing or invalid expected checksum: {source_id}")
        if not entry["destination_path"].startswith(root + "\\"):
            raise ValueError(f"download outside authorized root: {source_id}")
        if entry["classification"] != "immutable_raw":
            raise ValueError(f"raw acquisition must be immutable: {source_id}")
        if entry["content_execution_allowed"] is not False:
            raise ValueError(f"downloaded content execution enabled: {source_id}")
        official_host = entry["official_url"].split("/")[2]
        if entry["allowed_redirect_hosts"] != [official_host]:
            raise ValueError(f"redirect host allowlist is not exact: {source_id}")
        approval = entry["approval"]
        if approval.get("source_state") != "approved" or approval.get("execution_approved") is not False:
            raise ValueError(f"invalid acquisition approval state: {source_id}")
        retry = entry["retry_policy"]
        if retry.get("max_attempts", 0) < 1 or retry.get("backoff_seconds", 0) < 0:
            raise ValueError(f"invalid retry policy: {source_id}")
    approved_ids = {source_id for source_id, source in sources.items() if source["approval_state"] == "approved"}
    if seen != approved_ids:
        raise ValueError(f"plan must cover exactly approved sources: missing={sorted(approved_ids - seen)}")
    concentration = validate_concentration(entries)
    if concentration["result"] != "PASS":
        raise ValueError(f"source concentration violation: {concentration['violations']}")
    return {
        "schema_version": "darkmind-v2-corpus-v4-acquisition-plan-validation-v2",
        "result": "PASS",
        "entry_count": len(entries),
        "approved_source_coverage": f"{len(seen)}/{len(approved_ids)}",
        "concentration": concentration,
        "execution_authorized": False,
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, nargs="?", default=DEFAULT_PLAN)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    result = validate_acquisition_plan(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.registry.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
