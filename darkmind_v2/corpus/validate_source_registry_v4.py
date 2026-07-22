"""Validate the Phase 5B Corpus V4 candidate source registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.corpus.phase5b_source_lock import (
    STATES,
    approved_capacity,
    assert_planning_only,
    category_capacity,
    classify_source_lock,
    validate_capacity,
    validate_code_policy,
    validate_concentration,
    validate_official_evidence,
    validate_overlap_plan,
    validate_pii_policy,
    validate_state_transition,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
SCHEMA_VERSION = "darkmind-v2-source-registry-v4-candidates-v2"
REQUIRED_FIELDS = {
    "id",
    "official_source_name",
    "official_url",
    "snapshot_or_version",
    "language",
    "category",
    "license",
    "license_evidence_url",
    "source_cap_tokens",
    "quality_tier",
    "relationship_to_corpus_v3",
    "previous_approval_state",
    "approval_state",
    "official_evidence",
    "acceptance_gates",
    "capacity",
    "risk",
    "resolution_steps",
}


def validate_registry(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected source-registry V4 schema")
    assert_planning_only(payload)
    if payload.get("access_date") != "2026-07-22":
        raise ValueError("registry access date must be explicit")
    sources = payload.get("sources", [])
    ids = [item.get("id") for item in sources]
    if len(sources) != 20 or len(ids) != len(set(ids)):
        raise ValueError("registry must contain 20 unique candidate IDs")
    for source in sources:
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            raise ValueError(f"{source.get('id')} missing fields: {sorted(missing)}")
        if source["approval_state"] not in STATES:
            raise ValueError(f"invalid approval state: {source['id']}")
        validate_state_transition(source["previous_approval_state"], source["approval_state"])
        validate_official_evidence(source)
        validate_capacity(source)
        validate_overlap_plan(source)
        validate_code_policy(source)
        validate_pii_policy(source)
        if source["capacity"]["conservative_tokens"] > source["source_cap_tokens"]:
            raise ValueError(f"conservative capacity exceeds post-filter source cap: {source['id']}")
        if source["approval_state"] == "rejected" and any(
            source["capacity"][key] for key in ("optimistic_tokens", "expected_tokens", "conservative_tokens")
        ):
            raise ValueError(f"rejected source has non-zero capacity: {source['id']}")

    states = Counter(item["approval_state"] for item in sources)
    approved = approved_capacity(sources)
    categories = category_capacity(sources)
    concentration = validate_concentration(payload["approved_acquisition_caps"])
    classification = classify_source_lock(
        approved["expected_tokens"],
        approved["conservative_tokens"],
        categories,
        concentration["result"] == "PASS",
        payload["acquisition_manifest_complete_for_approved_sources"],
        payload["storage_plan_feasible"],
    )
    return {
        "schema_version": "darkmind-v2-source-registry-v4-validation-v2",
        "result": "PASS",
        "candidate_sources": len(sources),
        "approval_counts": {state: states.get(state, 0) for state in sorted(STATES)},
        "official_evidence_coverage": f"{len(sources)}/{len(sources)}",
        "approved_expected_tokens": approved["expected_tokens"],
        "approved_conservative_tokens": approved["conservative_tokens"],
        "category_capacity": categories,
        "concentration": concentration,
        "source_lock_classification": classification,
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path, nargs="?", default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
