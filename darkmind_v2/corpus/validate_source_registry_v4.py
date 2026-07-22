"""Validate the planning-only Corpus V4 candidate source registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
REQUIRED_FIELDS = {
    "id", "official_source_name", "official_url", "snapshot_or_version", "language", "category",
    "expected_raw_bytes", "expected_usable_unique_tokens", "license", "license_evidence_url",
    "redistribution_conditions", "attribution_requirements", "source_cap_tokens", "quality_tier",
    "extraction_readiness", "checksum_availability", "pii_risk", "duplicate_risk",
    "benchmark_contamination_risk", "relationship_to_corpus_v3", "approval_state",
}
APPROVAL_STATES = {"approved", "conditional", "deferred", "rejected"}
RISK_LEVELS = {"low", "medium", "high"}


def validate_registry(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "darkmind-v2-source-registry-v4-candidates-v1":
        raise ValueError("unexpected source-registry V4 schema")
    if payload.get("planning_only") is not True or payload.get("downloads_performed") is not False:
        raise ValueError("source registry must remain planning-only")
    sources = payload.get("sources", [])
    ids = [item.get("id") for item in sources]
    if not sources or len(ids) != len(set(ids)):
        raise ValueError("source IDs must be present and unique")
    for source in sources:
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            raise ValueError(f"{source.get('id')} missing fields: {sorted(missing)}")
        if source["approval_state"] not in APPROVAL_STATES:
            raise ValueError(f"invalid approval state: {source['id']}")
        if any(source[key] not in RISK_LEVELS for key in ("pii_risk", "duplicate_risk", "benchmark_contamination_risk")):
            raise ValueError(f"invalid risk level: {source['id']}")
        if not source["official_url"].startswith("https://") or not source["license_evidence_url"].startswith("https://"):
            raise ValueError(f"official HTTPS evidence required: {source['id']}")
        numeric = ("expected_raw_bytes", "expected_usable_unique_tokens", "source_cap_tokens")
        if any(not isinstance(source[key], int) or source[key] < 0 for key in numeric):
            raise ValueError(f"invalid capacity field: {source['id']}")
        if source["expected_usable_unique_tokens"] > source["source_cap_tokens"]:
            raise ValueError(f"expected capacity exceeds source cap: {source['id']}")
        if source["approval_state"] == "rejected" and any(source[key] != 0 for key in numeric):
            raise ValueError(f"rejected source has non-zero acquisition capacity: {source['id']}")
    states = Counter(item["approval_state"] for item in sources)
    capacity = {
        state: sum(item["expected_usable_unique_tokens"] for item in sources if item["approval_state"] == state)
        for state in sorted(APPROVAL_STATES)
    }
    return {
        "schema_version": "darkmind-v2-source-registry-v4-validation-v1",
        "result": "PASS",
        "candidate_sources": len(sources),
        "approval_counts": dict(sorted(states.items())),
        "expected_unique_token_capacity_by_state": capacity,
        "approved_plus_conditional_capacity": capacity["approved"] + capacity["conditional"],
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path, nargs="?", default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    print(json.dumps(validate_registry(json.loads(args.registry.read_text(encoding="utf-8"))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
