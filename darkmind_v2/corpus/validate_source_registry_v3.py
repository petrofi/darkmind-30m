"""Validate the Phase 3 source-research registry without downloading data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCHEMA_VERSION = "darkmind-v2-source-registry-v3-candidates-v1"
REQUIRED_FIELDS = {
    "id",
    "canonical_name",
    "official_url",
    "publisher",
    "edition",
    "expected_download_bytes",
    "expected_usable_tokens",
    "language",
    "content_category",
    "license",
    "license_evidence_url",
    "redistribution_terms",
    "attribution_requirements",
    "checksum_availability",
    "extraction_method",
    "quality_risks",
    "duplication_risks",
    "pii_privacy_risks",
    "approval_status",
    "rejection_reason",
    "maximum_source_cap_tokens",
    "trust_tier",
}
APPROVAL_STATUSES = {"approved", "deferred", "rejected"}
TRUST_TIERS = {"A", "B", "C", "rejected"}
LANGUAGES = {"tr", "en", "tr-en", "multilingual"}
BANNED_APPROVED_IDS = {"common_crawl_bulk", "reddit_user_content", "oscar_web_corpus"}


def _required_text(source: dict[str, Any], field: str) -> None:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source.get('id')}: {field} must be non-empty text")


def validate_source_registry_v3(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source registry v3 schema")
    if payload.get("planning_target_tokens") != 500_000_000:
        raise ValueError("source registry must plan against the 500M target")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source registry must contain candidates")
    ids: set[str] = set()
    names: set[str] = set()
    counts: Counter[str] = Counter()
    approved_tokens = 0
    approved_download_bytes = 0
    for source in sources:
        if not isinstance(source, dict) or set(source) != REQUIRED_FIELDS:
            missing = REQUIRED_FIELDS - set(source) if isinstance(source, dict) else REQUIRED_FIELDS
            extra = set(source) - REQUIRED_FIELDS if isinstance(source, dict) else set()
            raise ValueError(f"source fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        source_id = source["id"]
        if source_id in ids or source["canonical_name"] in names:
            raise ValueError(f"duplicate source identity: {source_id}")
        ids.add(source_id)
        names.add(source["canonical_name"])
        for field in (
            "id",
            "canonical_name",
            "official_url",
            "publisher",
            "edition",
            "content_category",
            "license",
            "license_evidence_url",
            "redistribution_terms",
            "attribution_requirements",
            "checksum_availability",
            "extraction_method",
        ):
            _required_text(source, field)
        for field in ("official_url", "license_evidence_url"):
            parsed = urlparse(source[field])
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"{source_id}: {field} must be an official HTTPS URL")
        status = source["approval_status"]
        if status not in APPROVAL_STATUSES:
            raise ValueError(f"{source_id}: invalid approval status")
        if source["trust_tier"] not in TRUST_TIERS:
            raise ValueError(f"{source_id}: invalid trust tier")
        if source["language"] not in LANGUAGES:
            raise ValueError(f"{source_id}: invalid language")
        for field in ("quality_risks", "duplication_risks", "pii_privacy_risks"):
            risks = source[field]
            if not isinstance(risks, list) or not risks or not all(isinstance(item, str) and item for item in risks):
                raise ValueError(f"{source_id}: {field} must be a non-empty text list")
        for field in ("expected_download_bytes", "expected_usable_tokens", "maximum_source_cap_tokens"):
            value = source[field]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{source_id}: {field} must be a non-negative integer")
        if source["maximum_source_cap_tokens"] > 175_000_000:
            raise ValueError(f"{source_id}: source cap exceeds 35 percent of the 500M target")
        if status == "approved":
            if source_id in BANNED_APPROVED_IDS:
                raise ValueError(f"{source_id}: forbidden source cannot be approved")
            if source["expected_download_bytes"] <= 0 or source["expected_usable_tokens"] <= 0:
                raise ValueError(f"{source_id}: approved source estimates must be positive")
            if source["maximum_source_cap_tokens"] <= 0:
                raise ValueError(f"{source_id}: approved source cap must be positive")
            if source["rejection_reason"] is not None:
                raise ValueError(f"{source_id}: approved source may not have a rejection reason")
            approved_tokens += source["expected_usable_tokens"]
            approved_download_bytes += source["expected_download_bytes"]
        else:
            _required_text(source, "rejection_reason")
        if status == "rejected":
            if source["expected_usable_tokens"] != 0 or source["maximum_source_cap_tokens"] != 0:
                raise ValueError(f"{source_id}: rejected sources must contribute zero tokens")
            if source["trust_tier"] != "rejected":
                raise ValueError(f"{source_id}: rejected source must use rejected trust tier")
        counts[status] += 1
    return {
        "result": "PASS",
        "candidate_sources": len(sources),
        "status_counts": dict(sorted(counts.items())),
        "approved_expected_usable_tokens": approved_tokens,
        "approved_expected_download_bytes": approved_download_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_source_registry_v3(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
