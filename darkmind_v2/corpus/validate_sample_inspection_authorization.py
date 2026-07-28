"""Validate Phase 5D bounded sample inspection authorization; never download."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHORIZATION = ROOT / "darkmind_v2" / "config" / "corpus_v4_sample_inspection_authorization.json"
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
SCHEMA_VERSION = "darkmind-v2-corpus-v4-sample-inspection-authorization-v1"
HARD_TOTAL_BYTES = 10_000_000_000
HARD_CANDIDATE_COUNT = 8
HARD_SOURCE_BYTES = 2_000_000_000
HEX = re.compile(r"^[0-9a-f]+$")
REQUIRED_ENTRY_FIELDS = {
    "source_id", "official_artifact_url", "official_metadata_url", "exact_version_snapshot",
    "expected_filename", "license_identity", "inspection_license_basis", "expected_bytes",
    "expected_checksum", "checksum_unavailable_reason", "permitted_redirect_domains",
    "authorized_url_prefixes", "destination_path", "extraction_method", "sample_selection_rule",
    "maximum_documents_files", "maximum_raw_bytes", "no_training",
}


def url_is_authorized(entry: dict[str, Any], url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    if parsed.hostname.lower() not in {item.lower() for item in entry["permitted_redirect_domains"]}:
        return False
    return any(url.startswith(prefix) for prefix in entry["authorized_url_prefixes"])


def validate_sample_authorization(payload: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected sample-authorization schema")
    if payload.get("selection_seed") != 20260723:
        raise ValueError("Phase 5D deterministic seed must be 20260723")
    limits = (
        ("maximum_total_downloaded_bytes", HARD_TOTAL_BYTES),
        ("maximum_candidates_sampled", HARD_CANDIDATE_COUNT),
        ("maximum_downloaded_bytes_per_candidate", HARD_SOURCE_BYTES),
    )
    for field, ceiling in limits:
        value = payload.get(field)
        if not isinstance(value, int) or value <= 0 or value > ceiling:
            raise ValueError(f"sample authorization exceeds hard limit: {field}")
    forbidden_true = (
        "full_production_acquisition", "corpus_construction", "training_use",
        "production_tokenization", "redistribution", "deletion", "external_upload",
        "execute_downloaded_content",
    )
    if any(payload.get(field) is not False for field in forbidden_true):
        raise ValueError("sample authorization enables a forbidden operation")
    entries = payload.get("entries", [])
    if not entries or len(entries) > payload["maximum_candidates_sampled"]:
        raise ValueError("invalid authorized candidate count")
    if payload.get("authorized_candidate_count") != len(entries):
        raise ValueError("authorized candidate count summary mismatch")
    registry_sources = {item["id"]: item for item in registry.get("sources", [])}
    seen: set[str] = set()
    entry_limit_sum = 0
    runtime_root = PureWindowsPath(payload["runtime_root"])
    for entry in entries:
        missing = REQUIRED_ENTRY_FIELDS - set(entry)
        if missing:
            raise ValueError(f"sample entry missing fields: {sorted(missing)}")
        source_id = entry["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate sample source: {source_id}")
        seen.add(source_id)
        source = registry_sources.get(source_id)
        sampled_resolution = source.get("phase5d_sample_resolution", {}) if source else {}
        was_authorized_before_resolution = (
            sampled_resolution.get("authorization_id") == payload.get("authorization_id")
            and source.get("previous_approval_state") in {"conditional", "approved"}
        )
        if not source or (
            source["approval_state"] not in {"conditional", "approved"}
            and not was_authorized_before_resolution
        ):
            raise ValueError(f"sample source is not legally eligible: {source_id}")
        if not url_is_authorized(entry, entry["official_artifact_url"]):
            raise ValueError(f"unlisted or non-official artifact URL: {source_id}")
        if not entry["official_metadata_url"].startswith("https://"):
            raise ValueError(f"official metadata URL is missing: {source_id}")
        if not entry["license_identity"] or not entry["inspection_license_basis"]:
            raise ValueError(f"inspection license basis is missing: {source_id}")
        maximum = entry["maximum_raw_bytes"]
        expected = entry["expected_bytes"]
        if not isinstance(maximum, int) or maximum <= 0 or maximum > HARD_SOURCE_BYTES:
            raise ValueError(f"per-source sample byte limit exceeded: {source_id}")
        if expected.get("minimum", -1) < 0 or expected.get("maximum", 0) > maximum:
            raise ValueError(f"expected byte range exceeds authorization: {source_id}")
        entry_limit_sum += maximum
        checksum = entry["expected_checksum"]
        if checksum is None:
            if not entry["checksum_unavailable_reason"]:
                raise ValueError(f"checksum-unavailable declaration missing: {source_id}")
        else:
            algorithm = checksum.get("algorithm")
            value = checksum.get("value", "")
            expected_length = 64 if algorithm == "sha256" else 40 if algorithm == "git_commit" else 0
            if len(value) != expected_length or not HEX.fullmatch(value):
                raise ValueError(f"invalid expected checksum: {source_id}")
        if not entry["expected_filename"] or Path(entry["expected_filename"]).name != entry["expected_filename"]:
            raise ValueError(f"invalid expected filename: {source_id}")
        destination = PureWindowsPath(entry["destination_path"])
        if runtime_root not in destination.parents or destination.name != source_id:
            raise ValueError(f"sample destination escapes runtime root: {source_id}")
        if entry["maximum_documents_files"] <= 0 or entry["no_training"] is not True:
            raise ValueError(f"sample entry lacks bounded no-training marker: {source_id}")
    if payload.get("authorized_entry_byte_sum") != entry_limit_sum:
        raise ValueError("authorized entry byte sum mismatch")
    if entry_limit_sum > payload["maximum_total_downloaded_bytes"]:
        raise ValueError("entry byte limits exceed total authorization")
    if payload.get("projected_extracted_and_intermediate_bytes", 0) <= 0:
        raise ValueError("projected intermediate storage must be explicit")
    if payload.get("minimum_free_bytes_after_completion") != 25_000_000_000:
        raise ValueError("minimum free reserve must be 25 GB")
    return {
        "schema_version": "darkmind-v2-corpus-v4-sample-authorization-validation-v1",
        "result": "PASS",
        "authorized_candidates": len(entries),
        "authorized_entry_byte_sum": entry_limit_sum,
        "hard_total_bytes": payload["maximum_total_downloaded_bytes"],
        "training_use": False,
        "full_production_acquisition": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("authorization", type=Path, nargs="?", default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    result = validate_sample_authorization(
        json.loads(args.authorization.read_text(encoding="utf-8")),
        json.loads(args.registry.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
