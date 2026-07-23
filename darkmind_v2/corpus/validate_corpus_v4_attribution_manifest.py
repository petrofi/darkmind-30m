"""Validate the planning-only Corpus V4 attribution manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "darkmind_v2" / "config" / "corpus_v4_attribution_manifest.json"
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
REQUIRED_ENTRY_FIELDS = {
    "source_id", "official_source_name", "snapshot_version", "official_url",
    "license_identity", "license_evidence_url", "attribution_record",
    "modification_notice", "record_key_template", "approval_state",
    "acquisition_execution_authorized",
}


def validate_attribution_manifest(manifest: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "darkmind-v2-corpus-v4-attribution-manifest-v1":
        raise ValueError("unexpected attribution-manifest schema")
    if manifest.get("planning_only") is not True:
        raise ValueError("attribution manifest must remain planning-only")
    for key in ("downloads_performed", "scraping_performed", "execution_authorized"):
        if manifest.get(key) is not False:
            raise ValueError(f"attribution manifest forbids {key}")
    approved = {item["id"] for item in registry["sources"] if item["approval_state"] == "approved"}
    entries = manifest.get("entries", [])
    ids = [item.get("source_id") for item in entries]
    if len(ids) != len(set(ids)) or set(ids) != approved:
        raise ValueError("attribution entries must map exactly to approved sources")
    for entry in entries:
        missing = REQUIRED_ENTRY_FIELDS - set(entry)
        if missing:
            raise ValueError(f"incomplete attribution entry: {entry.get('source_id')}: {sorted(missing)}")
        if entry["approval_state"] != "approved" or entry["acquisition_execution_authorized"] is not False:
            raise ValueError(f"invalid attribution approval: {entry['source_id']}")
        if not entry["official_url"].startswith("https://") or not entry["license_evidence_url"].startswith("https://"):
            raise ValueError(f"attribution URLs must be official HTTPS: {entry['source_id']}")
        if not entry["snapshot_version"] or not entry["license_identity"] or not entry["modification_notice"]:
            raise ValueError(f"attribution identity is incomplete: {entry['source_id']}")
        if not isinstance(entry["attribution_record"], dict) or not entry["attribution_record"]:
            raise ValueError(f"attribution record is empty: {entry['source_id']}")
    coverage = manifest.get("coverage", {})
    if coverage != {"approved_sources": len(approved), "manifest_entries": len(entries), "complete": True}:
        raise ValueError("attribution coverage summary is inconsistent")
    return {
        "schema_version": "darkmind-v2-corpus-v4-attribution-validation-v1",
        "result": "PASS",
        "manifest_id": manifest["manifest_id"],
        "approved_source_coverage": f"{len(entries)}/{len(approved)}",
        "execution_authorized": False,
        "downloads_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    result = validate_attribution_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.registry.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
