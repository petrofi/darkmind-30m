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
    classify_phase5c_source_lock,
    validate_capacity,
    validate_code_policy,
    validate_concentration,
    validate_official_evidence,
    validate_overlap_plan,
    validate_pii_policy,
    validate_state_transition,
    validate_exclusive_category_allocation,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
SCHEMA_VERSION = "darkmind-v2-source-registry-v4-candidates-v3"
PHASE5C_QUESTIONS = {
    "exact_downloadable_artifact",
    "exact_snapshot_release_dump_tag_or_commit",
    "content_license_explicit",
    "collection_or_database_license_explicit",
    "redistribution_rights_explicit",
    "modification_and_derivative_processing_rights_explicit",
    "machine_processing_or_model_training_prohibited",
    "attribution_obligations_executable",
    "official_acquisition_reproducible",
    "checksum_or_signed_manifest_available",
    "language_composition_officially_supported",
    "corpus_v3_relationship_understood",
    "conservative_capacity_estimable_without_download",
    "source_concentration_violation",
    "benchmark_pii_private_or_contamination_concerns",
}
NEW_EVIDENCE_FIELDS = {
    "stable_source_id", "official_name", "official_artifact_url", "official_metadata_url",
    "official_license_url", "exact_snapshot_version_commit_date", "language", "category",
    "artifact_type", "expected_raw_bytes", "expected_documents_or_files", "optimistic_tokens",
    "expected_tokens", "conservative_post_filter_unique_tokens", "loss_estimates_percent",
    "capacity_basis", "confidence_level", "evidence_access_date", "attribution_text_or_template",
    "source_cap_tokens", "concentration_family", "pii_risk", "benchmark_risk",
    "private_data_risk", "approval_status", "unresolved_questions",
}
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
    if len(sources) != 28 or len(ids) != len(set(ids)):
        raise ValueError("registry must contain 28 unique candidate IDs")
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

    resolved = [source for source in sources if source.get("phase5c_resolution")]
    if len(resolved) != 11:
        raise ValueError("all 11 Phase 5B conditional candidates require resolution records")
    for source in resolved:
        resolution = source["phase5c_resolution"]
        if resolution.get("question_count") != 15 or set(resolution.get("answers", {})) != PHASE5C_QUESTIONS:
            raise ValueError(f"incomplete Phase 5C resolution record: {source['id']}")
        if resolution.get("final_status") != source["approval_state"]:
            raise ValueError(f"resolution state mismatch: {source['id']}")
        if resolution.get("resolution_type") == "bounded_sample_required":
            sample = resolution.get("future_sample_plan", {})
            if sample.get("execution_authorized") is not False or sample.get("training_authorized") is not False:
                raise ValueError(f"bounded sample is executable: {source['id']}")

    discovered = [source for source in sources if source.get("phase5c_discovery")]
    if len(discovered) != 8:
        raise ValueError("Phase 5C must record eight replacement candidates")
    for source in discovered:
        evidence = source.get("new_candidate_evidence", {})
        missing = NEW_EVIDENCE_FIELDS - set(evidence)
        if missing:
            raise ValueError(f"new candidate evidence is incomplete: {source['id']}: {sorted(missing)}")
        if evidence["stable_source_id"] != source["id"] or evidence["approval_status"] != source["approval_state"]:
            raise ValueError(f"new candidate evidence identity mismatch: {source['id']}")
        if not evidence["official_artifact_url"].startswith("https://"):
            raise ValueError(f"new candidate lacks an official artifact URL: {source['id']}")
        if not evidence["capacity_basis"]:
            raise ValueError(f"new candidate lacks capacity basis: {source['id']}")

    states = Counter(item["approval_state"] for item in sources)
    approved = approved_capacity(sources)
    categories = category_capacity(sources)
    exclusive = validate_exclusive_category_allocation(sources)
    concentration = validate_concentration(payload["approved_acquisition_caps"])
    classification = classify_phase5c_source_lock(
        approved["expected_tokens"],
        approved["conservative_tokens"],
        categories,
        concentration["result"] == "PASS",
        payload["acquisition_manifest_complete_for_approved_sources"],
        True,
        payload["storage_plan_feasible"],
    )
    if payload.get("source_lock_classification") != classification:
        raise ValueError("stored source-lock classification does not match Phase 5C policy")
    if payload.get("conditional_capacity_counted_as_approved") is not False:
        raise ValueError("conditional capacity must not be counted as approved")
    if payload.get("acquisition_enabled") is not False:
        raise ValueError("Phase 5C acquisition must remain disabled")
    return {
        "schema_version": "darkmind-v2-source-registry-v4-validation-v3",
        "result": "PASS",
        "candidate_sources": len(sources),
        "approval_counts": {state: states.get(state, 0) for state in sorted(STATES)},
        "official_evidence_coverage": f"{len(sources)}/{len(sources)}",
        "approved_expected_tokens": approved["expected_tokens"],
        "approved_conservative_tokens": approved["conservative_tokens"],
        "category_capacity": categories,
        "exclusive_approved_allocation": exclusive,
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
