"""Pure validation helpers for the planning-only Corpus V4 source lock."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


STATES = {"approved", "conditional", "deferred", "rejected"}
ALLOWED_TRANSITIONS = {
    "approved": STATES,
    "conditional": STATES,
    "deferred": STATES,
    "rejected": {"rejected"},
}
EVIDENCE_FIELDS = {
    "access_date", "official_dataset_url", "official_license_urls", "artifact_license",
    "content_license", "database_license", "exact_snapshot", "download_mechanism",
    "redistribution_rights", "modification_rights", "attribution_requirements",
    "share_alike_requirements", "noncommercial_restrictions", "machine_learning_restrictions",
    "terms_of_service_restrictions", "rate_limits", "checksum_or_manifest",
    "official_file_inventory", "expected_raw_size", "expected_document_count",
    "expected_language_distribution", "update_status",
}
CATEGORY_TARGETS = {
    "turkish_general_educational": 72_000_000,
    "english_general_educational": 50_000_000,
    "technical_documentation": 42_000_000,
    "code_structured_text": 26_000_000,
    "controlled_bilingual": 10_000_000,
}
OVERLAP_FIELDS = {
    "document_id", "url", "hash", "normalized_fingerprint", "near_dedup",
    "ngram_sampling", "corpus_v3_wikimedia", "corpus_v3_python_docs", "benchmark",
}


def validate_state_transition(previous: str, current: str) -> None:
    if previous not in STATES or current not in STATES:
        raise ValueError("unknown source acceptance state")
    if current not in ALLOWED_TRANSITIONS[previous]:
        raise ValueError(f"source transition is not allowed: {previous} -> {current}")


def validate_official_evidence(source: dict[str, Any]) -> None:
    evidence = source.get("official_evidence", {})
    missing = EVIDENCE_FIELDS - set(evidence)
    if missing:
        raise ValueError(f"{source.get('id')} missing official evidence: {sorted(missing)}")
    urls = [evidence["official_dataset_url"], *evidence["official_license_urls"]]
    if not urls or any(not isinstance(url, str) or not url.startswith("https://") for url in urls):
        raise ValueError(f"{source.get('id')} requires official HTTPS evidence")
    if evidence["content_license"] == "unknown" or evidence["database_license"] == "unknown":
        if source.get("approval_state") == "approved":
            raise ValueError(f"{source.get('id')} has unresolved content/database licensing")
    if source.get("approval_state") == "approved":
        gates = source.get("acceptance_gates", {})
        required_true = {
            "artifact_identity_clear", "snapshot_reproducible", "license_explicit",
            "machine_processing_permitted", "acquisition_method_permitted",
            "attribution_actionable", "quality_suitable", "overlap_manageable",
        }
        failed = sorted(key for key in required_true if gates.get(key) is not True)
        if failed:
            raise ValueError(f"approved source {source.get('id')} fails gates: {failed}")
        if source.get("resolution_steps"):
            raise ValueError(f"approved source {source.get('id')} still has resolution steps")
    elif source.get("approval_state") == "conditional" and not source.get("resolution_steps"):
        raise ValueError(f"conditional source {source.get('id')} needs exact resolution steps")


def validate_capacity(source: dict[str, Any]) -> None:
    capacity = source.get("capacity", {})
    required = {"optimistic_tokens", "expected_tokens", "conservative_tokens", "expected_rejection_percent", "confidence", "basis"}
    missing = required - set(capacity)
    if missing:
        raise ValueError(f"{source.get('id')} missing capacity fields: {sorted(missing)}")
    optimistic = capacity["optimistic_tokens"]
    expected = capacity["expected_tokens"]
    conservative = capacity["conservative_tokens"]
    if any(not isinstance(value, int) or value < 0 for value in (optimistic, expected, conservative)):
        raise ValueError(f"{source.get('id')} has invalid capacity")
    if not optimistic >= expected >= conservative:
        raise ValueError(f"{source.get('id')} capacity must be optimistic >= expected >= conservative")
    rejection = capacity["expected_rejection_percent"]
    if not isinstance(rejection, (int, float)) or not 0 <= rejection <= 100:
        raise ValueError(f"{source.get('id')} has invalid rejection percentage")
    if capacity["confidence"] not in {"low", "medium", "high"} or not capacity["basis"]:
        raise ValueError(f"{source.get('id')} has invalid capacity evidence")


def approved_capacity(sources: list[dict[str, Any]]) -> dict[str, int]:
    approved = [source for source in sources if source["approval_state"] == "approved"]
    return {
        "expected_tokens": sum(source["capacity"]["expected_tokens"] for source in approved),
        "conservative_tokens": sum(source["capacity"]["conservative_tokens"] for source in approved),
    }


def category_capacity(sources: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals = {category: {"expected": 0, "conservative": 0} for category in CATEGORY_TARGETS}
    for source in sources:
        if source["approval_state"] != "approved":
            continue
        for category, values in source.get("approved_category_capacity", {}).items():
            if category not in totals:
                raise ValueError(f"unknown allocation category: {category}")
            totals[category]["expected"] += values["expected_tokens"]
            totals[category]["conservative"] += values["conservative_tokens"]
    for category, values in totals.items():
        values["target"] = CATEGORY_TARGETS[category]
        values["remaining_deficit"] = max(0, CATEGORY_TARGETS[category] - values["conservative"])
        values["locked"] = values["conservative"] >= CATEGORY_TARGETS[category]
    return totals


def validate_overlap_plan(source: dict[str, Any]) -> None:
    if source["approval_state"] != "approved":
        return
    overlap = source.get("overlap_plan", {})
    missing = OVERLAP_FIELDS - set(overlap)
    if missing or any(not overlap.get(field) for field in OVERLAP_FIELDS):
        raise ValueError(f"{source.get('id')} has incomplete overlap plan: {sorted(missing)}")


def validate_code_policy(source: dict[str, Any]) -> None:
    if "code_structured" not in source.get("category", []):
        return
    policy = source.get("code_policy", {})
    required = {
        "license_scope", "language_mix", "exclude_generated_vendor_minified",
        "exclude_test_fixtures", "secret_scan", "pii_scan", "dependency_metadata",
        "fork_dedup", "benchmark_solution_filter", "max_files_per_project",
        "max_code_to_natural_language_ratio",
    }
    if required - set(policy):
        raise ValueError(f"{source.get('id')} has incomplete code policy")
    if source["approval_state"] == "approved" and policy["license_scope"] in {"unknown", "repository_visibility"}:
        raise ValueError(f"approved code source {source.get('id')} lacks identifiable licensing")


def validate_pii_policy(source: dict[str, Any]) -> None:
    risk = source.get("risk", {})
    required = {"pii", "private_data", "unsafe_content", "benchmark_contamination", "copyrighted_long_form", "boilerplate", "spam"}
    if required - set(risk):
        raise ValueError(f"{source.get('id')} has incomplete risk assessment")
    if risk["pii"] == "high" and source["approval_state"] == "approved" and not source.get("pii_mitigation"):
        raise ValueError(f"high-PII source {source.get('id')} cannot be approved without mitigation")


def validate_concentration(entries: list[dict[str, Any]], tranche_tokens: int = 200_000_000) -> dict[str, Any]:
    dataset_limit = int(tranche_tokens * 0.15)
    family_limit = int(tranche_tokens * 0.20)
    code_limit = int(tranche_tokens * 0.05)
    bilingual_limit = int(tranche_tokens * 0.05)
    families: defaultdict[str, int] = defaultdict(int)
    violations: list[str] = []
    for entry in entries:
        cap = entry["post_filter_cap_tokens"]
        families[entry["source_family"]] += cap
        if cap > dataset_limit:
            violations.append(f"dataset:{entry['source_id']}")
        if entry.get("single_code_ecosystem") and cap > code_limit:
            violations.append(f"code:{entry['source_id']}")
        if entry.get("bilingual_source") and cap > bilingual_limit:
            violations.append(f"bilingual:{entry['source_id']}")
        if entry.get("generated_text") and cap:
            violations.append(f"generated:{entry['source_id']}")
    violations.extend(f"family:{family}" for family, cap in families.items() if cap > family_limit)
    return {"result": "PASS" if not violations else "FAIL", "violations": sorted(violations)}


def classify_source_lock(
    approved_expected: int,
    approved_conservative: int,
    categories: dict[str, dict[str, Any]],
    concentration_passes: bool,
    acquisition_manifest_complete: bool,
    storage_feasible: bool,
    evidence_materially_improved: bool = True,
) -> str:
    categories_locked = all(item.get("locked") is True for item in categories.values())
    if (
        approved_expected >= 250_000_000
        and approved_conservative >= 200_000_000
        and categories_locked
        and concentration_passes
        and acquisition_manifest_complete
        and storage_feasible
    ):
        return "LOCKED"
    if evidence_materially_improved:
        return "PARTIALLY LOCKED"
    return "NOT LOCKED"


def assert_planning_only(payload: dict[str, Any]) -> None:
    if payload.get("planning_only") is not True:
        raise ValueError("Phase 5B artifact must be planning-only")
    for field in ("downloads_performed", "scraping_performed", "execution_authorized", "training_authorized"):
        if field in payload and payload[field] is not False:
            raise ValueError(f"Phase 5B forbids {field}")
