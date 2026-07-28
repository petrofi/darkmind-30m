"""Build redacted Phase 5D source-lock artifacts from bounded runtime evidence.

This module performs no network, corpus-construction, token-sharding, or
training work. Runtime samples remain outside the repository.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from darkmind_v2.corpus.phase5b_source_lock import CATEGORY_TARGETS, category_capacity


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "darkmind_v2" / "corpus"
CONFIG = ROOT / "darkmind_v2" / "config"
REPORTS = ROOT / "darkmind_v2" / "reports"
RUNTIME = Path(r"C:\DarkMindRuntime\phase5d")
SUMMARY_PATH = RUNTIME / "reports" / "sample_run_summary.json"
REGISTRY_PATH = CORPUS / "source_registry.v4.candidates.json"
PLAN_PATH = CONFIG / "corpus_v4_acquisition_plan.json"
ATTRIBUTION_PATH = CONFIG / "corpus_v4_attribution_manifest.json"
RESULTS_PATH = CONFIG / "corpus_v4_sample_inspection_results.json"
ACCESS_DATE = "2026-07-23"
DECISION = "DARKMIND V2 CORPUS V4 REQUIRES A REDUCED OPEN-ONLY TARGET OR LICENSED DATA PARTNERSHIPS"

DECISIONS = {
    "govuk_content_ogl3_20260722": (
        "conditional",
        "The bounded API response proves metadata extraction, not a reproducible rights-filtered content inventory or portal capacity.",
    ),
    "govinfo_federal_register_2025_xml": (
        "deferred",
        "The official bulk-data endpoint returned an HTML service error and no XML inventory, so artifact identity and extraction yield remain unproven.",
    ),
    "plos_ccby_jats_allowlist": (
        "conditional",
        "The bounded search response contains metadata and sparse abstracts; item-level CC BY JATS identity and useful full-text yield remain unresolved.",
    ),
    "go_1_26_5_source_docs": (
        "approved",
        "The exact official archive, published SHA-256, project license, deterministic extraction, quality, safety filtering, and Corpus V3 overlap checks passed.",
    ),
    "kubernetes_website_f2987ba": (
        "approved",
        "The exact official commit, CC BY 4.0 website scope, deterministic English-content extraction, safety filtering, and Corpus V3 overlap checks passed.",
    ),
    "nodejs_24_18_0_source_docs": (
        "conditional",
        "Useful yield is demonstrated, but bundled third-party license mapping and the elevated credential/private-key pattern rejection require a path allowlist.",
    ),
}

APPROVED_ALLOCATIONS = {
    "go_1_26_5_source_docs": {
        "technical_documentation": {"expected_tokens": 3_180, "conservative_tokens": 1_908},
        "code_structured_text": {"expected_tokens": 7_496_820, "conservative_tokens": 4_498_092},
    },
    "kubernetes_website_f2987ba": {
        "technical_documentation": {"expected_tokens": 5_645_265, "conservative_tokens": 4_233_948},
    },
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_by_id(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    return next(item for item in registry["sources"] if item["id"] == source_id)


def checksum_result(source_id: str) -> dict[str, Any]:
    manifest = load(RUNTIME / "manifests" / f"{source_id}_raw_manifest.json")
    request = manifest["requests"][-1]
    checksums = manifest.get("checksums", [])
    if checksums:
        check = checksums[-1]
        return {
            "algorithm": check["algorithm"],
            "result": check["result"],
            "local_sha256": request["sha256"],
            "published_checksum_verified": bool(check.get("published") or check["algorithm"] == "git_commit"),
        }
    return {
        "algorithm": "local_sha256",
        "result": "LOCAL_SHA256_ONLY",
        "local_sha256": request["sha256"],
        "published_checksum_verified": False,
    }


def build_results(summary: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for metrics in summary["sources"]:
        status, reason = DECISIONS[metrics["source_id"]]
        sources.append({
            **copy.deepcopy(metrics),
            "checksum": checksum_result(metrics["source_id"]),
            "final_status": status,
            "decision_reason": reason,
        })
    return {
        "schema_version": "darkmind-v2-corpus-v4-sample-inspection-results-v1",
        "authorization_id": summary["authorization_id"],
        "inspection_date": ACCESS_DATE,
        "selection_seed": summary["selection_seed"],
        "runtime_evidence_root": str(RUNTIME),
        "runtime_samples_committed": False,
        "sampled_candidates": len(sources),
        "total_downloaded_bytes": summary["total_downloaded_bytes"],
        "maximum_total_downloaded_bytes": summary["maximum_total_downloaded_bytes"],
        "corpus_v3_reference_documents": summary["corpus_v3_reference_documents"],
        "corpus_v3_near_reference_documents": summary["corpus_v3_near_reference_documents"],
        "downloaded_content_executed": False,
        "training_use": False,
        "production_tokenization": False,
        "sources": sources,
    }


def update_sampled_sources(registry: dict[str, Any], results: dict[str, Any]) -> None:
    inspections = {item["source_id"]: item for item in results["sources"]}
    for source_id, inspection in inspections.items():
        source = source_by_id(registry, source_id)
        status, reason = DECISIONS[source_id]
        source["approval_state"] = status
        source["capacity"] = {
            **copy.deepcopy(inspection["capacity"]),
            "expected_rejection_percent": round(
                100 * (1 - inspection["capacity"]["expected_tokens"] / max(1, inspection["capacity"]["optimistic_tokens"])),
                2,
            ),
            "confidence": "medium" if inspection["capacity"]["sample_documents"] else "low",
            "basis": inspection["capacity"]["basis"] + "; Phase 5D bounded inspection evidence",
        }
        for transient in (
            "inventory_count", "sample_documents", "sample_coverage", "uncertainty_band",
            "extraction_loss_rate", "quality_filter_loss_rate", "pii_secret_loss_rate",
            "internal_dedup_loss_rate", "corpus_v3_overlap_loss_rate",
        ):
            source["capacity"].pop(transient, None)
        source["resolution_steps"] = [] if status == "approved" else [reason]
        source["phase5d_sample_resolution"] = {
            "inspection_date": ACCESS_DATE,
            "authorization_id": results["authorization_id"],
            "sample_bytes": inspection["raw_bytes"],
            "sample_inventory_count": inspection["capacity"]["inventory_count"],
            "sample_documents": inspection["capacity"]["sample_documents"],
            "accepted_documents": inspection["accepted_documents"],
            "post_corpus_v3_overlap_tokens": inspection["tokenizer"]["post_corpus_v3_overlap_tokens"],
            "capacity": copy.deepcopy(inspection["capacity"]),
            "checksum": copy.deepcopy(inspection["checksum"]),
            "final_status": status,
            "decision_reason": reason,
            "training_use": False,
        }
        if status == "approved":
            source["acceptance_gates"] = {key: True for key in source["acceptance_gates"]}
            source["approved_category_capacity"] = copy.deepcopy(APPROVED_ALLOCATIONS[source_id])
        else:
            source.pop("approved_category_capacity", None)
        if source.get("phase5c_discovery"):
            evidence = source["new_candidate_evidence"]
            evidence["approval_status"] = status
            evidence["optimistic_tokens"] = inspection["capacity"]["optimistic_tokens"]
            evidence["expected_tokens"] = inspection["capacity"]["expected_tokens"]
            evidence["conservative_post_filter_unique_tokens"] = inspection["capacity"]["conservative_tokens"]
            evidence["capacity_basis"] = source["capacity"]["basis"]
            evidence["confidence_level"] = source["capacity"]["confidence"]
            evidence["unresolved_questions"] = [] if status == "approved" else [reason]

    go = source_by_id(registry, "go_1_26_5_source_docs")
    go["official_evidence"].update({
        "expected_raw_size": "34140216 bytes observed",
        "expected_document_count": "5593 eligible archive files; 500 deterministically sampled",
        "expected_language_distribution": {"code": 0.999576, "en": 0.000424},
        "checksum_or_manifest": "sha256:495be4bc87176ac567392e5b4116abd98466d33d7b49d41e764ccc6976b2dc42 verified",
        "update_status": "Phase 5D bounded sample approved 2026-07-23",
    })
    kubernetes = source_by_id(registry, "kubernetes_website_f2987ba")
    kubernetes["official_evidence"].update({
        "official_dataset_url": "https://github.com/kubernetes/website/archive/f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
        "expected_raw_size": "334486939 bytes observed",
        "expected_document_count": "2269 eligible English documentation files; 500 deterministically sampled",
        "expected_language_distribution": {"en": 1.0},
        "checksum_or_manifest": "git commit f2987ba1cceaa85fcd44cd1a221010d745d7335c; local archive sha256:e1f9c43641e8d85489041b1e6c193f55cb33cfedf92b275256de9b76d0a42798",
        "update_status": "Phase 5D bounded sample approved 2026-07-23",
    })
    govinfo = source_by_id(registry, "govinfo_federal_register_2025_xml")
    govinfo["acceptance_gates"]["artifact_identity_clear"] = False
    govinfo["acceptance_gates"]["snapshot_reproducible"] = False
    govinfo["acceptance_gates"]["quality_suitable"] = False
    govinfo["acceptance_gates"]["overlap_manageable"] = False
    govinfo["official_evidence"]["update_status"] = "Phase 5D official endpoint returned HTML service error; deferred 2026-07-23"


def build_registry(registry: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    registry = copy.deepcopy(registry)
    predecessor = registry.get("phase5d_predecessor") or {
        "registry_id": registry["registry_id"],
        "sha256": sha256(REGISTRY_PATH),
    }
    update_sampled_sources(registry, results)
    registry.update({
        "schema_version": "darkmind-v2-source-registry-v4-candidates-v4",
        "registry_id": "corpus-v4-phase5d-bounded-sample-resolution-20260723",
        "access_date": ACCESS_DATE,
        "bounded_sample_inspection_completed": True,
        "bounded_sample_downloaded_bytes": results["total_downloaded_bytes"],
        "bounded_sample_results_path": "darkmind_v2/config/corpus_v4_sample_inspection_results.json",
        "conditional_capacity_counted_as_approved": False,
        "acquisition_enabled": False,
        "source_lock_classification": "OPEN-ONLY LIMITED",
        "source_lock_decision": DECISION,
        "phase5d_predecessor": predecessor,
    })
    phase5d_ids = set(APPROVED_ALLOCATIONS)
    caps = [item for item in registry["approved_acquisition_caps"] if item["source_id"] not in phase5d_ids]
    caps.extend([
        {
            "source_id": "go_1_26_5_source_docs", "source_family": "go_project",
            "post_filter_cap_tokens": 10_000_000, "single_code_ecosystem": True,
            "bilingual_source": False, "generated_text": False,
        },
        {
            "source_id": "kubernetes_website_f2987ba", "source_family": "kubernetes_project",
            "post_filter_cap_tokens": 10_000_000, "single_code_ecosystem": False,
            "bilingual_source": False, "generated_text": False,
        },
    ])
    registry["approved_acquisition_caps"] = caps
    counts = Counter(item["approval_state"] for item in registry["sources"])
    registry["candidate_counts"] = {
        "total": len(registry["sources"]),
        "approved": counts["approved"],
        "conditional": counts["conditional"],
        "deferred": counts["deferred"],
        "rejected": counts["rejected"],
        "new": sum(bool(item.get("phase5c_discovery")) for item in registry["sources"]),
        "phase5b_conditionals_resolved": sum(bool(item.get("phase5c_resolution")) for item in registry["sources"]),
        "phase5d_sampled": len(results["sources"]),
    }
    approved_expected = sum(
        item["capacity"]["expected_tokens"] for item in registry["sources"] if item["approval_state"] == "approved"
    )
    approved_conservative = sum(
        item["capacity"]["conservative_tokens"] for item in registry["sources"] if item["approval_state"] == "approved"
    )
    registry["capacity_model"] = {
        "formal_thresholds": {"expected_tokens": 250_000_000, "conservative_tokens": 200_000_000},
        "preferred_reserve_thresholds": {"expected_tokens": 275_000_000, "conservative_tokens": 220_000_000},
        "approved": {"expected_tokens": approved_expected, "conservative_tokens": approved_conservative},
        "formal_deficit": {
            "expected_tokens": 250_000_000 - approved_expected,
            "conservative_tokens": 200_000_000 - approved_conservative,
        },
        "preferred_reserve_deficit": {
            "expected_tokens": 275_000_000 - approved_expected,
            "conservative_tokens": 220_000_000 - approved_conservative,
        },
        "largest_defensible_open_only_tranche_tokens": approved_conservative,
        "conditional_capacity_is_scenario_only": True,
    }
    categories = category_capacity(registry["sources"])
    registry["exclusive_category_allocation"] = {
        name: {
            "target": values["target"],
            "expected": values["expected"],
            "conservative": values["conservative"],
            "deficit": values["remaining_deficit"],
            "reserve_tokens": values["conservative"] - values["target"],
            "locked": values["locked"],
        }
        for name, values in categories.items()
    }
    registry["storage_projection"] = {
        "approved_raw_min_bytes": 641_039_725,
        "approved_raw_max_bytes": 1_591_039_725,
        "fits_external_ssd_plan": True,
        "conditional_sources_excluded": True,
        "full_acquisition_authorized": False,
    }
    return registry


def acquisition_entries() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "go_1_26_5_source_docs",
            "source_family": "go_project",
            "official_url": "https://go.dev/dl/go1.26.5.src.tar.gz",
            "snapshot_version": "go1.26.5.src.tar.gz",
            "expected_filename": "go1.26.5.src.tar.gz",
            "expected_size": {"min_bytes": 34_140_216, "max_bytes": 34_140_216},
            "expected_checksum": {
                "algorithm": "sha256",
                "value": "495be4bc87176ac567392e5b4116abd98466d33d7b49d41e764ccc6976b2dc42",
            },
            "license_identity": "Go project BSD-style license with path-level notices",
            "license_evidence_url": "https://go.dev/LICENSE",
            "attribution_record": {
                "creator": "The Go Authors",
                "source_rule": "archive path and Go 1.26.5 release",
                "license": "Go project BSD-style license",
                "notices_required": True,
            },
            "download_command_template": "curl --fail --location --retry 2 --output {destination} https://go.dev/dl/go1.26.5.src.tar.gz",
            "retry_policy": {"max_attempts": 3, "backoff_seconds": 30, "resume_partial": True},
            "rate_limit": "one HTTPS archive request",
            "destination_path": "$EXTERNAL_SSD_ROOT\\DarkMindArchive\\corpus-v4\\raw\\go_1_26_5_source_docs\\go1.26.5.src.tar.gz",
            "classification": "immutable_raw",
            "approval": {
                "source_state": "approved",
                "evidence_review": "Phase 5D bounded sample passed; full acquisition still requires separate authorization",
                "review_date": ACCESS_DATE,
                "execution_approved": False,
            },
            "post_filter_cap_tokens": 10_000_000,
            "single_code_ecosystem": True,
            "bilingual_source": False,
            "generated_text": False,
            "allowed_redirect_hosts": ["go.dev"],
            "content_execution_allowed": False,
        },
        {
            "source_id": "kubernetes_website_f2987ba",
            "source_family": "kubernetes_project",
            "official_url": "https://github.com/kubernetes/website/archive/f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
            "snapshot_version": "f2987ba1cceaa85fcd44cd1a221010d745d7335c",
            "expected_filename": "kubernetes-website-f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
            "expected_size": {"min_bytes": 334_486_939, "max_bytes": 334_486_939},
            "expected_checksum": {"algorithm": "git_commit", "value": "f2987ba1cceaa85fcd44cd1a221010d745d7335c"},
            "license_identity": "CC BY 4.0 for English website content subject to path-level notices",
            "license_evidence_url": "https://github.com/kubernetes/website/blob/f2987ba1cceaa85fcd44cd1a221010d745d7335c/LICENSE",
            "attribution_record": {
                "creator": "The Kubernetes Authors",
                "source_rule": "English documentation path and exact commit",
                "license": "CC BY 4.0",
                "modification_notice_required": True,
            },
            "download_command_template": "curl --fail --location --retry 2 --output {destination} https://github.com/kubernetes/website/archive/f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
            "retry_policy": {"max_attempts": 3, "backoff_seconds": 60, "resume_partial": True},
            "rate_limit": "one HTTPS archive request",
            "destination_path": "$EXTERNAL_SSD_ROOT\\DarkMindArchive\\corpus-v4\\raw\\kubernetes_website_f2987ba\\kubernetes-website-f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
            "classification": "immutable_raw",
            "approval": {
                "source_state": "approved",
                "evidence_review": "Phase 5D bounded sample passed; full acquisition still requires separate authorization",
                "review_date": ACCESS_DATE,
                "execution_approved": False,
            },
            "post_filter_cap_tokens": 10_000_000,
            "single_code_ecosystem": False,
            "bilingual_source": False,
            "generated_text": False,
            "allowed_redirect_hosts": ["github.com"],
            "content_execution_allowed": False,
        },
    ]


def build_plan(plan: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    predecessor = result.get("predecessor") if result["plan_id"].endswith("phase5d-20260723") else {
        "plan_id": result["plan_id"],
        "sha256": sha256(PLAN_PATH),
    }
    phase5d_ids = set(APPROVED_ALLOCATIONS)
    result.update({
        "plan_id": "darkmind-v2-corpus-v4-approved-sources-phase5d-20260723",
        "revised_date": ACCESS_DATE,
        "predecessor": predecessor,
        "entries": [
            *(item for item in result["entries"] if item["source_id"] not in phase5d_ids),
            *acquisition_entries(),
        ],
        "acquisition_enabled": False,
        "execution_authorized": False,
    })
    result["execution_controls"]["maximum_total_bytes"] = sum(
        item["expected_size"]["max_bytes"] for item in result["entries"]
    )
    assert {item["source_id"] for item in result["entries"]} == {
        item["id"] for item in registry["sources"] if item["approval_state"] == "approved"
    }
    return result


def attribution_entries() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "go_1_26_5_source_docs",
            "official_source_name": "Go 1.26.5 source and documentation",
            "snapshot_version": "go1.26.5.src.tar.gz",
            "official_url": "https://go.dev/dl/go1.26.5.src.tar.gz",
            "license_identity": "Go project BSD-style license with path-level notices",
            "license_evidence_url": "https://go.dev/LICENSE",
            "attribution_record": {
                "creator": "The Go Authors",
                "source_rule": "archive path and Go 1.26.5 release",
                "license": "Go project BSD-style license",
                "notices_required": True,
            },
            "modification_notice": "Selected, normalized, filtered, deduplicated, and token-counted for research corpus planning.",
            "record_key_template": "{source_id}:{snapshot_version}:{document_path_or_url}",
            "approval_state": "approved",
            "acquisition_execution_authorized": False,
        },
        {
            "source_id": "kubernetes_website_f2987ba",
            "official_source_name": "Kubernetes English documentation website",
            "snapshot_version": "f2987ba1cceaa85fcd44cd1a221010d745d7335c",
            "official_url": "https://github.com/kubernetes/website/archive/f2987ba1cceaa85fcd44cd1a221010d745d7335c.tar.gz",
            "license_identity": "CC BY 4.0 for English website content subject to path-level notices",
            "license_evidence_url": "https://github.com/kubernetes/website/blob/f2987ba1cceaa85fcd44cd1a221010d745d7335c/LICENSE",
            "attribution_record": {
                "creator": "The Kubernetes Authors",
                "source_rule": "English documentation path and exact commit",
                "license": "CC BY 4.0",
                "modification_notice_required": True,
            },
            "modification_notice": "Selected, normalized, filtered, deduplicated, and token-counted for research corpus planning.",
            "record_key_template": "{source_id}:{snapshot_version}:{document_path_or_url}",
            "approval_state": "approved",
            "acquisition_execution_authorized": False,
        },
    ]


def build_attribution(manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    phase5d_ids = set(APPROVED_ALLOCATIONS)
    result.update({
        "manifest_id": "darkmind-v2-corpus-v4-attribution-phase5d-20260723",
        "acquisition_plan_id": plan["plan_id"],
        "entries": [
            *(item for item in result["entries"] if item["source_id"] not in phase5d_ids),
            *attribution_entries(),
        ],
        "execution_authorized": False,
    })
    result["coverage"] = {
        "approved_sources": len(result["entries"]),
        "manifest_entries": len(result["entries"]),
        "complete": True,
    }
    return result


def table_rows(results: dict[str, Any]) -> list[str]:
    rows = []
    for item in results["sources"]:
        extraction = item["extraction"]
        rows.append(
            f"| `{item['source_id']}` | {item['raw_bytes']:,} | {extraction['selected_records']} | "
            f"{extraction['extracted_records']} | {extraction['extraction_success_rate']:.1%} | "
            f"{item['accepted_documents']} | {item['tokenizer']['post_corpus_v3_overlap_tokens']:,} | "
            f"{item['final_status'].title()} |"
        )
    return rows


def write_reports(results: dict[str, Any], registry: dict[str, Any]) -> None:
    by_id = {item["source_id"]: item for item in results["sources"]}
    rows = "\n".join(table_rows(results))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase5d_sample_acquisition_summary.md").write_text(
        f"""# Phase 5D Bounded Sample Acquisition Summary

## Scope and controls

- Authorization: `{results['authorization_id']}`
- Selection seed: `{results['selection_seed']}`
- Official candidates sampled: {results['sampled_candidates']} of 8 maximum
- Downloaded bytes: {results['total_downloaded_bytes']:,} of 10,000,000,000 maximum
- Per-source maximum: 2,000,000,000 bytes; every source passed
- Runtime root: `C:\\DarkMindRuntime\\phase5d`
- Samples committed to Git: no
- Downloaded content executed: no
- Training or production tokenization: no

| Source | Bytes | Selected | Extracted | Success | Accepted | Post-overlap tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
{rows}

Go and Node.js published SHA-256 checks passed. Kubernetes exact-commit identity and local archive SHA-256 passed.
Dynamic API responses and the GovInfo service response have local SHA-256 records but no published checksum.
No redirect left an authorized domain. No unofficial mirror was used.
""",
        encoding="utf-8",
    )
    extraction_rows = []
    for item in results["sources"]:
        ext, text, quality, safety = item["extraction"], item["text"], item["quality"], item["safety"]
        extraction_rows.append(
            f"| `{item['source_id']}` | {ext['extraction_success_rate']:.1%} | {ext['empty_extraction_rate']:.1%} | "
            f"{ext['parse_failure_rate']:.1%} | {quality['quality_rejected_documents']} | {text['boilerplate_hits']} | "
            f"{safety['emails']} | {safety['telephone_numbers']} | {safety['credential_patterns']} | {safety['private_keys']} |"
        )
    (REPORTS / "phase5d_extraction_quality.md").write_text(
        """# Phase 5D Extraction and Quality Inspection

| Source | Extraction success | Empty | Parse failure | Quality rejected | Boilerplate hits | Emails | Phone patterns | Credential patterns | Private keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(extraction_rows) + """

All inspected files were digitally extractable; OCR was not used. Archive adapters opened only the deterministic selected members and never extracted or executed packages.
Quality scoring covered sentence completeness, alphabetic and symbol balance, repetition, minimum length, and template density.
Code and prose were assigned mutually exclusively. Go was 99.9576% code by raw sample tokens; Kubernetes was 100% English technical documentation; Node.js was 52.7365% code and 47.2635% technical documentation.
GOV.UK supplied short English metadata/descriptions rather than full publication bodies. PLOS supplied sparse metadata/abstract fields, with 80% empty extraction and 189 quality rejections.
Telephone and credential-pattern counts are conservative pattern hits, including code-shaped false positives. Matching documents were excluded where policy required. No raw personal data or secret value is reproduced here.
""",
        encoding="utf-8",
    )
    capacity_rows = []
    for item in results["sources"]:
        cap = item["capacity"]
        capacity_rows.append(
            f"| `{item['source_id']}` | {cap['sample_coverage']:.2%} | {cap['optimistic_tokens']:,} | "
            f"{cap['expected_tokens']:,} | {cap['conservative_tokens']:,} | "
            f"{cap['extraction_loss_rate']:.2%} | {cap['quality_filter_loss_rate']:.2%} | "
            f"{cap['pii_secret_loss_rate']:.2%} | {cap['internal_dedup_loss_rate']:.2%} | "
            f"{cap['corpus_v3_overlap_loss_rate']:.2%} |"
        )
    (REPORTS / "phase5d_sample_capacity_model.md").write_text(
        """# Phase 5D Sample Capacity Model

| Source | Coverage | Optimistic | Expected | Conservative | Extraction loss | Quality loss | PII/secret loss | Internal dedup loss | V3 overlap loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(capacity_rows) + f"""

Portal-wide extrapolation is prohibited. GOV.UK and PLOS capacities cover only the exact bounded API responses. GovInfo has zero evidenced capacity because no XML artifact was obtained.
Only Approved source capacity enters the lock: expected {registry['capacity_model']['approved']['expected_tokens']:,}; conservative {registry['capacity_model']['approved']['conservative_tokens']:,}.
The uncertainty bands remain in `corpus_v4_sample_inspection_results.json`.
""",
        encoding="utf-8",
    )
    overlap_rows = []
    for item in results["sources"]:
        dedup = item["deduplication"]
        overlap_rows.append(
            f"| `{item['source_id']}` | {dedup['internal_and_cross_source_exact_duplicates']} | "
            f"{dedup['internal_and_cross_source_near_duplicates']} | "
            f"{dedup['corpus_v3_exact_overlap_documents']} | {dedup['corpus_v3_near_overlap_documents']} |"
        )
    (REPORTS / "phase5d_corpus_v3_overlap_results.md").write_text(
        """# Phase 5D Corpus V3 Overlap Results

| Source | Internal/cross exact | Internal/cross near | Corpus V3 exact | Corpus V3 near |
|---|---:|---:|---:|---:|
""" + "\n".join(overlap_rows) + f"""

Corpus V3 exact comparison covered {results['corpus_v3_reference_documents']:,} accepted documents.
Near comparison used a deterministic 1/16 hash sample of {results['corpus_v3_near_reference_documents']:,} documents and 64-bit SimHash with maximum Hamming distance 3.
Exact normalized SHA-256, URL/document identity, near fingerprints, and token losses were kept separate. No sample matched Corpus V3 exactly or within the configured near threshold. Corpus V3 was read only and unchanged.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_turkish_data_feasibility.md").write_text(
        """# Phase 5D Turkish Data Feasibility

## Explicit answers

1. The 72M conservative Turkish general/educational target cannot be reached from currently approved open official artifacts.
2. Approved Turkish general/educational capacity is **0 tokens**.
3. Identified conditional Turkish scenario capacity is **11.5M conservative tokens**: 1.5M from the Turkish librarianship candidate and 10M from the rights-filtered DergiPark candidate. Conditional capacity is not approved.
4. Blockers are article/issue-level license identity, platform-versus-content rights, reproducible manifests, third-party exceptions, deduplication, and defensible post-filter yield.
5. Public-domain Turkish books are not sufficient: no reproducible rights-cleared allowlist with demonstrated capacity is approved.
6. Government and institutional publications are not sufficiently licensed as a broad class; written source-specific permission or explicit artifact licensing is required.
7. Existing candidates are too narrow, formal, license-fragmented, or potentially overlapping to cover the target.
8. Direct licensing or institutional partnership is required unless the Turkish tranche is explicitly reduced.

## Scenario T1 - Open-only

Achievable conservative approved Turkish capacity: **0**. Deficit: **72M**.

## Scenario T2 - Open plus institutional permission

The two identified conditional sources could contribute up to **11.5M conservative tokens** after written permission, exact allowlisting, and quality/overlap validation. At least **60.5M additional Turkish capacity** would still require further licensed partners. Outreach is required; cooperation is not assumed.

## Scenario T3 - Reduced Turkish tranche

Arithmetic removal of the unavailable 72M Turkish target reduces the nominal 200M plan to 128M, but this would erase the project priority and is not accepted silently. The currently approved 15.034M technical/code tranche is not a balanced substitute.

Missing Phase 5A human review scores do not change this legal and capacity conclusion. No human score is invented.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_bilingual_data_feasibility.md").write_text(
        """# Phase 5D Bilingual Data Feasibility

- Approved Turkish-English aligned capacity: **0 tokens**.
- Conditional capacity with artifact-level alignment and licensing evidence: **0 tokens currently quantified**.
- The 10M target is not realistic from the present registry.
- Source and target sides can have asymmetric rights; both sides and the alignment metadata must permit machine processing and model training.
- Wikipedia-derived pairs risk duplicating Corpus V3 and cannot be used to fill the gap.
- Translation-memory boilerplate, weak alignment, and machine-translated filler must be rejected.
- Bilingual material remains a separate mutually exclusive category until alignment and rights pass; it must not be double-counted in Turkish or English totals.

Licensed bilingual contributions or an institutionally cleared translation-memory partnership are required.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_turkish_data_partnership_plan.md").write_text(
        """# Phase 5D Turkish Data Partnership Plan

Potential partner categories are universities, open-course publishers, educational foundations, public institutions, publishers, technical communities, and content owners. This list does not imply willingness to cooperate.

Priority is rights-cleared Turkish general and educational prose, followed by genuinely aligned Turkish-English material. Each discussion must request machine processing, model training, internal storage, creation and publication of derived model weights, attribution terms, raw-text redistribution or explicit non-redistribution, retention period, auditability, withdrawal procedure, and treatment of already trained weights.

The operational path is: identify a rights holder; request an exact artifact/version; obtain written rights; build an immutable manifest; run a small bounded inspection; perform quality, PII, contamination, and Corpus V3 overlap checks; then seek a separate acquisition decision.

No outreach was sent. This plan is not legal advice.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_data_license_request_template.md").write_text(
        """# Phase 5D Data License Request Template

Subject: Request for permission to use [ARTIFACT] in DarkMind v2 research training

We request written permission for the exact artifact/version `[IDENTITY]` to be machine processed, stored internally, filtered, normalized, deduplicated, and used to train language models. Please state whether derived model weights may be published and whether raw text may be redistributed or must remain private.

Please specify: rights holder; covered files; excluded third-party material; machine-processing rights; model-training rights; internal storage; derived-weight creation and publication; required attribution; raw-text redistribution rule; retention period; security requirements; withdrawal procedure; and the effect of withdrawal on already trained weights.

We will preserve an immutable manifest, source/version identifiers, checksums, attribution records, and modification notices. We will not infer permission from public accessibility.

This template is not legal advice and no request has been sent.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_dataset_contribution_requirements.md").write_text(
        """# Phase 5D Dataset Contribution Requirements

A contribution must include the rights holder, exact artifact/version, file inventory, official delivery route, checksums, content and collection licenses, third-party exclusions, language/category metadata, attribution text, and a contact for rights questions.

Required permissions cover machine processing, model training, internal storage, derived model weights, weight publication, raw-text redistribution or non-redistribution, retention, and withdrawal. Contributions must permit deterministic sampling and safety review.

Technical intake requires UTF-8 or documented encoding, stable document IDs, provenance URLs, no executable requirement, and enough metadata for exact/near deduplication. PII, secrets, answer keys, benchmark solutions, spam, generated filler, and unclear-license records are rejected or quarantined.

Acceptance requires bounded inspection before full acquisition. Public accessibility alone is not permission.
""",
        encoding="utf-8",
    )
    allocations = registry["exclusive_category_allocation"]
    (REPORTS / "phase5d_corpus_v4_strategy_scenarios.md").write_text(
        f"""# Phase 5D Corpus V4 Strategy Scenarios

## Strategy A - Full 200M open-only tranche

Not feasible. Approved conservative capacity is **{registry['capacity_model']['approved']['conservative_tokens']:,}**, leaving **{registry['capacity_model']['formal_deficit']['conservative_tokens']:,}** tokens uncovered. No formal category target is locked.

## Strategy B - Smaller open-only tranche

The largest evidence-backed tranche is exactly the sum of approved conservative capacities: **{registry['capacity_model']['largest_defensible_open_only_tranche_tokens']:,} tokens**. It is technical/code-heavy and is not a balanced 200M replacement.

## Strategy C - Full 200M with licensed partnerships

Approved open contribution: **{registry['capacity_model']['approved']['conservative_tokens']:,}**. Required additional conservative capacity: **{registry['capacity_model']['formal_deficit']['conservative_tokens']:,}**, including {allocations['turkish_general_educational']['deficit']:,} Turkish general, {allocations['english_general_educational']['deficit']:,} English general, and {allocations['controlled_bilingual']['deficit']:,} bilingual tokens. Written rights, immutable artifact delivery, attribution, and bounded validation are dependencies.

## Strategy D - Technical-heavy interim tranche

Go and Kubernetes improve technical/code evidence, but an interim technical-heavy tranche would deepen the Turkish, English-general, and bilingual imbalance. It is useful for inspection or a separately authorized experiment, not recommended as the default Corpus V4 target merely because acquisition is easier.
""",
        encoding="utf-8",
    )
    (REPORTS / "phase5d_source_lock_decision.md").write_text(
        f"""# Phase 5D Source Lock Decision

## OPEN-ONLY LIMITED

Bounded evidence promoted Go 1.26.5 and Kubernetes website documentation to Approved. GovInfo moved to Deferred; GOV.UK, PLOS, and Node.js remain Conditional.

- Approved expected capacity: {registry['capacity_model']['approved']['expected_tokens']:,} / 250,000,000
- Approved conservative capacity: {registry['capacity_model']['approved']['conservative_tokens']:,} / 200,000,000
- Largest defensible open-only tranche: {registry['capacity_model']['largest_defensible_open_only_tranche_tokens']:,}
- Turkish general approved: 0 / 72,000,000
- English general approved: 0 / 50,000,000
- Controlled bilingual approved: 0 / 10,000,000
- Attribution coverage: 5/5 approved sources
- Full acquisition enabled: no

The missing human review packet score does not alter the legal, reproducibility, category, or capacity deficits. Human review remains required before later corpus use.

Next action: seek written Turkish/general and bilingual data rights while resolving exact artifact allowlists for the strongest remaining conditional sources. Any further download needs a new bounded authorization.

{DECISION}
""",
        encoding="utf-8",
    )


def main() -> None:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"missing bounded sample summary: {SUMMARY_PATH}")
    summary = load(SUMMARY_PATH)
    if summary.get("result") != "PASS" or summary.get("training_use") is not False:
        raise ValueError("bounded sample evidence did not pass no-training controls")
    results = build_results(summary)
    registry = build_registry(load(REGISTRY_PATH), results)
    plan = build_plan(load(PLAN_PATH), registry)
    attribution = build_attribution(load(ATTRIBUTION_PATH), plan)
    dump(RESULTS_PATH, results)
    dump(REGISTRY_PATH, registry)
    dump(PLAN_PATH, plan)
    dump(ATTRIBUTION_PATH, attribution)
    write_reports(results, registry)
    print(json.dumps({
        "result": "PASS",
        "sampled_sources": results["sampled_candidates"],
        "downloaded_bytes": results["total_downloaded_bytes"],
        "approval_counts": registry["candidate_counts"],
        "approved_expected_tokens": registry["capacity_model"]["approved"]["expected_tokens"],
        "approved_conservative_tokens": registry["capacity_model"]["approved"]["conservative_tokens"],
        "source_lock_classification": registry["source_lock_classification"],
        "acquisition_enabled": registry["acquisition_enabled"],
        "training_use": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
