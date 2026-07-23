"""Build deterministic, planning-only Phase 5C Corpus V4 artifacts.

This module transforms the reviewed Phase 5B registry. It performs no network,
dataset, corpus, or training operations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "darkmind_v2" / "corpus"
CONFIG = ROOT / "darkmind_v2" / "config"
REPORTS = ROOT / "darkmind_v2" / "reports"
REGISTRY_PATH = CORPUS / "source_registry.v4.candidates.json"
PLAN_PATH = CONFIG / "corpus_v4_acquisition_plan.json"
ATTRIBUTION_PATH = CONFIG / "corpus_v4_attribution_manifest.json"
ACCESS_DATE = "2026-07-22"
PHASE5B_PLAN_SHA256 = "017af8d60071d47fb2e5cf3a5a40f598a95296586253ce24aba14a73b6626f68"
PHASE5B_REGISTRY_SHA256 = "6acca0c1f648d51121b775e1244190549c8e4bd9dbd109bd500f611f22a907c1"
DECISION_LINE = (
    "DARKMIND V2 CORPUS V4 SOURCE PLAN REQUIRES ADDITIONAL OFFICIAL "
    "SOURCES BEFORE ACQUISITION"
)

CATEGORY_TARGETS = {
    "turkish_general_educational": 72_000_000,
    "english_general_educational": 50_000_000,
    "technical_documentation": 42_000_000,
    "code_structured_text": 26_000_000,
    "controlled_bilingual": 10_000_000,
}

QUESTION_KEYS = (
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
)

# Every Phase 5B conditional candidate receives one terminal review outcome.
RESOLUTIONS = {
    "govuk_content_ogl3_20260722": (
        "conditional", "metadata_resolvable",
        "Pin a Content API URL inventory and exclude third-party or personal-data pages."
    ),
    "turk_kutuphaneciligi_ccby_20260722": (
        "conditional", "bounded_sample_required",
        "The exact issue/article inventory and item-level CC BY evidence remain unresolved."
    ),
    "nist_publications_cleared_subset": (
        "conditional", "metadata_resolvable",
        "Pin a publication inventory and exclude every marked third-party work."
    ),
    "eur_lex_reusable_text_20260722": (
        "conditional", "metadata_resolvable",
        "Pin a CELLAR/data-dump release and preserve document-level reuse notices."
    ),
    "dergipark_ccby_only_subset_20260722": (
        "conditional", "bounded_sample_required",
        "DergiPark is a platform, not a portal-wide content licence; journals must be allowlisted."
    ),
    "project_gutenberg_turkish_vetted": (
        "deferred", "metadata_resolvable",
        "Rights and trademark conditions are work- and jurisdiction-specific; no Turkish allowlist is pinned."
    ),
    "pmc_oa_commercial_reuse_subset": (
        "conditional", "metadata_resolvable",
        "Admit only exact OA articles carrying CC0, CC BY, CC BY-SA, or separately approved terms."
    ),
    "openstax_legacy_ccby_editions": (
        "deferred", "metadata_resolvable",
        "Legacy CC BY editions and embedded third-party exceptions require an edition-level manifest."
    ),
    "stack_exchange_ccbysa_dump": (
        "deferred", "bounded_sample_required",
        "Attribution reconstruction, PII, benchmark leakage, and current dump identity remain unresolved."
    ),
    "github_permissive_repo_allowlist": (
        "rejected", "policy_rejected",
        "Public repository visibility is not a licence and a platform-wide allowlist is not auditable."
    ),
    "arxiv_explicit_commercial_cc_subset": (
        "deferred", "bounded_sample_required",
        "Author-selected licences, version identity, long-form overlap, and benchmark leakage need a frozen allowlist."
    ),
}


def _evidence(
    dataset_url: str,
    license_urls: list[str],
    snapshot: str,
    content_license: str,
    database_license: str,
    mechanism: str,
    raw_size: str,
    documents: str,
    language: dict[str, float],
    checksum: str = "not_published; generate SHA-256 manifest before approval",
) -> dict[str, Any]:
    return {
        "access_date": ACCESS_DATE,
        "official_dataset_url": dataset_url,
        "official_license_urls": license_urls,
        "artifact_license": content_license,
        "content_license": content_license,
        "database_license": database_license,
        "exact_snapshot": snapshot,
        "download_mechanism": mechanism,
        "redistribution_rights": "must be explicit for every admitted artifact",
        "modification_rights": "must permit normalization, filtering, and derived corpus records",
        "attribution_requirements": "record creator, title, source URL, version, licence, and modifications",
        "share_alike_requirements": "preserve artifact-level obligations; unresolved items excluded",
        "noncommercial_restrictions": "NC material excluded",
        "machine_learning_restrictions": "none identified in cited licence; approval still required",
        "terms_of_service_restrictions": "official distribution route only; no scraping",
        "rate_limits": "single-worker bounded acquisition only after separate authorization",
        "checksum_or_manifest": checksum,
        "official_file_inventory": snapshot,
        "expected_raw_size": raw_size,
        "expected_document_count": documents,
        "expected_language_distribution": language,
        "update_status": f"official evidence reviewed {ACCESS_DATE}",
    }


def _risk(pii: str = "low", benchmark: str = "low", copyright: str = "low") -> dict[str, str]:
    return {
        "pii": pii,
        "private_data": "low",
        "unsafe_content": "low",
        "benchmark_contamination": benchmark,
        "copyrighted_long_form": copyright,
        "boilerplate": "medium",
        "spam": "low",
    }


def _overlap() -> dict[str, str]:
    return {
        "document_id": "stable official artifact plus internal path",
        "url": "canonical official URL comparison",
        "hash": "raw and normalized SHA-256",
        "normalized_fingerprint": "exact normalized-text digest",
        "near_dedup": "MinHash/LSH against admitted Corpus V3 and V4 records",
        "ngram_sampling": "sampled long n-gram overlap audit",
        "corpus_v3_wikimedia": "reject copied or translated Wikimedia passages",
        "corpus_v3_python_docs": "reject copied Python documentation passages",
        "benchmark": "benchmark name and prompt fingerprint denylist",
    }


def _sample_plan(source_id: str) -> dict[str, Any]:
    return {
        "execution_authorized": False,
        "maximum_download_bytes": 25_000_000,
        "exact_files": "must be named in a future approved manifest",
        "checksum_requirement": "SHA-256 before inspection",
        "destination": f"$EXTERNAL_SSD_ROOT\\DarkMindArchive\\corpus-v4\\samples\\{source_id}",
        "retention": "retain immutable sample and audit output; no training use",
        "metrics": ["licence_coverage", "language_mix", "PII", "quality", "Corpus_V3_overlap"],
        "training_authorized": False,
    }


def _resolution_record(source: dict[str, Any], status: str, kind: str, reason: str) -> dict[str, Any]:
    ev = source["official_evidence"]
    answers = {
        "exact_downloadable_artifact": ev["official_file_inventory"],
        "exact_snapshot_release_dump_tag_or_commit": ev["exact_snapshot"],
        "content_license_explicit": ev["content_license"],
        "collection_or_database_license_explicit": ev["database_license"],
        "redistribution_rights_explicit": ev["redistribution_rights"],
        "modification_and_derivative_processing_rights_explicit": ev["modification_rights"],
        "machine_processing_or_model_training_prohibited": ev["machine_learning_restrictions"],
        "attribution_obligations_executable": ev["attribution_requirements"],
        "official_acquisition_reproducible": ev["download_mechanism"],
        "checksum_or_signed_manifest_available": ev["checksum_or_manifest"],
        "language_composition_officially_supported": ev["expected_language_distribution"],
        "corpus_v3_relationship_understood": source["relationship_to_corpus_v3"],
        "conservative_capacity_estimable_without_download": source["capacity"]["basis"],
        "source_concentration_violation": source["source_cap_tokens"] > 30_000_000,
        "benchmark_pii_private_or_contamination_concerns": source["risk"],
    }
    assert tuple(answers) == QUESTION_KEYS
    record = {
        "review_date": ACCESS_DATE,
        "question_count": 15,
        "answers": answers,
        "resolution_type": kind,
        "final_status": status,
        "decision_reason": reason,
        "remaining_blockers": [] if status in {"approved", "rejected"} else [reason],
        "next_action": "none" if status == "rejected" else reason,
    }
    if kind == "bounded_sample_required":
        record["future_sample_plan"] = _sample_plan(source["id"])
    return record


def _candidate(
    *, source_id: str, name: str, url: str, metadata_url: str, license_url: str,
    snapshot: str, language: list[str], category: list[str], artifact_type: str,
    raw_min: int, raw_max: int, documents: str, optimistic: int, expected: int,
    conservative: int, cap: int, family: str, status: str, reason: str,
    content_license: str, database_license: str = "not_applicable",
    mechanism: str = "official HTTPS artifact only", confidence: str = "low",
    pii: str = "low", benchmark: str = "low", copyright: str = "low",
    checksum: str = "not_published; generate SHA-256 manifest before approval",
) -> dict[str, Any]:
    loss = {
        "extraction": 8, "language_filter": 3, "pii_secret": 2,
        "exact_dedup": 8, "near_dedup": 12, "corpus_v3_overlap": 8,
        "contamination": 4, "license_rejection": 10,
    }
    evidence = _evidence(
        url, [license_url], snapshot, content_license, database_license,
        mechanism, f"{raw_min}-{raw_max} bytes planning range", documents,
        {code: round(1 / len(language), 3) for code in language}, checksum,
    )
    gates = {
        "artifact_identity_clear": bool(snapshot and "unresolved" not in snapshot),
        "snapshot_reproducible": bool(snapshot and "unresolved" not in snapshot),
        "license_explicit": "unknown" not in content_license.lower(),
        "machine_processing_permitted": False,
        "acquisition_method_permitted": False,
        "attribution_actionable": bool(license_url),
        "quality_suitable": False,
        "overlap_manageable": False,
    }
    result: dict[str, Any] = {
        "id": source_id,
        "official_source_name": name,
        "official_url": metadata_url,
        "snapshot_or_version": snapshot,
        "language": language,
        "category": category,
        "license": content_license,
        "license_evidence_url": license_url,
        "source_cap_tokens": cap,
        "quality_tier": "candidate",
        "relationship_to_corpus_v3": "new candidate; mandatory V3 hash and near-duplicate exclusion",
        "previous_approval_state": status,
        "approval_state": status,
        "official_evidence": evidence,
        "acceptance_gates": gates,
        "capacity": {
            "optimistic_tokens": optimistic,
            "expected_tokens": expected,
            "conservative_tokens": conservative,
            "expected_rejection_percent": 55,
            "confidence": confidence,
            "basis": "official metadata plus bounded planning assumptions; no files downloaded",
        },
        "risk": _risk(pii, benchmark, copyright),
        "resolution_steps": [] if status == "rejected" else [reason],
        "overlap_plan": _overlap(),
        "phase5c_discovery": True,
        "new_candidate_evidence": {
            "stable_source_id": source_id,
            "official_name": name,
            "official_artifact_url": url,
            "official_metadata_url": metadata_url,
            "official_license_url": license_url,
            "exact_snapshot_version_commit_date": snapshot,
            "language": language,
            "category": category,
            "artifact_type": artifact_type,
            "expected_raw_bytes": {"minimum": raw_min, "maximum": raw_max, "basis": "official metadata or conservative planning range"},
            "expected_documents_or_files": documents,
            "optimistic_tokens": optimistic,
            "expected_tokens": expected,
            "conservative_post_filter_unique_tokens": conservative,
            "loss_estimates_percent": loss,
            "capacity_basis": "official metadata plus bounded planning assumptions; no acquisition",
            "confidence_level": confidence,
            "evidence_access_date": ACCESS_DATE,
            "attribution_text_or_template": "Creator; title/path; official URL; snapshot; licence; modifications",
            "source_cap_tokens": cap,
            "concentration_family": family,
            "pii_risk": pii,
            "benchmark_risk": benchmark,
            "private_data_risk": "low",
            "approval_status": status,
            "unresolved_questions": [] if status == "rejected" else [reason],
        },
    }
    if "code_structured" in category:
        result["code_policy"] = {
            "license_scope": "exact release/repository licence plus path-level exceptions",
            "language_mix": "documentation prose and source code measured separately",
            "exclude_generated_vendor_minified": True,
            "exclude_test_fixtures": True,
            "secret_scan": True,
            "pii_scan": True,
            "dependency_metadata": "record dependency and bundled third-party notices",
            "fork_dedup": "repository and normalized-file hash",
            "benchmark_solution_filter": "deny benchmark and challenge solutions",
            "max_files_per_project": 20_000,
            "max_code_to_natural_language_ratio": 0.45,
        }
    return result


def _new_candidates() -> list[dict[str, Any]]:
    return [
        _candidate(
            source_id="mozilla_common_voice_tr_26", name="Mozilla Common Voice Scripted Speech Turkish 26.0",
            url="https://mozilladatacollective.com/datasets/cmqinosfq00x4nr07gnk0rdf9",
            metadata_url="https://datacollective.mozillafoundation.org/datasets/cmqinosfq00x4nr07gnk0rdf9",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            snapshot="cv-corpus-26.0-2026-06-12; 2.78 GB; 413,915 sentences", language=["tr"],
            category=["turkish_general_educational"], artifact_type="speech metadata and prompts",
            raw_min=2_780_000_000, raw_max=2_780_000_000, documents="413915 sentences",
            optimistic=0, expected=0, conservative=0, cap=0, family="mozilla_common_voice",
            status="rejected", reason="Dataset terms forbid re-hosting and 85% of prompts derive from Turkish Wikipedia, creating unacceptable redistribution and Corpus V3 overlap risk.",
            content_license="CC0 label subject to Mozilla dataset terms and no-rehost restriction", copyright="high",
        ),
        _candidate(
            source_id="leipzig_turkish_corpora_official_route", name="Leipzig Corpora Collection Turkish download route",
            url="https://wortschatz.uni-leipzig.de/en/download/Turkish",
            metadata_url="https://www.wortschatz.uni-leipzig.de/en/usage",
            license_url="https://www.wortschatz.uni-leipzig.de/en/usage",
            snapshot="official Turkish route returned no artifact on 2026-07-22", language=["tr"],
            category=["turkish_general_educational"], artifact_type="sentence corpus",
            raw_min=0, raw_max=0, documents="no official Turkish artifact inventory",
            optimistic=0, expected=0, conservative=0, cap=0, family="leipzig_corpora",
            status="rejected", reason="No exact Turkish artifact is currently exposed and the FAQ describes copyrighted underlying web documents.",
            content_license="download pages describe CC BY, but Turkish artifact and source rights are unresolved", copyright="high",
        ),
        _candidate(
            source_id="govinfo_federal_register_2025_xml", name="GovInfo Federal Register 2025 XML bulk data",
            url="https://www.govinfo.gov/bulkdata/FR/2025/", metadata_url="https://www.govinfo.gov/developers",
            license_url="https://www.govinfo.gov/about/policies",
            snapshot="Federal Register 2025 XML bulk directory; exact file manifest unresolved", language=["en"],
            category=["english_general_educational", "technical_documentation"], artifact_type="XML bulk data",
            raw_min=500_000_000, raw_max=2_000_000_000, documents="one XML package per issue; inventory to pin",
            optimistic=45_000_000, expected=30_000_000, conservative=18_000_000, cap=30_000_000,
            family="us_federal_register", status="conditional", reason="Pin exact 2025 file inventory, checksums, notices, and reusable text fields.",
            content_license="US federal public-domain presumption with item-level third-party exceptions", confidence="medium",
        ),
        _candidate(
            source_id="bccampus_ccby_textbook_allowlist", name="BCcampus exact-edition CC BY textbook allowlist",
            url="https://open.bccampus.ca/browse-our-collection/find-open-textbooks/",
            metadata_url="https://open.bccampus.ca/help/", license_url="https://open.bccampus.ca/create-open-textbooks/creative-commons-open-licences-for-authors/",
            snapshot="exact edition allowlist unresolved", language=["en"], category=["english_general_educational"],
            artifact_type="PDF/EPUB/HTML textbooks", raw_min=1_000_000_000, raw_max=8_000_000_000,
            documents="edition-level inventory required", optimistic=40_000_000, expected=25_000_000,
            conservative=15_000_000, cap=30_000_000, family="bccampus",
            status="conditional", reason="Pin exact CC BY editions and remove third-party media and exceptions.",
            content_license="CC BY only after edition-level verification", copyright="medium",
        ),
        _candidate(
            source_id="plos_ccby_jats_allowlist", name="PLOS CC BY JATS article allowlist",
            url="https://api.plos.org/", metadata_url="https://plos.org/terms-of-use/",
            license_url="https://plos.org/terms-of-use/", snapshot="exact article and licence allowlist unresolved",
            language=["en"], category=["technical_documentation"], artifact_type="JATS XML/API records",
            raw_min=2_000_000_000, raw_max=12_000_000_000, documents="article inventory required",
            optimistic=55_000_000, expected=35_000_000, conservative=20_000_000, cap=30_000_000,
            family="plos", status="conditional", reason="Pin exact CC BY article IDs, JATS checksums, corrections, and exclusions.",
            content_license="generally CC BY or comparable; item-level notices control", benchmark="medium", copyright="medium",
        ),
        _candidate(
            source_id="go_1_26_5_source_docs", name="Go 1.26.5 source and documentation",
            url="https://go.dev/dl/go1.26.5.src.tar.gz", metadata_url="https://go.dev/dl/",
            license_url="https://go.dev/LICENSE", snapshot="go1.26.5.src.tar.gz",
            language=["en", "code"], category=["technical_documentation", "code_structured"],
            artifact_type="source tarball", raw_min=33_000_000, raw_max=33_000_000,
            documents="archive file inventory after checksum verification", optimistic=8_000_000,
            expected=5_000_000, conservative=3_000_000, cap=10_000_000, family="go_project",
            status="conditional", reason="Verify archive checksum, path licences, generated files, and final filtered capacity.",
            content_license="Go project BSD-style licence with path-level notices",
            checksum="sha256:495be4bc87176ac567392e5b4116abd98466d33d7b49d41e764ccc6976b2dc42", confidence="medium",
        ),
        _candidate(
            source_id="kubernetes_website_f2987ba", name="Kubernetes documentation website",
            url="https://github.com/kubernetes/website", metadata_url="https://github.com/kubernetes/website",
            license_url="https://github.com/kubernetes/website/blob/main/LICENSE",
            snapshot="git commit f2987ba1cceaa85fcd44cd1a221010d745d7335c", language=["en", "code"],
            category=["technical_documentation", "code_structured"], artifact_type="Git repository",
            raw_min=200_000_000, raw_max=1_500_000_000, documents="repository paths at pinned commit",
            optimistic=15_000_000, expected=10_000_000, conservative=5_000_000, cap=10_000_000,
            family="kubernetes_project", status="conditional", reason="Pin archive hash and exclude translations, generated files, vendored assets, and benchmark-like tutorials.",
            content_license="CC BY 4.0 for website content subject to path-level notices", confidence="medium",
        ),
        _candidate(
            source_id="nodejs_24_18_0_source_docs", name="Node.js 24.18.0 source and documentation",
            url="https://nodejs.org/dist/v24.18.0/node-v24.18.0.tar.xz",
            metadata_url="https://nodejs.org/en/blog/release/v24.18.0",
            license_url="https://github.com/nodejs/node/blob/v24.18.0/LICENSE",
            snapshot="node-v24.18.0.tar.xz", language=["en", "code"],
            category=["technical_documentation", "code_structured"], artifact_type="source tarball",
            raw_min=30_000_000, raw_max=30_000_000, documents="archive paths after third-party inventory",
            optimistic=18_000_000, expected=12_000_000, conservative=6_000_000, cap=10_000_000,
            family="nodejs_project", status="conditional", reason="Resolve bundled third-party licences and select only eligible documentation/source paths.",
            content_license="MIT root licence plus bundled third-party component licences",
            checksum="sha256:e94afde24db08e0c564ee7110a2d5aab51ee0059382c9fd8233c54eec47b28f9", confidence="medium",
        ),
    ]


def _json_sha(payload: dict[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _approved_totals(sources: list[dict[str, Any]]) -> dict[str, int]:
    selected = [item for item in sources if item["approval_state"] == "approved"]
    return {
        "expected_tokens": sum(item["capacity"]["expected_tokens"] for item in selected),
        "conservative_tokens": sum(item["capacity"]["conservative_tokens"] for item in selected),
    }


def _category_totals(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals = {
        key: {"target": target, "expected": 0, "conservative": 0}
        for key, target in CATEGORY_TARGETS.items()
    }
    for source in sources:
        if source["approval_state"] != "approved":
            continue
        for category, values in source.get("approved_category_capacity", {}).items():
            totals[category]["expected"] += values["expected_tokens"]
            totals[category]["conservative"] += values["conservative_tokens"]
    for values in totals.values():
        values["deficit"] = max(0, values["target"] - values["conservative"])
        values["reserve_tokens"] = values["conservative"] - values["target"]
        values["locked"] = values["conservative"] >= values["target"]
    return totals


def build_registry(phase5b: dict[str, Any]) -> dict[str, Any]:
    if phase5b.get("schema_version") == "darkmind-v2-source-registry-v4-candidates-v3":
        phase5b["phase5b_predecessor"]["sha256"] = PHASE5B_REGISTRY_SHA256
        return phase5b
    registry = copy.deepcopy(phase5b)
    sources = registry["sources"]
    for source in sources:
        source["phase5b_approval_state"] = source["approval_state"]
        if source["id"] in RESOLUTIONS:
            status, kind, reason = RESOLUTIONS[source["id"]]
            source["approval_state"] = status
            source["phase5c_resolution"] = _resolution_record(source, status, kind, reason)
            if status == "rejected":
                source["source_cap_tokens"] = 0
                for key in ("optimistic_tokens", "expected_tokens", "conservative_tokens"):
                    source["capacity"][key] = 0
                source["resolution_steps"] = []
            else:
                source["resolution_steps"] = [reason]
        if source["id"] == "dgt_acquis_tr_en_20260722":
            source["approval_state"] = "rejected"
            source["source_cap_tokens"] = 0
            for key in ("optimistic_tokens", "expected_tokens", "conservative_tokens"):
                source["capacity"][key] = 0
            source["resolution_steps"] = []
            source["phase5c_correction"] = {
                "reason": "The official DGT-TM page lists 24 EU languages; Turkish is not covered.",
                "official_evidence_url": "https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-translation-memory_en",
                "review_date": ACCESS_DATE,
            }
    sources.extend(_new_candidates())
    approved = _approved_totals(sources)
    categories = _category_totals(sources)
    states = Counter(item["approval_state"] for item in sources)
    registry.update({
        "schema_version": "darkmind-v2-source-registry-v4-candidates-v3",
        "registry_id": "corpus-v4-phase5c-source-resolution-20260722",
        "phase5b_predecessor": {
            "registry_id": phase5b["registry_id"],
            "sha256": PHASE5B_REGISTRY_SHA256,
        },
        "phase5c_resolution_complete": True,
        "conditional_capacity_counted_as_approved": False,
        "acquisition_enabled": False,
        "source_lock_classification": "PARTIALLY LOCKED",
        "source_lock_decision": DECISION_LINE,
        "candidate_counts": {
            "total": len(sources),
            **{state: states.get(state, 0) for state in ("approved", "conditional", "deferred", "rejected")},
            "new": len(_new_candidates()),
            "phase5b_conditionals_resolved": len(RESOLUTIONS),
        },
        "capacity_model": {
            "formal_thresholds": {"expected_tokens": 250_000_000, "conservative_tokens": 200_000_000},
            "preferred_reserve_thresholds": {"expected_tokens": 275_000_000, "conservative_tokens": 220_000_000},
            "approved": approved,
            "formal_deficit": {
                "expected_tokens": 250_000_000 - approved["expected_tokens"],
                "conservative_tokens": 200_000_000 - approved["conservative_tokens"],
            },
            "preferred_reserve_deficit": {
                "expected_tokens": 275_000_000 - approved["expected_tokens"],
                "conservative_tokens": 220_000_000 - approved["conservative_tokens"],
            },
            "conditional_capacity_is_scenario_only": True,
        },
        "exclusive_category_allocation": categories,
        "storage_projection": {
            "approved_raw_min_bytes": 272_412_570,
            "approved_raw_max_bytes": 1_222_412_570,
            "fits_external_ssd_plan": True,
            "conditional_sources_excluded": True,
        },
    })
    return registry


def build_plan(phase5b_plan: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    if phase5b_plan.get("schema_version") == "darkmind-v2-corpus-v4-acquisition-plan-v2":
        phase5b_plan["predecessor"]["sha256"] = PHASE5B_PLAN_SHA256
        return phase5b_plan
    plan = copy.deepcopy(phase5b_plan)
    plan.update({
        "schema_version": "darkmind-v2-corpus-v4-acquisition-plan-v2",
        "plan_id": "darkmind-v2-corpus-v4-approved-sources-phase5c-20260722",
        "revised_date": ACCESS_DATE,
        "predecessor": {"plan_id": phase5b_plan["plan_id"], "sha256": PHASE5B_PLAN_SHA256},
        "acquisition_enabled": False,
        "conditional_sources_allowed": False,
        "allowed_source_states": ["approved"],
        "execution_controls": {
            "requires_separate_human_authorization": True,
            "verify_expected_size_before_commit": True,
            "verify_checksum_before_extraction": True,
            "redirects_must_remain_on_official_host": True,
            "extract_or_execute_downloaded_content": False,
            "maximum_total_bytes": 1_222_412_570,
        },
    })
    approved_ids = {item["id"] for item in registry["sources"] if item["approval_state"] == "approved"}
    plan["entries"] = [entry for entry in plan["entries"] if entry["source_id"] in approved_ids]
    for entry in plan["entries"]:
        entry["approval"]["evidence_review"] = "Phase 5C inherited approval; no execution authorization"
        entry["approval"]["execution_approved"] = False
        entry["allowed_redirect_hosts"] = [entry["official_url"].split("/")[2]]
        entry["content_execution_allowed"] = False
    return plan


def build_attribution(plan: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    sources = {item["id"]: item for item in registry["sources"]}
    entries = []
    for acquisition in plan["entries"]:
        source = sources[acquisition["source_id"]]
        entries.append({
            "source_id": source["id"],
            "official_source_name": source["official_source_name"],
            "snapshot_version": acquisition["snapshot_version"],
            "official_url": acquisition["official_url"],
            "license_identity": acquisition["license_identity"],
            "license_evidence_url": acquisition["license_evidence_url"],
            "attribution_record": acquisition["attribution_record"],
            "modification_notice": "Normalized, filtered, deduplicated, and tokenized for research corpus use.",
            "record_key_template": "{source_id}:{snapshot_version}:{document_path_or_url}",
            "approval_state": "approved",
            "acquisition_execution_authorized": False,
        })
    return {
        "schema_version": "darkmind-v2-corpus-v4-attribution-manifest-v1",
        "manifest_id": "darkmind-v2-corpus-v4-attribution-phase5c-20260722",
        "created_date": ACCESS_DATE,
        "planning_only": True,
        "downloads_performed": False,
        "scraping_performed": False,
        "execution_authorized": False,
        "acquisition_plan_id": plan["plan_id"],
        "coverage": {"approved_sources": len(entries), "manifest_entries": len(entries), "complete": True},
        "entries": entries,
    }


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_reports(registry: dict[str, Any], plan: dict[str, Any], attribution: dict[str, Any]) -> None:
    sources = registry["sources"]
    resolved = [item for item in sources if item.get("phase5c_resolution")]
    new = [item for item in sources if item.get("phase5c_discovery")]
    counts = registry["candidate_counts"]
    capacity = registry["capacity_model"]
    categories = registry["exclusive_category_allocation"]

    lines = [
        "# Phase 5C Conditional Source Resolution", "",
        "Planning-only technical due diligence; not legal advice. No data was downloaded.", "",
        f"Resolved Phase 5B conditional records: **{len(resolved)}/11**.", "",
    ]
    for source in resolved:
        review = source["phase5c_resolution"]
        lines.extend([
            f"## {source['id']}", "",
            f"Final status: **{review['final_status'].upper()}**",
            f"Resolution path: `{review['resolution_type']}`",
            f"Reason: {review['decision_reason']}", "",
            "| # | Required question | Recorded answer |", "|---:|---|---|",
        ])
        for index, key in enumerate(QUESTION_KEYS, 1):
            lines.append(f"| {index} | `{key}` | {_cell(review['answers'][key])} |")
        lines.extend(["", f"Next action: {review['next_action']}", ""])
        if "future_sample_plan" in review:
            sample = review["future_sample_plan"]
            lines.extend([
                "Future bounded sample (not authorized):",
                f"maximum {sample['maximum_download_bytes']:,} bytes; destination `{sample['destination']}`; "
                "SHA-256 required; training use is false.", "",
            ])
    (REPORTS / "phase5c_conditional_source_resolution.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    lines = [
        "# Phase 5C Replacement Source Discovery", "",
        "Official metadata was reviewed without acquiring any dataset or corpus file.", "",
        "| Source | Category | Exact artifact/snapshot | Status | Expected / conservative tokens | Decision |",
        "|---|---|---|---|---:|---|",
    ]
    for source in new:
        ev = source["new_candidate_evidence"]
        reason = source["resolution_steps"][0] if source["resolution_steps"] else "Rejected; no next action."
        lines.append(
            f"| `{source['id']}` | {_cell(source['category'])} | {_cell(source['snapshot_or_version'])} | "
            f"{source['approval_state']} | {source['capacity']['expected_tokens']:,} / "
            f"{source['capacity']['conservative_tokens']:,} | {_cell(reason)} |"
        )
    lines.extend([
        "", "Turkish general prose remains the largest unresolved gap. Common Voice Turkish was rejected because "
        "the official record combines no-rehosting terms with heavy Turkish-Wikipedia prompt provenance. The Leipzig "
        "route was rejected because no exact Turkish artifact was available and source-rights evidence was inadequate.",
        "", "DGT-TM was corrected to rejected: its official 24-language coverage does not include Turkish.",
    ])
    (REPORTS / "phase5c_replacement_source_discovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 5C Official Evidence Matrix", "",
        "| Source | State | Official artifact | Licence evidence | Snapshot | Evidence date |",
        "|---|---|---|---|---|---|",
    ]
    for source in sources:
        ev = source["official_evidence"]
        lines.append(
            f"| `{source['id']}` | {source['approval_state']} | {ev['official_dataset_url']} | "
            f"{source['license_evidence_url']} | {_cell(ev['exact_snapshot'])} | {ev['access_date']} |"
        )
    lines.extend(["", f"Official evidence coverage: **{len(sources)}/{len(sources)}** records."])
    (REPORTS / "phase5c_official_evidence_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conditional = [item for item in sources if item["approval_state"] == "conditional"]
    scenario_expected = sum(item["capacity"]["expected_tokens"] for item in conditional)
    scenario_conservative = sum(item["capacity"]["conservative_tokens"] for item in conditional)
    lines = [
        "# Phase 5C Capacity Model", "",
        "Conditional capacity is scenario-only and is never counted as approved.", "",
        "| Measure | Expected | Conservative |", "|---|---:|---:|",
        f"| Approved | {capacity['approved']['expected_tokens']:,} | {capacity['approved']['conservative_tokens']:,} |",
        f"| Formal threshold | 250,000,000 | 200,000,000 |",
        f"| Formal deficit | {capacity['formal_deficit']['expected_tokens']:,} | {capacity['formal_deficit']['conservative_tokens']:,} |",
        f"| Preferred reserve | 275,000,000 | 220,000,000 |",
        f"| Preferred-reserve deficit | {capacity['preferred_reserve_deficit']['expected_tokens']:,} | {capacity['preferred_reserve_deficit']['conservative_tokens']:,} |",
        f"| Conditional scenario (excluded) | {scenario_expected:,} | {scenario_conservative:,} |",
        "", "## Category allocation", "",
        "| Category | Target | Approved expected | Approved conservative | Deficit | Locked |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for category, values in categories.items():
        lines.append(
            f"| `{category}` | {values['target']:,} | {values['expected']:,} | "
            f"{values['conservative']:,} | {values['deficit']:,} | {values['locked']} |"
        )
    lines.extend(["", "Capacity basis for every discovered candidate is recorded in `new_candidate_evidence`; all estimates remain unapproved."])
    (REPORTS / "phase5c_capacity_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 5C Allocation Lock", "",
        "Each approved token estimate is assigned to one category only. Conditional sources have no locked allocation.", "",
        "| Category | Conservative | Target | Reserve | Result |", "|---|---:|---:|---:|---|",
    ]
    for category, values in categories.items():
        lines.append(
            f"| `{category}` | {values['conservative']:,} | {values['target']:,} | "
            f"{values['reserve_tokens']:,} | {'PASS' if values['locked'] else 'FAIL'} |"
        )
    lines.extend([
        "", "Concentration policy: family <=20%, exact dataset <=15%, single code ecosystem <=5%, "
        "bilingual family <=5%, Wikimedia <=20%, generated text =0%.",
        "", "Approved acquisition caps pass concentration checks, but every category remains below its conservative target.",
    ])
    (REPORTS / "phase5c_allocation_lock.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 5C Licence And Attribution", "",
        f"Manifest: `{attribution['manifest_id']}`. Coverage: **{attribution['coverage']['manifest_entries']}/"
        f"{attribution['coverage']['approved_sources']} approved sources**.", "",
        "| Source | Snapshot | Licence | Required attribution |", "|---|---|---|---|",
    ]
    for entry in attribution["entries"]:
        lines.append(
            f"| `{entry['source_id']}` | {_cell(entry['snapshot_version'])} | {_cell(entry['license_identity'])} | "
            f"{_cell(entry['attribution_record'])} |"
        )
    lines.extend([
        "", "Conditional, deferred, and rejected sources are absent from the executable mapping. "
        "Every manifest entry preserves source, version, licence, modification notice, and record-key rules.",
    ])
    (REPORTS / "phase5c_license_and_attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    lines = [
        "# Phase 5C Source Lock Decision", "", "## PARTIALLY LOCKED", "",
        DECISION_LINE, "",
        f"Candidates: {counts['total']} ({counts['approved']} approved, {counts['conditional']} conditional, "
        f"{counts['deferred']} deferred, {counts['rejected']} rejected; {counts['new']} new).",
        f"Approved capacity: {capacity['approved']['expected_tokens']:,} expected / "
        f"{capacity['approved']['conservative_tokens']:,} conservative tokens.",
        "The 250M/200M formal thresholds, 275M/220M preferred reserve, and all category targets are unmet.",
        "Official evidence materially improved, but unresolved legal, artifact, attribution, language, and capacity gates remain.",
        f"Acquisition plan: `{plan['plan_id']}` (`sha256:{plan_sha}`), disabled.",
        f"Attribution manifest: `{attribution['manifest_id']}`, complete for the three approved sources.",
        "Storage projection for approved artifacts: 272,412,570 to 1,222,412,570 bytes; feasible on the external-SSD plan.",
        "", "Next action: obtain exact official Turkish general-prose and controlled Turkish-English candidates, then resolve "
        "the official metadata gates for the strongest English/technical candidates before any acquisition authorization.",
        "", "No corpus download, scraping, construction, or training was performed.",
    ]
    (REPORTS / "phase5c_source_lock_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase5b_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    phase5b_plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    registry = build_registry(phase5b_registry)
    plan = build_plan(phase5b_plan, registry)
    attribution = build_attribution(plan, registry)
    _write_json(REGISTRY_PATH, registry)
    _write_json(PLAN_PATH, plan)
    _write_json(ATTRIBUTION_PATH, attribution)
    write_reports(registry, plan, attribution)
    print(json.dumps({
        "result": "PASS",
        "registry_id": registry["registry_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "attribution_manifest_id": attribution["manifest_id"],
        "classification": registry["source_lock_classification"],
        "acquisition_enabled": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
