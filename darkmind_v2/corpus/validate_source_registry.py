"""Validate the DarkMind v2 approved corpus source registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ALLOWED_LANGUAGES = {"tr", "en", "mixed_tr_en"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high", "rejected"}
ALLOWED_RETRIEVAL_METHODS = {
    "direct_download",
    "chunked_download",
    "documented_manual_sample",
    "official_api",
    "local_fixture_only",
}
REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "source_name",
    "official_homepage",
    "official_domains",
    "language",
    "content_type",
    "license_id",
    "official_license_url",
    "attribution_requirements",
    "redistribution_requirements",
    "commercial_use_status",
    "modification_status",
    "jurisdiction_warning",
    "source_version",
    "snapshot_date",
    "estimated_download_size",
    "estimated_download_size_bytes",
    "intended_sample_size_characters",
    "checksum_available",
    "retrieval_method",
    "max_download_bytes",
    "max_sample_characters",
    "approved",
    "approval_reason",
    "risk_level",
    "notes",
}
UNKNOWN_LICENSE_MARKERS = {
    "",
    "unknown",
    "ambiguous",
    "various",
    "mixed",
    "unverified",
    "custom",
    "not reviewed",
    "common crawl",
}
UNKNOWN_STATUS_MARKERS = {"", "unknown", "ambiguous", "not_reviewed", "not reviewed", "varies"}
COMMON_CRAWL_DERIVED_MARKERS = {
    "common crawl",
    "fineweb",
    "oscar",
    "culturax",
    "mc4",
    "cc100",
    "refinedweb",
}
REJECTED_SOURCE_MARKERS = {
    "leaked",
    "private message",
    "private_messages",
    "personal data",
    "social media",
    "reddit",
    "twitter",
    "x.com",
    "facebook",
    "instagram",
    "telegram",
    "whatsapp",
}
URL_RE = re.compile(r"https?://[^\s)>,]+")


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _host(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").lower()


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _host_matches(host: str, allowed_domains: set[str]) -> bool:
    normalized = host.lower()
    for domain in allowed_domains:
        domain = domain.lower()
        if normalized == domain or normalized.endswith("." + domain):
            return True
    return False


def _contains_any(text: str, markers: set[str]) -> bool:
    lowered = text.casefold()
    return any(marker and marker in lowered for marker in markers)


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    sources = payload.get("sources")
    hard_download_cap = payload.get("hard_download_cap_bytes")

    if not isinstance(sources, list) or not sources:
        failures.append("registry must contain at least one source")
        sources = []
    if not isinstance(hard_download_cap, int) or hard_download_cap <= 0:
        failures.append("hard_download_cap_bytes must be a positive integer")
        hard_download_cap = 0

    seen_ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    approved_count = 0

    for index, source in enumerate(sources, start=1):
        prefix = f"source {index}"
        if not isinstance(source, dict):
            failures.append(f"{prefix}: source record must be an object")
            continue

        source_id = _as_text(source.get("source_id")) or f"<missing-{index}>"
        prefix = f"{source_id}"
        missing = sorted(REQUIRED_SOURCE_FIELDS - set(source))
        if missing:
            failures.append(f"{prefix}: missing required fields {missing}")

        if source_id in seen_ids:
            failures.append(f"{prefix}: duplicate source_id")
        seen_ids.add(source_id)

        language = source.get("language")
        risk_level = source.get("risk_level")
        language_counts[str(language)] += 1
        risk_counts[str(risk_level)] += 1

        if language not in ALLOWED_LANGUAGES:
            failures.append(f"{prefix}: language must be one of {sorted(ALLOWED_LANGUAGES)}")
        if risk_level not in ALLOWED_RISK_LEVELS:
            failures.append(f"{prefix}: risk_level must be one of {sorted(ALLOWED_RISK_LEVELS)}")
        if risk_level == "rejected":
            failures.append(f"{prefix}: rejected sources cannot be approved for retrieval")
        if risk_level == "high":
            warnings.append(f"{prefix}: high-risk approved source needs explicit human review before Phase 1B")

        if source.get("approved") is not True:
            failures.append(f"{prefix}: approved must be explicitly true")
        else:
            approved_count += 1

        official_domains = source.get("official_domains")
        if not isinstance(official_domains, list) or not all(isinstance(item, str) and item.strip() for item in official_domains):
            failures.append(f"{prefix}: official_domains must be a non-empty list of host names")
            allowed_domains: set[str] = set()
        else:
            allowed_domains = {item.strip().lower() for item in official_domains}

        homepage = _as_text(source.get("official_homepage"))
        if not _is_http_url(homepage):
            failures.append(f"{prefix}: official_homepage must be an HTTP(S) URL")
        elif allowed_domains and not _host_matches(_host(homepage), allowed_domains):
            failures.append(f"{prefix}: official_homepage host is not in official_domains")

        download_url = _as_text(source.get("official_download_url"))
        retrieval_notes = _as_text(source.get("documented_retrieval_method"))
        if not download_url and not retrieval_notes:
            failures.append(f"{prefix}: provide official_download_url or documented_retrieval_method")
        if download_url:
            if not _is_http_url(download_url):
                failures.append(f"{prefix}: official_download_url must be an HTTP(S) URL")
            elif allowed_domains and not _host_matches(_host(download_url), allowed_domains):
                failures.append(f"{prefix}: official_download_url host is not an approved official domain")

        method_urls = URL_RE.findall(retrieval_notes)
        for method_url in method_urls:
            host = _host(method_url)
            if allowed_domains and not _host_matches(host, allowed_domains):
                failures.append(f"{prefix}: documented retrieval URL {method_url} is not an approved official domain")

        license_id = _as_text(source.get("license_id"))
        if not license_id or _contains_any(license_id, UNKNOWN_LICENSE_MARKERS):
            failures.append(f"{prefix}: license_id is unknown or ambiguous")
        official_license_url = _as_text(source.get("official_license_url"))
        if not _is_http_url(official_license_url):
            failures.append(f"{prefix}: official_license_url must provide official license evidence")

        if not _as_text(source.get("attribution_requirements")):
            failures.append(f"{prefix}: attribution_requirements cannot be empty")
        if not _as_text(source.get("redistribution_requirements")):
            failures.append(f"{prefix}: redistribution_requirements cannot be empty")

        for field in ("commercial_use_status", "modification_status"):
            status = _as_text(source.get(field))
            if not status or _contains_any(status, UNKNOWN_STATUS_MARKERS):
                failures.append(f"{prefix}: {field} is unknown or ambiguous")

        if not _as_text(source.get("source_version")) and not _as_text(source.get("snapshot_date")):
            failures.append(f"{prefix}: source_version or snapshot_date is required")

        retrieval_method = source.get("retrieval_method")
        if retrieval_method not in ALLOWED_RETRIEVAL_METHODS:
            failures.append(f"{prefix}: retrieval_method must be one of {sorted(ALLOWED_RETRIEVAL_METHODS)}")

        for field in (
            "estimated_download_size_bytes",
            "intended_sample_size_characters",
            "max_download_bytes",
            "max_sample_characters",
        ):
            value = source.get(field)
            if not isinstance(value, int) or value <= 0:
                failures.append(f"{prefix}: {field} must be a positive integer")

        max_download_bytes = source.get("max_download_bytes")
        estimated_download_size_bytes = source.get("estimated_download_size_bytes")
        intended_sample_size_characters = source.get("intended_sample_size_characters")
        max_sample_characters = source.get("max_sample_characters")
        if isinstance(max_download_bytes, int) and max_download_bytes > hard_download_cap:
            failures.append(f"{prefix}: max_download_bytes exceeds registry hard_download_cap_bytes")
        if isinstance(estimated_download_size_bytes, int) and isinstance(max_download_bytes, int):
            if estimated_download_size_bytes > max_download_bytes:
                failures.append(f"{prefix}: estimated_download_size_bytes exceeds max_download_bytes")
        if isinstance(intended_sample_size_characters, int) and isinstance(max_sample_characters, int):
            if intended_sample_size_characters > max_sample_characters:
                failures.append(f"{prefix}: intended_sample_size_characters exceeds max_sample_characters")

        searchable = " ".join(
            _as_text(source.get(field))
            for field in (
                "source_id",
                "source_name",
                "official_homepage",
                "official_download_url",
                "documented_retrieval_method",
                "license_id",
                "notes",
            )
        )
        if _contains_any(searchable, COMMON_CRAWL_DERIVED_MARKERS):
            failures.append(f"{prefix}: Common Crawl-derived datasets require separate review and cannot be auto-approved")
        if _contains_any(searchable, REJECTED_SOURCE_MARKERS):
            failures.append(f"{prefix}: social, private, leaked, or personal datasets are rejected")

    report = {
        "source_count": len(sources),
        "approved_count": approved_count,
        "language_distribution": dict(sorted(language_counts.items())),
        "risk_distribution": dict(sorted(risk_counts.items())),
        "warnings": warnings,
    }
    return report, failures


def validate_registry(path: Path) -> tuple[dict[str, Any], list[str]]:
    return validate_registry_payload(load_registry(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DarkMind v2 corpus source registry.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name("source_registry.example.json"),
    )
    args = parser.parse_args()

    report, failures = validate_registry(args.registry)
    payload = {"result": "FAIL" if failures else "PASS", "report": report, "failures": failures}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
