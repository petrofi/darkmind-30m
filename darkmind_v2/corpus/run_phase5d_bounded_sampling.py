"""Run authorized Phase 5D bounded sampling without constructing a corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import statistics
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from darkmind_v2.corpus.deduplicate_text import exact_hash, jaccard_similarity, token_shingles
from darkmind_v2.corpus.detect_language import detect_language
from darkmind_v2.corpus.normalize_text import normalize_text
from darkmind_v2.corpus.validate_sample_inspection_authorization import (
    validate_sample_authorization,
    url_is_authorized,
)
from darkmind_v2.tokenizer.load_frozen_tokenizer import FrozenTokenizer


ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_PATH = ROOT / "darkmind_v2" / "config" / "corpus_v4_sample_inspection_authorization.json"
REGISTRY_PATH = ROOT / "darkmind_v2" / "corpus" / "source_registry.v4.candidates.json"
RUNTIME = Path(r"C:\DarkMindRuntime\phase5d")
CORPUS_V3_DOCUMENTS = ROOT / "darkmind_v2" / "data" / "phase3c1" / "final_text_retry1" / "documents.jsonl"
USER_AGENT = "DarkMind-v2-Phase5D/1.0 (petrofi/darkmind-30m; bounded research inspection)"
TEXT_EXTENSIONS = {".md", ".markdown", ".rst", ".txt", ".html", ".htm", ".xml", ".go", ".js", ".mjs", ".cjs"}
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{7,}\d)(?!\d)")
ADDRESS_RE = re.compile(r"\b\d{1,6}\s+[A-Za-z][A-Za-z .'-]{2,}\s(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln)\b", re.IGNORECASE)
SECRET_RE = re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.IGNORECASE)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
BENCHMARK_RE = re.compile(r"\b(?:HumanEval|MBPP|MMLU|GSM8K|benchmark answer|answer key|solution set)\b", re.IGNORECASE)
BOILERPLATE_RE = re.compile(r"\b(?:cookie|privacy policy|terms of use|skip to main content|all rights reserved)\b", re.IGNORECASE)
GENERATED_PATH_RE = re.compile(r"(?:^|/)(?:vendor|node_modules|third_party|generated|dist|build|testdata|fixtures?)(?:/|$)", re.IGNORECASE)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_rank(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


class BoundedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_domains: set[str]) -> None:
        super().__init__()
        self.allowed_domains = {item.lower() for item in allowed_domains}

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        hostname = (urllib.parse.urlparse(newurl).hostname or "").lower()
        if hostname not in self.allowed_domains:
            raise ValueError(f"redirect to unapproved domain rejected: {hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener(entry: dict[str, Any]) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(BoundedRedirectHandler(set(entry["permitted_redirect_domains"])))


def _content_type_allowed(filename: str, content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    suffixes = "".join(Path(filename).suffixes).lower()
    if suffixes.endswith((".tar.gz", ".tgz")):
        return media_type in {"application/gzip", "application/x-gzip", "application/octet-stream"}
    if suffixes.endswith((".tar.xz", ".xz")):
        return media_type in {"application/x-xz", "application/octet-stream", "application/x-tar"}
    if filename.endswith(".json"):
        return media_type in {"application/json", "text/json", "text/plain", "application/octet-stream"}
    if filename.endswith(".xml"):
        return media_type in {"application/xml", "text/xml", "text/plain", "application/octet-stream"}
    if filename.endswith(".html"):
        return media_type in {"text/html", "application/xhtml+xml", "text/plain"}
    return False


def download_url(
    entry: dict[str, Any], url: str, destination: Path, maximum_bytes: int,
    *, require_authorized_prefix: bool = True,
) -> dict[str, Any]:
    if require_authorized_prefix and not url_is_authorized(entry, url):
        raise ValueError(f"URL is not authorized for {entry['source_id']}: {url}")
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    if hostname not in {item.lower() for item in entry["permitted_redirect_domains"]}:
        raise ValueError(f"initial URL domain is not authorized: {hostname}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size = destination.stat().st_size
        if size > maximum_bytes:
            raise ValueError(f"existing sample exceeds byte limit: {destination.name}")
        return {
            "url": url, "final_url": url, "status": 200, "content_type": "reused/verified-local",
            "content_length_header": str(size), "etag": None, "last_modified": None,
            "bytes": size, "sha256": sha256_file(destination), "attempt": 0,
            "elapsed_seconds": 0.0, "reused_existing_runtime_evidence": True,
        }
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    last_error: Exception | None = None
    started = time.time()
    for attempt in range(1, 4):
        temporary = destination.with_suffix(destination.suffix + f".part.attempt{attempt}")
        try:
            digest = hashlib.sha256()
            byte_count = 0
            with _opener(entry).open(request, timeout=90) as response, temporary.open("wb") as output:
                final_url = response.geturl()
                final_host = (urllib.parse.urlparse(final_url).hostname or "").lower()
                if final_host not in {item.lower() for item in entry["permitted_redirect_domains"]}:
                    raise ValueError(f"final URL domain is not authorized: {final_host}")
                content_type = response.headers.get("Content-Type", "").strip()
                if not _content_type_allowed(destination.name, content_type):
                    raise ValueError(f"unexpected content type for {destination.name}: {content_type}")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > maximum_bytes:
                    raise ValueError(f"declared content length exceeds byte limit: {declared}")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > maximum_bytes:
                        raise ValueError(f"download exceeded byte limit: {destination.name}")
                    digest.update(chunk)
                    output.write(chunk)
                metadata = {
                    "url": url, "final_url": final_url, "status": getattr(response, "status", 200),
                    "content_type": content_type, "content_length_header": declared,
                    "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                    "bytes": byte_count, "sha256": digest.hexdigest(), "attempt": attempt,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            temporary.replace(destination)
            return metadata
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"bounded download failed after retries: {url}: {last_error}")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def verify_expected_checksum(entry: dict[str, Any], path: Path, actual_sha256: str) -> dict[str, Any]:
    expected = entry["expected_checksum"]
    if expected is None:
        return {"algorithm": "sha256", "actual": actual_sha256, "published": False, "result": "LOCAL_ONLY_PASS"}
    if expected["algorithm"] == "sha256":
        if actual_sha256 != expected["value"]:
            raise ValueError(f"published SHA-256 mismatch: {entry['source_id']}")
        return {"algorithm": "sha256", "expected": expected["value"], "actual": actual_sha256, "published": True, "result": "PASS"}
    return {
        "algorithm": "git_commit", "expected": expected["value"], "actual_archive_sha256": actual_sha256,
        "published_archive_checksum": False, "result": "IDENTITY_PASS_LOCAL_SHA256_PASS",
    }


def discover_govinfo(entry: dict[str, Any], source_dir: Path, seed: int) -> tuple[list[Path], dict[str, Any]]:
    base = entry["official_artifact_url"]
    inventory_path = source_dir / entry["expected_filename"]
    requests: list[dict[str, Any]] = [download_url(entry, base, inventory_path, 5_000_000)]
    parser = LinkParser()
    parser.feed(inventory_path.read_text(encoding="utf-8", errors="replace"))
    xml_urls = {
        urllib.parse.urljoin(base, link) for link in parser.links
        if urllib.parse.urljoin(base, link).startswith(base) and link.lower().endswith(".xml")
    }
    if not xml_urls:
        child_urls = sorted({
            urllib.parse.urljoin(base, link) for link in parser.links
            if re.fullmatch(r"(?:0?[1-9]|1[0-2])/?", link.strip())
        })
        for index, child_url in enumerate(child_urls, 1):
            child_path = source_dir / f"inventory-{index:02d}.html"
            requests.append(download_url(entry, child_url, child_path, 5_000_000, require_authorized_prefix=False))
            child_parser = LinkParser()
            child_parser.feed(child_path.read_text(encoding="utf-8", errors="replace"))
            xml_urls.update(
                urllib.parse.urljoin(child_url, link) for link in child_parser.links
                if urllib.parse.urljoin(child_url, link).startswith(base) and link.lower().endswith(".xml")
            )
    inventory = sorted(xml_urls)
    selected_urls = sorted(inventory, key=lambda item: (selection_rank(seed, item), item))[: entry["maximum_documents_files"]]
    selected_paths: list[Path] = []
    used = sum(item["bytes"] for item in requests)
    for url in selected_urls:
        remaining = entry["maximum_raw_bytes"] - used
        if remaining <= 0:
            break
        filename = Path(urllib.parse.urlparse(url).path).name
        path = source_dir / filename
        metadata = download_url(entry, url, path, remaining, require_authorized_prefix=False)
        requests.append(metadata)
        used += metadata["bytes"]
        selected_paths.append(path)
    selection = {
        "source_id": entry["source_id"], "seed": seed, "inventory_count": len(inventory),
        "complete_inventory": inventory, "selected_ids": selected_urls[:len(selected_paths)],
        "rejected_ids": [item for item in inventory if item not in set(selected_urls)],
        "selection_algorithm": "sha256(seed:URL), ascending",
    }
    selection["selection_manifest_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selected_paths, {"requests": requests, "selection": selection}


def acquire_entry(entry: dict[str, Any], seed: int) -> dict[str, Any]:
    source_dir = Path(entry["destination_path"])
    source_dir.mkdir(parents=True, exist_ok=True)
    if entry["extraction_method"] == "govinfo_directory_xml":
        paths, details = discover_govinfo(entry, source_dir, seed)
        checksums = [
            {"filename": path.name, "sha256": sha256_file(path), "published": False, "result": "LOCAL_ONLY_PASS"}
            for path in paths
        ]
        total_bytes = sum(item["bytes"] for item in details["requests"])
        result = {"source_id": entry["source_id"], "files": [str(path) for path in paths], "bytes": total_bytes, "checksums": checksums, **details}
    else:
        destination = source_dir / entry["expected_filename"]
        request = download_url(entry, entry["official_artifact_url"], destination, entry["maximum_raw_bytes"])
        checksum = verify_expected_checksum(entry, destination, request["sha256"])
        result = {
            "source_id": entry["source_id"], "files": [str(destination)], "bytes": request["bytes"],
            "checksums": [{"filename": destination.name, **checksum}], "requests": [request], "selection": None,
        }
    atomic_json(RUNTIME / "manifests" / f"{entry['source_id']}_raw_manifest.json", result)
    return result


def strip_markup(text: str) -> str:
    text = re.sub(r"(?is)<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"(?m)^\s*(?:import|require\().*$", " ", text)
    return html.unescape(text)


def _record(source_id: str, identifier: str, text: str, category_hint: str, path: str) -> dict[str, Any]:
    normalized, modifications = normalize_text(text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return {
        "id": f"{source_id}:{identifier}", "source_id": source_id, "source_path": path,
        "category_hint": category_hint, "text": normalized,
        "normalization_modifications": len(modifications),
    }


def extract_json(entry: dict[str, Any], path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if entry["extraction_method"] == "govuk_search_json":
        documents = payload.get("results", [])
        for index, item in enumerate(documents[:entry["maximum_documents_files"]]):
            identifier = str(item.get("content_id") or item.get("link") or index)
            text = "\n\n".join(str(item.get(key) or "") for key in ("title", "description") if item.get(key))
            records.append(_record(entry["source_id"], identifier, text, "english_general_educational", str(item.get("link") or path)))
    else:
        documents = payload.get("response", {}).get("docs", [])
        for index, item in enumerate(documents[:entry["maximum_documents_files"]]):
            identifier = str(item.get("id") or index)
            values: list[str] = []
            for key in ("title", "abstract"):
                value = item.get(key)
                if isinstance(value, list):
                    values.extend(str(part) for part in value)
                elif value:
                    values.append(str(value))
            records.append(_record(entry["source_id"], identifier, "\n\n".join(values), "technical_documentation", identifier))
    return records, {
        "raw_records": len(documents),
        "selected_records": len(records),
        "extracted_records": sum(bool(item["text"]) for item in records),
        "opened_files": 1,
        "parse_failures": 0,
    }


def extract_xml(entry: dict[str, Any], paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures = 0
    for path in paths:
        try:
            root = ET.parse(path).getroot()
            text = " ".join(part.strip() for part in root.itertext() if part.strip())
            records.append(_record(entry["source_id"], path.stem, text, "english_general_educational", str(path)))
            root.clear()
        except (ET.ParseError, UnicodeError, OSError):
            failures += 1
    return records, {
        "raw_records": len(paths),
        "selected_records": len(paths),
        "extracted_records": sum(bool(item["text"]) for item in records),
        "opened_files": len(paths) - failures,
        "parse_failures": failures,
    }


def eligible_tar_members(entry: dict[str, Any], archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    source_id = entry["source_id"]
    for member in archive.getmembers():
        name = member.name.replace("\\", "/")
        suffix = Path(name).suffix.lower()
        if not member.isfile() or suffix not in TEXT_EXTENSIONS or member.size > 2_000_000:
            continue
        if GENERATED_PATH_RE.search(name):
            continue
        lower = name.lower()
        if source_id == "kubernetes_website_f2987ba" and "/content/en/" not in lower:
            continue
        if source_id == "nodejs_24_18_0_source_docs" and not ("/doc/api/" in lower or "/lib/" in lower):
            continue
        if source_id == "go_1_26_5_source_docs" and not ("/doc/" in lower or "/src/" in lower):
            continue
        members.append(member)
    return sorted(members, key=lambda item: item.name)


def extract_tar(entry: dict[str, Any], path: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures = 0
    with tarfile.open(path, "r:*") as archive:
        inventory = eligible_tar_members(entry, archive)
        selected = sorted(inventory, key=lambda item: (selection_rank(seed, item.name), item.name))[:entry["maximum_documents_files"]]
        for member in selected:
            try:
                extracted = archive.extractfile(member)
                if extracted is None:
                    failures += 1
                    continue
                raw = extracted.read(2_000_001)
                if len(raw) > 2_000_000:
                    failures += 1
                    continue
                text = raw.decode("utf-8")
                suffix = Path(member.name).suffix.lower()
                if suffix in {".html", ".htm", ".xml"}:
                    text = strip_markup(text)
                category = "code_structured_text" if suffix in {".go", ".js", ".mjs", ".cjs"} else "technical_documentation"
                records.append(_record(entry["source_id"], member.name, text, category, member.name))
            except (UnicodeError, OSError, KeyError):
                failures += 1
    selected_names = [item.name for item in selected]
    selection = {
        "source_id": entry["source_id"], "seed": seed, "inventory_count": len(inventory),
        "complete_available_inventory": [item.name for item in inventory], "selected_ids": selected_names,
        "rejected_ids": [item.name for item in inventory if item.name not in set(selected_names)],
        "selection_algorithm": "sha256(seed:archive-member), ascending",
    }
    selection["selection_manifest_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metrics = {
        "raw_records": len(inventory),
        "selected_records": len(selected),
        "extracted_records": sum(bool(item["text"]) for item in records),
        "opened_files": len(records),
        "parse_failures": failures,
    }
    return records, metrics, selection


def simhash64(text: str) -> int:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if not words:
        return 0
    shingles = [" ".join(words[index:index + 5]) for index in range(max(1, len(words) - 4))]
    if len(shingles) > 128:
        step = max(1, len(shingles) // 128)
        shingles = shingles[::step][:128]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_corpus_v3_reference() -> dict[str, Any]:
    exact_hashes: set[str] = set()
    near_buckets: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    documents = 0
    near_documents = 0
    with CORPUS_V3_DOCUMENTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            digest = str(payload.get("normalized_content_sha256") or "")
            if digest:
                exact_hashes.add(digest)
            documents += 1
            if digest and int(digest[:2], 16) % 16 == 0:
                signature = simhash64(str(payload.get("text") or ""))
                for band in range(4):
                    near_buckets[(band, (signature >> (band * 16)) & 0xFFFF)].add(signature)
                near_documents += 1
    return {
        "exact_hashes": exact_hashes, "near_buckets": near_buckets,
        "documents": documents, "near_documents": near_documents, "near_sample_rate": "1/16 deterministic hash sample",
    }


def near_match(signature: int, buckets: dict[tuple[int, int], set[int]], maximum_distance: int = 3) -> bool:
    candidates: set[int] = set()
    for band in range(4):
        candidates.update(buckets.get((band, (signature >> (band * 16)) & 0xFFFF), set()))
    return any(hamming(signature, other) <= maximum_distance for other in candidates)


def line_repetition(text: str) -> float:
    lines = [line.strip().casefold() for line in text.splitlines() if len(line.strip()) >= 20]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def quality_score(text: str, alphabetic_ratio: float, symbol_ratio: float, repetition: float, boilerplate: int) -> float:
    length_score = min(1.0, len(text) / 1000)
    score = 0.35 * length_score + 0.35 * alphabetic_ratio + 0.15 * (1 - symbol_ratio) + 0.15 * (1 - repetition)
    score -= min(0.25, boilerplate * 0.03)
    return round(max(0.0, min(1.0, score)), 4)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def analyze_source(
    entry: dict[str, Any], records: list[dict[str, Any]], extraction: dict[str, Any],
    selection: dict[str, Any] | None, raw_bytes: int, tokenizer: FrozenTokenizer,
    v3: dict[str, Any], global_exact: dict[str, str], global_near: dict[tuple[int, int], set[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted_records: list[dict[str, Any]] = []
    internal_exact: set[str] = set()
    internal_near: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    counters: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    category_tokens: Counter[str] = Counter()
    quality_values: list[float] = []
    lengths: list[int] = []
    post_quality = post_pii = post_dedup = post_v3 = 0
    total_token_count = byte_fallback = unknown = 0
    for record in records:
        text = record["text"]
        counters["extracted_characters"] += len(text)
        words = WORD_RE.findall(text)
        counters["extracted_words"] += len(words)
        lengths.append(len(text))
        replacement = text.count("\ufffd")
        counters["malformed_unicode"] += replacement
        alphabetic = sum(char.isalpha() for char in text)
        symbol = sum(not char.isalnum() and not char.isspace() for char in text)
        denominator = max(1, len(text))
        alphabetic_ratio = alphabetic / denominator
        symbol_ratio = symbol / denominator
        repetition = line_repetition(text)
        boilerplate = len(BOILERPLATE_RE.findall(text))
        score = quality_score(text, alphabetic_ratio, symbol_ratio, repetition, boilerplate)
        quality_values.append(score)
        counters["boilerplate_hits"] += boilerplate
        counters["emails"] += len(EMAIL_RE.findall(text))
        counters["telephone_numbers"] += len(PHONE_RE.findall(text))
        counters["street_addresses"] += len(ADDRESS_RE.findall(text))
        counters["credential_patterns"] += len(SECRET_RE.findall(text))
        counters["private_keys"] += len(PRIVATE_KEY_RE.findall(text))
        counters["benchmark_hits"] += len(BENCHMARK_RE.findall(text))
        tokens = tokenizer.encode(text, add_eos=True) if text else []
        token_count = len(tokens)
        total_token_count += token_count
        byte_fallback += sum(tokenizer.is_byte_fallback_id(token_id) for token_id in tokens)
        unknown += sum(tokenizer.is_unknown_id(token_id) for token_id in tokens)
        category = record["category_hint"]
        if category == "code_structured_text":
            language = "code"
        elif entry["source_id"] in {
            "govuk_content_ogl3_20260722", "govinfo_federal_register_2025_xml",
            "plos_ccby_jats_allowlist", "go_1_26_5_source_docs",
            "kubernetes_website_f2987ba", "nodejs_24_18_0_source_docs",
        }:
            language = "en"
        else:
            language = detect_language(text)
        language_tokens[language] += token_count
        category_tokens[category] += token_count
        record_metrics = {
            "quality_score": score, "alphabetic_ratio": round(alphabetic_ratio, 5),
            "symbol_ratio": round(symbol_ratio, 5), "repetition": round(repetition, 5),
            "language": language, "category": category, "tokens": token_count,
        }
        if len(text) < 120 or alphabetic_ratio < 0.35 or replacement or score < 0.40:
            counters["quality_rejected"] += 1
            continue
        post_quality += token_count
        if SECRET_RE.search(text) or PRIVATE_KEY_RE.search(text):
            counters["pii_secret_rejected"] += 1
            continue
        post_pii += token_count
        digest = exact_hash(text)
        signature = simhash64(text)
        if digest in internal_exact or digest in global_exact:
            counters["exact_duplicates"] += 1
            continue
        internal_candidates: set[int] = set()
        for band in range(4):
            internal_candidates.update(internal_near.get((band, (signature >> (band * 16)) & 0xFFFF), set()))
            internal_candidates.update(global_near.get((band, (signature >> (band * 16)) & 0xFFFF), set()))
        if any(hamming(signature, other) <= 3 for other in internal_candidates):
            counters["near_duplicates"] += 1
            continue
        post_dedup += token_count
        if digest in v3["exact_hashes"]:
            counters["corpus_v3_exact_overlap"] += 1
            continue
        if near_match(signature, v3["near_buckets"]):
            counters["corpus_v3_near_overlap"] += 1
            continue
        post_v3 += token_count
        internal_exact.add(digest)
        global_exact[digest] = record["id"]
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            internal_near[key].add(signature)
            global_near[key].add(signature)
        accepted_records.append({
            "id": record["id"], "source_id": entry["source_id"], "normalized_sha256": digest,
            "simhash64": f"{signature:016x}", "language": language, "category": category,
            "token_count": token_count, "metrics": record_metrics, "text": text,
        })
    total_language = sum(language_tokens.values()) or 1
    total_category = sum(category_tokens.values()) or 1
    inventory_count = selection["inventory_count"] if selection else max(1, extraction["raw_records"])
    sample_coverage = len(records) / max(1, inventory_count)
    observed_usable = post_v3
    exact_collection_estimate = int(observed_usable / max(sample_coverage, 1e-9))
    source_cap = next(item["source_cap_tokens"] for item in json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["sources"] if item["id"] == entry["source_id"])
    optimistic = min(source_cap, exact_collection_estimate)
    expected = min(optimistic, int(optimistic * 0.75))
    conservative = min(expected, int(expected * (0.60 if sample_coverage < 0.20 else 0.75)))
    if entry["extraction_method"] in {"govuk_search_json", "plos_search_json"}:
        optimistic = observed_usable
        expected = int(observed_usable * 0.90)
        conservative = int(observed_usable * 0.75)
    metrics = {
        "schema_version": "darkmind-v2-phase5d-source-inspection-v1",
        "source_id": entry["source_id"], "raw_bytes": raw_bytes,
        "extraction": {
            **extraction,
            "extraction_success_rate": round(extraction["extracted_records"] / max(1, extraction["selected_records"]), 6),
            "empty_extraction_rate": round(sum(not item["text"] for item in records) / max(1, len(records)), 6),
            "parse_failure_rate": round(extraction["parse_failures"] / max(1, extraction["raw_records"]), 6),
            "ocr_dependent_documents": 0, "ocr_used": False,
        },
        "text": {
            "extracted_characters": counters["extracted_characters"], "extracted_words": counters["extracted_words"],
            "document_length_min": min(lengths) if lengths else 0, "document_length_median": statistics.median(lengths) if lengths else 0,
            "document_length_p95": percentile([float(item) for item in lengths], 0.95), "malformed_unicode": counters["malformed_unicode"],
            "boilerplate_hits": counters["boilerplate_hits"],
        },
        "quality": {
            "score_min": min(quality_values) if quality_values else 0.0, "score_median": statistics.median(quality_values) if quality_values else 0.0,
            "score_p95": percentile(quality_values, 0.95), "quality_rejected_documents": counters["quality_rejected"],
        },
        "language": {key: {"tokens": value, "share": round(value / total_language, 6)} for key, value in sorted(language_tokens.items())},
        "category": {key: {"tokens": value, "share": round(value / total_category, 6)} for key, value in sorted(category_tokens.items())},
        "safety": {
            "emails": counters["emails"], "telephone_numbers": counters["telephone_numbers"],
            "street_addresses": counters["street_addresses"], "credential_patterns": counters["credential_patterns"],
            "private_keys": counters["private_keys"], "benchmark_hits": counters["benchmark_hits"],
            "pii_secret_rejected_documents": counters["pii_secret_rejected"],
        },
        "deduplication": {
            "internal_and_cross_source_exact_duplicates": counters["exact_duplicates"],
            "internal_and_cross_source_near_duplicates": counters["near_duplicates"],
            "corpus_v3_exact_overlap_documents": counters["corpus_v3_exact_overlap"],
            "corpus_v3_near_overlap_documents": counters["corpus_v3_near_overlap"],
            "corpus_v3_reference_documents": v3["documents"], "corpus_v3_near_reference_documents": v3["near_documents"],
            "corpus_v3_near_reference_sample_rate": v3["near_sample_rate"], "simhash_max_hamming_distance": 3,
        },
        "tokenizer": {
            "name": tokenizer.manifest["tokenizer_name"], "raw_tokens": total_token_count,
            "tokens_per_raw_mb": round(total_token_count / max(raw_bytes / 1_000_000, 1e-9), 3),
            "tokens_per_accepted_document": round(post_v3 / max(1, len(accepted_records)), 3),
            "byte_fallback_tokens": byte_fallback, "byte_fallback_rate": round(byte_fallback / max(1, total_token_count), 8),
            "unknown_tokens": unknown, "unknown_token_rate": round(unknown / max(1, total_token_count), 8),
            "eos_boundaries": len(records), "post_quality_filter_tokens": post_quality,
            "post_pii_filter_tokens": post_pii, "post_dedup_tokens": post_dedup,
            "post_corpus_v3_overlap_tokens": post_v3,
        },
        "capacity": {
            "inventory_count": inventory_count, "sample_documents": len(records), "sample_coverage": round(sample_coverage, 8),
            "optimistic_tokens": optimistic, "expected_tokens": expected, "conservative_tokens": conservative,
            "uncertainty_band": [conservative, optimistic], "basis": "bounded exact collection only; no portal-wide extrapolation",
            "extraction_loss_rate": round(1 - extraction["extracted_records"] / max(1, extraction["selected_records"]), 6),
            "quality_filter_loss_rate": round(1 - post_quality / max(1, total_token_count), 6),
            "pii_secret_loss_rate": round(1 - post_pii / max(1, post_quality), 6),
            "internal_dedup_loss_rate": round(1 - post_dedup / max(1, post_pii), 6),
            "corpus_v3_overlap_loss_rate": round(1 - post_v3 / max(1, post_dedup), 6),
        },
        "accepted_documents": len(accepted_records), "no_training": True,
    }
    return metrics, accepted_records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def extract_acquisition(entry: dict[str, Any], acquisition: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    paths = [Path(item) for item in acquisition["files"]]
    method = entry["extraction_method"]
    if method in {"govuk_search_json", "plos_search_json"}:
        records, metrics = extract_json(entry, paths[0])
        selection = {
            "source_id": entry["source_id"], "seed": seed, "inventory_count": metrics["raw_records"],
            "selected_ids": [item["id"] for item in records], "rejected_ids": [],
            "selection_algorithm": "complete bounded API response in stable source order",
        }
        selection["selection_manifest_sha256"] = hashlib.sha256(
            json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return records, metrics, selection
    if method == "govinfo_directory_xml":
        records, metrics = extract_xml(entry, paths)
        return records, metrics, acquisition["selection"]
    records, metrics, selection = extract_tar(entry, paths[0], seed)
    return records, metrics, selection


def run() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_sample_authorization(authorization, registry)
    disk_before = shutil.disk_usage(RUNTIME.anchor).free
    projected = authorization["projected_extracted_and_intermediate_bytes"]
    if disk_before - projected < authorization["minimum_free_bytes_after_completion"]:
        raise ValueError("Phase 5D free-space reserve would fall below 25 GB")
    for directory in ("manifests", "licenses", "samples", "extracted", "normalized", "rejected", "reports", "checksums", "temporary"):
        (RUNTIME / directory).mkdir(parents=True, exist_ok=True)
    atomic_json(RUNTIME / "manifests" / "sample_inspection_authorization.json", authorization)
    seed = authorization["selection_seed"]
    acquisitions: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for entry in authorization["entries"]:
        acquisition = acquire_entry(entry, seed)
        total_bytes += acquisition["bytes"]
        if total_bytes > authorization["maximum_total_downloaded_bytes"]:
            raise ValueError("Phase 5D total download byte limit exceeded")
        acquisitions[entry["source_id"]] = acquisition
        print(json.dumps({"event": "download_complete", "source_id": entry["source_id"], "bytes": acquisition["bytes"]}), flush=True)
    v3 = build_corpus_v3_reference()
    print(json.dumps({"event": "corpus_v3_reference_ready", "documents": v3["documents"], "near_documents": v3["near_documents"]}), flush=True)
    tokenizer = FrozenTokenizer()
    global_exact: dict[str, str] = {}
    global_near: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    source_results: list[dict[str, Any]] = []
    for entry in authorization["entries"]:
        acquisition = acquisitions[entry["source_id"]]
        records, extraction, selection = extract_acquisition(entry, acquisition, seed)
        write_jsonl(RUNTIME / "extracted" / f"{entry['source_id']}.jsonl", records)
        if selection:
            atomic_json(RUNTIME / "manifests" / f"{entry['source_id']}_selection_manifest.json", selection)
        metrics, accepted = analyze_source(
            entry, records, extraction, selection, acquisition["bytes"], tokenizer,
            v3, global_exact, global_near,
        )
        write_jsonl(RUNTIME / "normalized" / f"{entry['source_id']}.jsonl", accepted)
        atomic_json(RUNTIME / "reports" / f"{entry['source_id']}_inspection.json", metrics)
        source_results.append(metrics)
        print(json.dumps({
            "event": "inspection_complete", "source_id": entry["source_id"],
            "accepted_documents": metrics["accepted_documents"],
            "post_overlap_tokens": metrics["tokenizer"]["post_corpus_v3_overlap_tokens"],
        }), flush=True)
    disk_after = shutil.disk_usage(RUNTIME.anchor).free
    summary = {
        "schema_version": "darkmind-v2-phase5d-bounded-sample-run-v1",
        "result": "PASS", "authorization_id": authorization["authorization_id"],
        "selection_seed": seed, "authorized_sources": len(authorization["entries"]),
        "downloaded_sources": len(acquisitions), "total_downloaded_bytes": total_bytes,
        "maximum_total_downloaded_bytes": authorization["maximum_total_downloaded_bytes"],
        "disk_free_before_bytes": disk_before, "disk_free_after_bytes": disk_after,
        "minimum_required_free_reserve_bytes": authorization["minimum_free_bytes_after_completion"],
        "corpus_v3_reference_documents": v3["documents"],
        "corpus_v3_near_reference_documents": v3["near_documents"],
        "sources": source_results, "full_production_acquisition": False,
        "corpus_construction": False, "training_use": False,
        "production_tokenization": False, "downloaded_content_executed": False,
    }
    atomic_json(RUNTIME / "reports" / "sample_run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = run()
    print(json.dumps({
        "result": summary["result"], "downloaded_sources": summary["downloaded_sources"],
        "total_downloaded_bytes": summary["total_downloaded_bytes"],
        "disk_free_after_bytes": summary["disk_free_after_bytes"],
        "training_use": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
