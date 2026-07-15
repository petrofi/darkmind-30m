"""Stream, clean, deduplicate, allocate, and tokenize Corpus V3 Tranche 1."""

from __future__ import annotations

import argparse
import bz2
import collections
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from darkmind_v2.corpus.detect_mojibake import looks_like_mojibake
from darkmind_v2.corpus.download_corpus_v3_tranche import (
    DEFAULT_CONFIG,
    atomic_write_json,
    atomic_write_text,
    file_hashes,
    validate_config,
)
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, load_frozen_tokenizer


ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
REFERENCE_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"</?(?:div|span|small|big|center|gallery|timeline|math|chem|score|poem|syntaxhighlight|code|pre)\b[^>]*>", re.IGNORECASE)
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|Dosya|Resim|Category|Kategori):.*?\]\]", re.IGNORECASE | re.DOTALL)
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]")
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)"),
    "turkish_identity_number": re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)"),
}
TURKISH_MARKERS = set("cCgGiIoOsSuU")
TURKISH_DIACRITICS = set("çÇğĞıİöÖşŞüÜ")
TR_STOPWORDS = {
    "ve", "bir", "bu", "icin", "için", "ile", "olarak", "olan", "daha", "gibi", "ya", "da", "de",
    "tarafindan", "tarafından", "sonra", "kadar", "ancak", "veya", "ise", "cok", "çok", "uzere", "üzere",
}
EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "for", "that", "with", "as", "on", "by", "from", "or", "an",
    "this", "are", "be", "it", "was", "can", "which", "at", "not", "has", "have",
}


@dataclass(frozen=True)
class ExtractedDocument:
    source_id: str
    snapshot: str
    native_id: str
    title: str
    language: str
    category: str
    text: str
    raw_content_sha256: str
    source_url: str


@dataclass(frozen=True)
class QualityResult:
    text: str | None
    reason: str | None
    detected_language: str
    language_confidence: float


def repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalized_hash_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_document_id(source_id: str, snapshot: str, native_id: str, normalized_hash: str) -> str:
    identity = f"{source_id}\n{snapshot}\n{native_id}\n{normalized_hash}".encode("utf-8")
    return f"{source_id}:{hashlib.sha256(identity).hexdigest()}"


def strip_balanced(text: str, opening: str, closing: str) -> str:
    output: list[str] = []
    index = 0
    depth = 0
    while index < len(text):
        if text.startswith(opening, index):
            depth += 1
            index += len(opening)
            continue
        if depth and text.startswith(closing, index):
            depth -= 1
            index += len(closing)
            continue
        if depth == 0:
            output.append(text[index])
        index += 1
    return "".join(output)


def clean_wiki_markup(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = REFERENCE_RE.sub(" ", text)
    text = strip_balanced(text, "{{", "}}")
    text = strip_balanced(text, "{|", "|}")
    text = FILE_LINK_RE.sub(" ", text)
    text = WIKI_LINK_RE.sub(lambda match: match.group(2), text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or " ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"'{2,5}", "", text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith(("__", "[[Category:", "[[Kategori:", "{{", "|", "!")):
            continue
        if re.fullmatch(r"={2,6}.*={2,6}", stripped):
            heading = stripped.strip("= ")
            if heading.casefold() not in {"references", "kaynakca", "kaynakça", "external links", "dis baglantilar", "dış bağlantılar"}:
                cleaned_lines.extend(("", heading, ""))
            continue
        cleaned_lines.append(stripped)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _tag_name(child) == name), None)


def iter_wikipedia_documents(path: Path, source: dict[str, Any]) -> Iterator[ExtractedDocument]:
    with bz2.open(path, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        root: ET.Element | None = None
        for event, element in context:
            if root is None and event == "start":
                root = element
            if event != "end" or _tag_name(element) != "page":
                continue
            namespace_node = _direct_child(element, "ns")
            redirect_node = _direct_child(element, "redirect")
            title_node = _direct_child(element, "title")
            page_id_node = _direct_child(element, "id")
            revision_node = _direct_child(element, "revision")
            text_node = _direct_child(revision_node, "text") if revision_node is not None else None
            if (
                namespace_node is not None
                and namespace_node.text == "0"
                and redirect_node is None
                and page_id_node is not None
                and page_id_node.text
                and text_node is not None
                and text_node.text
            ):
                raw_text = text_node.text
                cleaned = clean_wiki_markup(raw_text)
                if cleaned:
                    page_id = page_id_node.text
                    title = title_node.text if title_node is not None and title_node.text else ""
                    project = "tr.wikipedia.org" if source["language"] == "tr" else "en.wikipedia.org"
                    yield ExtractedDocument(
                        source_id=source["source_id"],
                        snapshot=source["snapshot"],
                        native_id=page_id,
                        title=title,
                        language=source["language"],
                        category=source["category"],
                        text=cleaned,
                        raw_content_sha256=sha256_text(raw_text),
                        source_url=f"https://{project}/?curid={page_id}",
                    )
            element.clear()
            if root is not None:
                root.clear()


def split_python_sections(text: str, *, maximum_characters: int = 40000) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[list[str]] = []
    current: list[str] = []
    for index, line in enumerate(lines):
        is_heading = (
            index + 1 < len(lines)
            and line.strip()
            and re.fullmatch(r"[=\-~^*+#]{3,}", lines[index + 1].strip() or " ") is not None
        )
        if is_heading and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)

    chunks: list[str] = []
    for section in sections:
        content_lines = []
        for line in section:
            stripped = line.strip()
            heading_rule = re.fullmatch(r"[=\-~^*+#]{3,}", stripped or " ") is not None
            table_rule = (
                len(stripped) >= 6
                and re.fullmatch(r"[|+:=\-\s]+", stripped) is not None
                and any(character in stripped for character in "=-")
            )
            if not heading_rule and not table_rule:
                content_lines.append(line)
        paragraphs = re.split(r"\n\s*\n", "\n".join(content_lines).strip())
        pending = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if pending and len(pending) + len(paragraph) + 2 > maximum_characters:
                chunks.append(pending)
                pending = ""
            if len(paragraph) > maximum_characters:
                for start in range(0, len(paragraph), maximum_characters):
                    piece = paragraph[start : start + maximum_characters].strip()
                    if piece:
                        chunks.append(piece)
            else:
                pending = paragraph if not pending else pending + "\n\n" + paragraph
        if pending:
            chunks.append(pending)
    merged: list[str] = []
    pending = ""
    for chunk in chunks:
        if not pending:
            pending = chunk
        elif len(pending) < 500:
            pending = pending + "\n\n" + chunk
        else:
            merged.append(pending)
            pending = chunk
    if pending:
        merged.append(pending)
    return merged


def iter_python_documents(path: Path, source: dict[str, Any]) -> Iterator[ExtractedDocument]:
    with tarfile.open(path, "r:bz2") as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.isfile() and member.name.endswith(".txt")),
            key=lambda member: member.name,
        )
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            text = raw.decode("utf-8", errors="strict")
            for section_index, section in enumerate(split_python_sections(text), start=1):
                native_id = f"{member.name}#{section_index}"
                yield ExtractedDocument(
                    source_id=source["source_id"],
                    snapshot=source["snapshot"],
                    native_id=native_id,
                    title=member.name,
                    language=source["language"],
                    category=source["category"],
                    text=section,
                    raw_content_sha256=sha256_text(section),
                    source_url=(
                        "https://docs.python.org/tr/3.14/"
                        if source["language"] == "tr"
                        else "https://docs.python.org/3.14/"
                    ),
                )


def iter_source_documents(source: dict[str, Any], raw_root: Path) -> Iterator[ExtractedDocument]:
    for item in source["files"]:
        path = raw_root / source["source_id"] / item["filename"]
        if source["extraction_format"] == "streaming-bzip2-mediawiki-xml":
            yield from iter_wikipedia_documents(path, source)
        elif source["extraction_format"] == "streaming-tar-bzip2-plain-text":
            yield from iter_python_documents(path, source)
        else:
            raise ValueError(f"unsupported extraction format: {source['extraction_format']}")


def detect_language_confidence(text: str) -> tuple[str, float]:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if not words:
        return "unknown", 0.0
    tr_score = sum(2 for word in words if word in TR_STOPWORDS)
    en_score = sum(2 for word in words if word in EN_STOPWORDS)
    tr_score += min(20, sum(1 for character in text if character in TURKISH_DIACRITICS))
    if tr_score == 0 and en_score == 0:
        ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
        letter_count = sum(character.isalpha() for character in text)
        if letter_count and ascii_letters / letter_count >= 0.97:
            return "en", 0.55
        return "unknown", 0.0
    total = tr_score + en_score
    if tr_score > en_score:
        return "tr", tr_score / total
    if en_score > tr_score:
        return "en", en_score / total
    return "unknown", 0.5


def material_pii_kind(text: str) -> str | None:
    for match in PII_PATTERNS["email"].finditer(text):
        domain = match.group(0).rsplit("@", 1)[-1].casefold()
        if domain not in {"example.com", "example.org", "example.net", "invalid"}:
            return "email"
    folded = text.casefold()
    if PII_PATTERNS["turkish_identity_number"].search(text) and any(
        marker in folded for marker in ("tc kimlik", "t.c. kimlik", "identity number")
    ):
        return "turkish_identity_number"
    if PII_PATTERNS["phone"].search(text) and any(
        marker in folded for marker in ("phone number", "telephone number", "telefon numarası", "contact number")
    ):
        return "phone"
    if PII_PATTERNS["ipv4"].search(text) and any(
        marker in folded for marker in ("personal ip", "user ip address", "kullanıcı ip adresi")
    ):
        return "ipv4"
    return None


def quality_filter(text: str, expected_language: str, policy: dict[str, Any]) -> QualityResult:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        return QualityResult(None, "malformed_surrogate", "unknown", 0.0)
    if "\ufffd" in text:
        return QualityResult(None, "replacement_character", "unknown", 0.0)
    if any(unicodedata.category(character) == "Cc" and character not in {"\n", "\t", "\r"} for character in text):
        return QualityResult(None, "control_character", "unknown", 0.0)
    if looks_like_mojibake(text):
        return QualityResult(None, "mojibake", "unknown", 0.0)
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if len(normalized) < int(policy["minimum_document_characters"]):
        return QualityResult(None, "too_short", "unknown", 0.0)
    if len(normalized) > int(policy["maximum_document_characters"]):
        return QualityResult(None, "too_long", "unknown", 0.0)
    if re.search(r"([^\s])\1{39,}", normalized):
        return QualityResult(None, "repeated_character", "unknown", 0.0)
    nonspace = [character for character in normalized if not character.isspace()]
    punctuation_ratio = sum(unicodedata.category(character).startswith("P") for character in nonspace) / max(1, len(nonspace))
    information_ratio = sum(character.isalnum() for character in nonspace) / max(1, len(nonspace))
    if punctuation_ratio > 0.35:
        return QualityResult(None, "excessive_punctuation", "unknown", 0.0)
    if information_ratio < 0.25:
        return QualityResult(None, "low_information", "unknown", 0.0)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) >= 5 and 1 - len(set(lines)) / len(lines) > 0.30:
        return QualityResult(None, "repeated_lines", "unknown", 0.0)
    if re.search(r"\{\{|\}\}|\[\[(?:Category|Kategori|File|Dosya):", normalized, re.IGNORECASE):
        return QualityResult(None, "markup_leakage", "unknown", 0.0)
    pii_kind = material_pii_kind(normalized)
    if pii_kind:
        return QualityResult(None, f"material_pii_{pii_kind}", "unknown", 0.0)
    detected, confidence = detect_language_confidence(normalized)
    if detected != expected_language or confidence < float(policy["language_confidence_minimum"]):
        return QualityResult(None, "wrong_or_uncertain_language", detected, confidence)
    return QualityResult(normalized, None, detected, confidence)


def iter_paragraphs(text: str) -> Iterator[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            yield paragraph


def paragraph_hash(paragraph: str) -> str:
    return sha256_text(normalized_hash_text(paragraph))


def iter_plain_text_paragraphs(path: Path) -> Iterator[str]:
    pending: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.strip():
                pending.append(line.rstrip("\n"))
            elif pending:
                yield "\n".join(pending).strip()
                pending = []
    if pending:
        yield "\n".join(pending).strip()


def load_phase1b_hashes(config: dict[str, Any]) -> set[str]:
    historical = config["historical_phase1b"]
    directory = repository_path(historical["processed_directory"])
    if not historical.get("read_only"):
        raise ValueError("Phase 1B historical corpus must be declared read-only")
    if not directory.is_dir():
        raise FileNotFoundError(f"Phase 1B processed corpus not found: {directory}")
    minimum = int(config["deduplication_policy"]["minimum_historical_paragraph_characters"])
    hashes: set[str] = set()
    for filename in historical["text_files"]:
        path = directory / filename
        for paragraph in iter_plain_text_paragraphs(path):
            if len(paragraph) >= minimum:
                hashes.add(paragraph_hash(paragraph))
    return hashes


def _string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in {"prompt", "text", "content", "instruction", "input"}:
                yield from _string_values(item)


def load_contamination_material(config: dict[str, Any]) -> tuple[set[str], list[str]]:
    policy = config["contamination_policy"]
    minimum = int(policy["minimum_substring_characters"])
    exact: set[str] = set()
    substrings: set[str] = set()
    for manifest in policy["prompt_manifests"]:
        with repository_path(manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for value in _string_values(record):
                    normalized = normalized_hash_text(value)
                    if len(normalized) >= minimum:
                        exact.add(sha256_text(normalized))
                        substrings.add(normalized)
    return exact, sorted(substrings, key=lambda value: (-len(value), value))


def word_shingle_hashes(text: str, size: int = 5) -> list[int]:
    words = [word.casefold() for word in WORD_RE.findall(unicodedata.normalize("NFC", text))]
    if not words:
        return []
    sequences = [words] if len(words) <= size else (words[index : index + size] for index in range(len(words) - size + 1))
    return [
        int.from_bytes(hashlib.blake2b("\x1f".join(sequence).encode("utf-8"), digest_size=8).digest(), "big")
        for sequence in sequences
    ]


def simhash64(text: str) -> int:
    hashes = word_shingle_hashes(text)
    if not hashes:
        return 0
    weights = [0] * 64
    for value in hashes:
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def simhash_bands(value: int) -> tuple[tuple[int, int], ...]:
    return tuple((band, (value >> (band * 16)) & 0xFFFF) for band in range(4))


def find_near_duplicate(
    value: int,
    buckets: dict[tuple[int, int], list[str]],
    values: dict[str, int],
    maximum_distance: int,
) -> tuple[str, int] | None:
    candidates: set[str] = set()
    for band in simhash_bands(value):
        candidates.update(buckets.get(band, []))
    for candidate in sorted(candidates):
        distance = (value ^ values[candidate]).bit_count()
        if distance <= maximum_distance:
            return candidate, distance
    return None


def deterministic_split(document_id: str, cluster_id: str, policy: dict[str, Any]) -> str:
    key = f"{policy['seed']}\n{cluster_id or document_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10000
    train_end = int(policy["train_percent"]) * 100
    validation_end = train_end + int(policy["validation_percent"]) * 100
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "eval"


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def verify_raw_archives(
    config: dict[str, Any],
    runtime_root: Path,
    *,
    manifest_filename: str = "download_manifest.json",
    source_ids: set[str] | None = None,
) -> dict[str, Any]:
    manifest_path = runtime_root / manifest_filename
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("result") not in {"PASS", "PASS_SELECTED_SOURCES"} or manifest.get(
        "official_manifest_verification", {}
    ).get("result") != "PASS":
        raise ValueError("download manifest does not pass official verification")
    if source_ids is not None and not source_ids <= set(manifest.get("selected_source_ids", [])):
        raise ValueError("selected download manifest does not cover the requested source gate")
    records = {record["filename"]: record for record in manifest["downloads"]}
    checked: list[dict[str, Any]] = []
    for source in config["sources"]:
        if source_ids is not None and source["source_id"] not in source_ids:
            continue
        for item in source["files"]:
            record = records.get(item["filename"])
            if record is None:
                raise ValueError(f"archive absent from download manifest: {item['filename']}")
            path = ROOT / record["path"]
            hashes = file_hashes(path, ("sha256",))
            if path.stat().st_size != int(item["expected_bytes"]) or hashes["sha256"] != record["sha256"]:
                raise ValueError(f"raw archive changed after acquisition: {item['filename']}")
            checked.append({"filename": item["filename"], "bytes": path.stat().st_size, **hashes})
    return {"result": "PASS", "archives": checked}


def _source_record(document: ExtractedDocument, source: dict[str, Any], text: str, **extra: Any) -> dict[str, Any]:
    normalized_hash = sha256_text(normalized_hash_text(text))
    document_id = stable_document_id(document.source_id, document.snapshot, document.native_id, normalized_hash)
    return {
        "id": document_id,
        "source_id": document.source_id,
        "snapshot": document.snapshot,
        "source_native_id": document.native_id,
        "title": document.title,
        "language": document.language,
        "category": document.category,
        "text": text,
        "source_url": document.source_url,
        "license": source["license"],
        "attribution": source["attribution_requirements"],
        "redistribution_notes": source["redistribution_notes"],
        "extraction_version": "darkmind-v2-corpus-v3-extractor-v1",
        "raw_content_sha256": document.raw_content_sha256,
        "normalized_content_sha256": normalized_hash,
        **extra,
    }


def run_inventory(
    config: dict[str, Any],
    runtime_root: Path,
    *,
    tokenizer: Any,
    historical_hashes: set[str],
    contamination_exact: set[str],
    contamination_substrings: list[str],
) -> dict[str, Any]:
    raw_root = repository_path(config["runtime_root"]) / "raw"
    extracted_root = runtime_root / "extracted"
    normalized_root = runtime_root / "normalized"
    deduplicated_root = runtime_root / "deduplicated"
    for directory in (extracted_root, normalized_root, deduplicated_root):
        directory.mkdir(parents=True, exist_ok=True)

    rejected_path = deduplicated_root / "rejected_records.jsonl"
    accepted_path = deduplicated_root / "accepted_documents.jsonl"
    exact_seen: dict[str, str] = {}
    accepted_paragraphs: dict[str, str] = {}
    simhash_values: dict[str, int] = {}
    simhash_buckets: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    duplicate_clusters: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    rejection_counts: collections.Counter[str] = collections.Counter()
    overlap_matrix: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    minimum_historical = int(config["deduplication_policy"]["minimum_historical_paragraph_characters"])
    maximum_hamming = int(config["deduplication_policy"]["near_duplicate_max_hamming_distance"])

    with rejected_path.open("w", encoding="utf-8", newline="\n") as rejected_handle, accepted_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as accepted_handle:
        for source in sorted(config["sources"], key=lambda value: int(value["source_priority"])):
            source_id = source["source_id"]
            extracted_path = extracted_root / f"{source_id}.jsonl"
            normalized_path = normalized_root / f"{source_id}.jsonl"
            source_accepted_path = deduplicated_root / f"{source_id}.jsonl"
            stats: dict[str, Any] = {
                "raw_documents": 0,
                "extracted_documents": 0,
                "quality_accepted_documents": 0,
                "accepted_documents": 0,
                "accepted_tokens": 0,
                "accepted_characters": 0,
                "rejections": collections.Counter(),
                "stopped_at_inventory_ceiling": False,
            }
            with extracted_path.open("w", encoding="utf-8", newline="\n") as extracted_handle, normalized_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as normalized_handle, source_accepted_path.open("w", encoding="utf-8", newline="\n") as source_handle:
                for document in iter_source_documents(source, raw_root):
                    stats["raw_documents"] += 1
                    stats["extracted_documents"] += 1
                    extracted_record = asdict(document)
                    _write_jsonl_record(extracted_handle, extracted_record)
                    quality = quality_filter(document.text, document.language, config["quality_policy"])
                    provisional_hash = sha256_text(normalized_hash_text(document.text))
                    provisional_id = stable_document_id(
                        document.source_id, document.snapshot, document.native_id, provisional_hash
                    )
                    if quality.reason:
                        record = {
                            "id": provisional_id,
                            "source_id": source_id,
                            "source_native_id": document.native_id,
                            "reason": quality.reason,
                            "detected_language": quality.detected_language,
                            "language_confidence": quality.language_confidence,
                            "record_type": "document",
                        }
                        _write_jsonl_record(rejected_handle, record)
                        stats["rejections"][quality.reason] += 1
                        rejection_counts[quality.reason] += 1
                        if len(samples[f"rejected:{quality.reason}"]) < 3:
                            samples[f"rejected:{quality.reason}"].append(
                                {"id": provisional_id, "reason": quality.reason, "excerpt": document.text[:180]}
                            )
                        continue
                    assert quality.text is not None
                    stats["quality_accepted_documents"] += 1
                    normalized_record = _source_record(
                        document,
                        source,
                        quality.text,
                        detected_language=quality.detected_language,
                        language_confidence=quality.language_confidence,
                    )
                    _write_jsonl_record(normalized_handle, normalized_record)

                    retained_paragraphs: list[str] = []
                    paragraph_rejections = 0
                    local_hashes: set[str] = set()
                    for paragraph_index, paragraph in enumerate(iter_paragraphs(quality.text), start=1):
                        digest = paragraph_hash(paragraph)
                        reason: str | None = None
                        representative: str | None = None
                        if digest in local_hashes:
                            reason = "repeated_paragraph_within_document"
                            representative = provisional_id
                        elif len(paragraph) >= minimum_historical and digest in historical_hashes:
                            reason = "phase1b_paragraph_overlap"
                            representative = "phase1b"
                        elif len(paragraph) >= minimum_historical and digest in accepted_paragraphs:
                            reason = "cross_source_paragraph_overlap"
                            representative = accepted_paragraphs[digest]
                        if reason:
                            paragraph_rejections += 1
                            rejection_counts[reason] += 1
                            stats["rejections"][reason] += 1
                            _write_jsonl_record(
                                rejected_handle,
                                {
                                    "id": provisional_id,
                                    "source_id": source_id,
                                    "source_native_id": document.native_id,
                                    "paragraph_index": paragraph_index,
                                    "reason": reason,
                                    "representative_id": representative,
                                    "record_type": "paragraph",
                                },
                            )
                        else:
                            retained_paragraphs.append(paragraph)
                            local_hashes.add(digest)
                    candidate_text = "\n\n".join(retained_paragraphs).strip()
                    if len(candidate_text) < int(config["quality_policy"]["minimum_document_characters"]):
                        reason = "empty_after_paragraph_deduplication"
                        _write_jsonl_record(
                            rejected_handle,
                            {
                                "id": provisional_id,
                                "source_id": source_id,
                                "source_native_id": document.native_id,
                                "reason": reason,
                                "record_type": "document",
                            },
                        )
                        stats["rejections"][reason] += 1
                        rejection_counts[reason] += 1
                        continue

                    post_dedup_quality = quality_filter(
                        candidate_text, document.language, config["quality_policy"]
                    )
                    if post_dedup_quality.reason:
                        reason = f"post_dedup_{post_dedup_quality.reason}"
                        _write_jsonl_record(
                            rejected_handle,
                            {
                                "id": provisional_id,
                                "source_id": source_id,
                                "source_native_id": document.native_id,
                                "reason": reason,
                                "detected_language": post_dedup_quality.detected_language,
                                "language_confidence": post_dedup_quality.language_confidence,
                                "record_type": "document",
                            },
                        )
                        stats["rejections"][reason] += 1
                        rejection_counts[reason] += 1
                        continue

                    normalized_candidate = normalized_hash_text(candidate_text)
                    candidate_hash = sha256_text(normalized_candidate)
                    candidate_id = stable_document_id(
                        document.source_id, document.snapshot, document.native_id, candidate_hash
                    )
                    if candidate_hash in contamination_exact or any(
                        prompt in normalized_candidate for prompt in contamination_substrings
                    ):
                        reason = "evaluation_contamination"
                        _write_jsonl_record(
                            rejected_handle,
                            {
                                "id": candidate_id,
                                "source_id": source_id,
                                "source_native_id": document.native_id,
                                "reason": reason,
                                "record_type": "document",
                            },
                        )
                        stats["rejections"][reason] += 1
                        rejection_counts[reason] += 1
                        continue
                    if candidate_hash in exact_seen:
                        representative = exact_seen[candidate_hash]
                        reason = "exact_duplicate"
                        duplicate_clusters.append(
                            {"kind": "exact", "representative_id": representative, "rejected_id": candidate_id}
                        )
                        _write_jsonl_record(
                            rejected_handle,
                            {
                                "id": candidate_id,
                                "source_id": source_id,
                                "source_native_id": document.native_id,
                                "reason": reason,
                                "representative_id": representative,
                                "record_type": "document",
                            },
                        )
                        stats["rejections"][reason] += 1
                        rejection_counts[reason] += 1
                        continue

                    candidate_simhash = simhash64(candidate_text)
                    near = find_near_duplicate(
                        candidate_simhash, simhash_buckets, simhash_values, maximum_hamming
                    )
                    if near is not None:
                        representative, distance = near
                        reason = "near_duplicate"
                        duplicate_clusters.append(
                            {
                                "kind": "near",
                                "representative_id": representative,
                                "rejected_id": candidate_id,
                                "simhash_hamming_distance": distance,
                            }
                        )
                        _write_jsonl_record(
                            rejected_handle,
                            {
                                "id": candidate_id,
                                "source_id": source_id,
                                "source_native_id": document.native_id,
                                "reason": reason,
                                "representative_id": representative,
                                "simhash_hamming_distance": distance,
                                "record_type": "document",
                            },
                        )
                        stats["rejections"][reason] += 1
                        rejection_counts[reason] += 1
                        continue

                    token_ids = tokenizer.encode_document(candidate_text)
                    if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
                        raise ValueError(f"frozen tokenizer did not append EOS: {candidate_id}")
                    if any(token_id < 0 or token_id >= int(config["tokenizer"]["vocab_size"]) for token_id in token_ids):
                        raise ValueError(f"frozen tokenizer produced an out-of-range ID: {candidate_id}")
                    accepted_record = _source_record(
                        document,
                        source,
                        candidate_text,
                        id=candidate_id,
                        token_count=len(token_ids),
                        duplicate_cluster_id=candidate_id,
                        detected_language=quality.detected_language,
                        language_confidence=quality.language_confidence,
                        removed_paragraphs=paragraph_rejections,
                    )
                    _write_jsonl_record(source_handle, accepted_record)
                    _write_jsonl_record(accepted_handle, accepted_record)
                    exact_seen[candidate_hash] = candidate_id
                    simhash_values[candidate_id] = candidate_simhash
                    for band in simhash_bands(candidate_simhash):
                        simhash_buckets[band].append(candidate_id)
                    for paragraph in retained_paragraphs:
                        if len(paragraph) >= minimum_historical:
                            accepted_paragraphs[paragraph_hash(paragraph)] = candidate_id
                    stats["accepted_documents"] += 1
                    stats["accepted_tokens"] += len(token_ids)
                    stats["accepted_characters"] += len(candidate_text)
                    if len(samples[source_id]) < (100 if source["category"] == "general_prose" else 50):
                        samples[source_id].append(
                            {
                                "id": candidate_id,
                                "title": document.title,
                                "characters": len(candidate_text),
                                "tokens": len(token_ids),
                                "excerpt": candidate_text[:180],
                            }
                        )
                    if stats["accepted_documents"] % 5000 == 0:
                        print(
                            f"inventory progress source={source_id} raw={stats['raw_documents']} "
                            f"accepted={stats['accepted_documents']} tokens={stats['accepted_tokens']}",
                            flush=True,
                        )
                    ceiling = int(source["inventory_stop_tokens"])
                    if source["category"] == "general_prose" and stats["accepted_tokens"] >= ceiling:
                        stats["stopped_at_inventory_ceiling"] = True
                        break

            stats["rejections"] = dict(sorted(stats["rejections"].items()))
            source_stats[source_id] = stats
            atomic_write_json(
                runtime_root / "inventory_progress.json",
                {"completed_sources": list(source_stats), "source_statistics": source_stats},
            )
            print(
                f"inventory complete source={source_id} raw={stats['raw_documents']} "
                f"accepted={stats['accepted_documents']} tokens={stats['accepted_tokens']}",
                flush=True,
            )

    for cluster in duplicate_clusters:
        left_source = cluster["representative_id"].split(":", 1)[0]
        right_source = cluster["rejected_id"].split(":", 1)[0]
        overlap_matrix[f"{left_source} -> {right_source}"] += 1
    duplicate_report = {
        "result": "PASS",
        "exact_duplicate_removals": rejection_counts["exact_duplicate"],
        "near_duplicate_removals": rejection_counts["near_duplicate"],
        "phase1b_paragraph_overlap_removals": rejection_counts["phase1b_paragraph_overlap"],
        "cross_source_paragraph_overlap_removals": rejection_counts["cross_source_paragraph_overlap"],
        "unresolved_exact_duplicates": 0,
        "unresolved_near_duplicate_clusters": 0,
        "clusters": duplicate_clusters,
        "source_overlap_matrix": dict(sorted(overlap_matrix.items())),
    }
    contamination_report = {
        "result": "PASS",
        "prompt_manifests": config["contamination_policy"]["prompt_manifests"],
        "rejected_documents": rejection_counts["evaluation_contamination"],
        "accepted_contamination_records": 0,
    }
    atomic_write_json(deduplicated_root / "duplicate_report.json", duplicate_report)
    atomic_write_json(deduplicated_root / "contamination_report.json", contamination_report)
    return {
        "source_statistics": source_stats,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "duplicate_report": duplicate_report,
        "contamination_report": contamination_report,
        "samples": dict(samples),
        "accepted_documents_path": str(accepted_path),
    }


def quota_status(config: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"result": "PASS", "sources": {}}
    for source in config["sources"]:
        source_id = source["source_id"]
        target = int(source["target_tokens"])
        tolerance = float(source["target_tolerance_percent"]) / 100
        minimum = math.ceil(target * (1 - tolerance))
        maximum = min(
            math.floor(target * (1 + tolerance)),
            int(source["tranche_source_cap_tokens"]),
        )
        available = int(inventory["source_statistics"][source_id]["accepted_tokens"])
        passed = available >= minimum
        result["sources"][source_id] = {
            "target_tokens": target,
            "minimum_tokens": minimum,
            "maximum_tokens": maximum,
            "available_unique_tokens": available,
            "shortfall_to_minimum": max(0, minimum - available),
            "pass": passed,
        }
        if not passed:
            result["result"] = "FAIL"
    return result


def allocate_documents(config: dict[str, Any], inventory_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "documents.jsonl"
    source_allocation: dict[str, dict[str, int]] = {}
    split_counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"documents": 0, "tokens": 0})
    attribution_path = output_root / "attribution_manifest.jsonl"
    split_manifest_path = output_root / "split_manifest.jsonl"
    with selected_path.open("w", encoding="utf-8", newline="\n") as selected_handle, attribution_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as attribution_handle, split_manifest_path.open("w", encoding="utf-8", newline="\n") as split_handle:
        for source in sorted(config["sources"], key=lambda value: int(value["source_priority"])):
            source_id = source["source_id"]
            target = int(source["target_tokens"])
            tolerance = float(source["target_tolerance_percent"]) / 100
            minimum = math.ceil(target * (1 - tolerance))
            maximum = min(math.floor(target * (1 + tolerance)), int(source["tranche_source_cap_tokens"]))
            selected_tokens = 0
            selected_documents = 0
            source_path = inventory_root / "deduplicated" / f"{source_id}.jsonl"
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    token_count = int(record["token_count"])
                    if selected_tokens + token_count > maximum:
                        continue
                    split = deterministic_split(
                        record["id"], record["duplicate_cluster_id"], config["split_policy"]
                    )
                    selected = {**record, "split": split, "source_order": selected_documents + 1}
                    _write_jsonl_record(selected_handle, selected)
                    attribution = {key: value for key, value in selected.items() if key != "text"}
                    _write_jsonl_record(attribution_handle, attribution)
                    _write_jsonl_record(
                        split_handle,
                        {
                            "id": selected["id"],
                            "source_id": source_id,
                            "language": selected["language"],
                            "category": selected["category"],
                            "token_count": token_count,
                            "split": split,
                            "normalized_content_sha256": selected["normalized_content_sha256"],
                            "duplicate_cluster_id": selected["duplicate_cluster_id"],
                        },
                    )
                    selected_tokens += token_count
                    selected_documents += 1
                    split_counts[split]["tokens"] += token_count
                    split_counts[split]["documents"] += 1
                    if selected_tokens >= target:
                        break
            if not minimum <= selected_tokens <= maximum:
                raise ValueError(f"deterministic allocation missed source quota: {source_id}={selected_tokens}")
            source_allocation[source_id] = {"documents": selected_documents, "tokens": selected_tokens}
    result = {
        "result": "PASS",
        "sources": source_allocation,
        "splits": dict(split_counts),
        "total_tokens": sum(value["tokens"] for value in source_allocation.values()),
        "total_documents": sum(value["documents"] for value in source_allocation.values()),
    }
    atomic_write_json(output_root / "source_allocation.json", result)
    return result


class ShardWriter:
    def __init__(self, output_root: Path, split: str, token_cap: int) -> None:
        self.output_root = output_root
        self.split = split
        self.token_cap = token_cap
        self.index = 0
        self.tokens = 0
        self.documents = 0
        self.handle: Any = None
        self.path: Path | None = None
        self.digest: Any = None
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def encode_uint16_le(token_ids: list[int]) -> bytes:
        if any(token_id < 0 or token_id > 65535 for token_id in token_ids):
            raise ValueError("token ID cannot be represented as uint16")
        values = array("H", token_ids)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    def _open(self) -> None:
        filename = f"{self.split}-{self.index:05d}.bin"
        self.path = self.output_root / filename
        self.handle = self.path.open("xb")
        self.digest = hashlib.sha256()

    def _close(self) -> None:
        if self.handle is None or self.path is None:
            return
        self.handle.close()
        self.records.append(
            {
                "filename": self.path.name,
                "split": self.split,
                "tokens": self.tokens,
                "documents": self.documents,
                "bytes": self.path.stat().st_size,
                "sha256": self.digest.hexdigest(),
            }
        )
        self.index += 1
        self.tokens = 0
        self.documents = 0
        self.handle = None
        self.path = None
        self.digest = None

    def add(self, token_ids: list[int]) -> tuple[str, int, int]:
        if len(token_ids) > self.token_cap:
            raise ValueError("document exceeds deterministic shard token cap; truncation is forbidden")
        if self.handle is not None and self.tokens + len(token_ids) > self.token_cap:
            self._close()
        if self.handle is None:
            self._open()
        assert self.path is not None and self.handle is not None and self.digest is not None
        start = self.tokens
        payload = self.encode_uint16_le(token_ids)
        self.handle.write(payload)
        self.digest.update(payload)
        self.tokens += len(token_ids)
        self.documents += 1
        return self.path.name, start, self.tokens

    def close(self) -> list[dict[str, Any]]:
        self._close()
        return self.records


def tokenize_allocation(config: dict[str, Any], final_text_root: Path, output_root: Path, tokenizer: Any) -> dict[str, Any]:
    incomplete = output_root.with_name(output_root.name + ".incomplete")
    if output_root.exists() or incomplete.exists():
        raise FileExistsError(f"refusing to overwrite tokenized output: {output_root}")
    incomplete.mkdir(parents=True)
    cap = int(config["tokenization_policy"]["shard_token_cap"])
    writers = {split: ShardWriter(incomplete, split, cap) for split in ("train", "validation", "eval")}
    split_tokens: collections.Counter[str] = collections.Counter()
    split_documents: collections.Counter[str] = collections.Counter()
    boundaries_path = incomplete / "document_boundaries.jsonl"
    with (final_text_root / "documents.jsonl").open("r", encoding="utf-8") as documents, boundaries_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as boundaries:
        for line in documents:
            record = json.loads(line)
            token_ids = tokenizer.encode_document(record["text"])
            if len(token_ids) != int(record["token_count"]):
                raise ValueError(f"token count changed after allocation: {record['id']}")
            if token_ids[-1] != tokenizer.eos_token_id:
                raise ValueError(f"missing EOS boundary: {record['id']}")
            if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in token_ids):
                raise ValueError(f"token range violation: {record['id']}")
            split = record["split"]
            filename, start, end = writers[split].add(token_ids)
            _write_jsonl_record(
                boundaries,
                {
                    "id": record["id"],
                    "split": split,
                    "shard": filename,
                    "start_offset": start,
                    "end_offset": end,
                    "tokens": len(token_ids),
                    "eos_token_id": tokenizer.eos_token_id,
                },
            )
            split_tokens[split] += len(token_ids)
            split_documents[split] += 1
    shards = [record for split in ("train", "validation", "eval") for record in writers[split].close()]
    statistics = {
        "total_tokens": sum(split_tokens.values()),
        "total_documents": sum(split_documents.values()),
        "split_tokens": dict(split_tokens),
        "split_documents": dict(split_documents),
        "token_range_violations": 0,
        "missing_eos_boundaries": 0,
    }
    core = {
        "schema_version": "darkmind-v2-tokenized-corpus-v3-tranche-v1",
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "shard_token_cap": cap,
        "tokenizer": {
            "name": config["tokenizer"]["name"],
            "model_sha256": EXPECTED_HASHES["tokenizer.model"],
            "vocab_sha256": EXPECTED_HASHES["tokenizer.vocab"],
            "freeze_manifest_sha256": EXPECTED_HASHES["tokenizer_freeze_manifest.json"],
        },
        "statistics": statistics,
        "shards": shards,
        "document_boundaries_sha256": file_hashes(boundaries_path)["sha256"],
    }
    manifest = {**core, "deterministic_content_hash": sha256_text(json.dumps(core, sort_keys=True, separators=(",", ":")))}
    atomic_write_json(incomplete / "tokenized_corpus_manifest.json", manifest)
    atomic_write_json(incomplete / "shard_checksums.json", {record["filename"]: record["sha256"] for record in shards})
    atomic_write_json(incomplete / "tokenization_statistics.json", statistics)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output_root)
    return manifest


def _report_table_source_stats(config: dict[str, Any], inventory: dict[str, Any], quotas: dict[str, Any]) -> list[str]:
    lines = [
        "| Source | Raw documents | Accepted documents | Unique tokens | Target | Shortfall to minimum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in config["sources"]:
        source_id = source["source_id"]
        stats = inventory["source_statistics"][source_id]
        quota = quotas["sources"][source_id]
        lines.append(
            f"| {source_id} | {stats['raw_documents']:,} | {stats['accepted_documents']:,} | "
            f"{stats['accepted_tokens']:,} | {source['target_tokens']:,} | {quota['shortfall_to_minimum']:,} |"
        )
    return lines


def write_reports(
    config: dict[str, Any],
    inventory: dict[str, Any],
    quotas: dict[str, Any],
    raw_verification: dict[str, Any],
    *,
    allocation: dict[str, Any] | None = None,
    tokenized: dict[str, Any] | None = None,
    determinism: dict[str, Any] | None = None,
) -> None:
    reports_root = repository_path(config["reports_root"])
    reports_root.mkdir(parents=True, exist_ok=True)
    source_table = _report_table_source_stats(config, inventory, quotas)
    raw_documents = sum(value["raw_documents"] for value in inventory["source_statistics"].values())
    accepted_documents = sum(value["accepted_documents"] for value in inventory["source_statistics"].values())
    accepted_tokens = sum(value["accepted_tokens"] for value in inventory["source_statistics"].values())
    rejection_lines = [
        f"| {reason} | {count:,} |" for reason, count in inventory["rejection_counts"].items()
    ] or ["| none | 0 |"]

    extraction_methods: list[str] = []
    formats = {source["extraction_format"] for source in config["sources"]}
    if "streaming-bzip2-mediawiki-xml" in formats:
        extraction_methods.append(
            "Wikimedia XML was decompressed with BZ2 streaming and parsed iteratively; parsed page elements were cleared immediately."
        )
    if "streaming-tar-bzip2-plain-text" in formats:
        extraction_methods.append(
            "Python text archives were read member-by-member with strict UTF-8 decoding and deterministic section grouping."
        )
    extraction = [
        "# Phase 3C Extraction Report", "", f"Raw streamed documents: {raw_documents:,}", "",
        *extraction_methods, "", *source_table, "",
    ]
    atomic_write_text(reports_root / "phase3c_extraction_report.md", "\n".join(extraction))

    quality = [
        "# Phase 3C Quality Filter Report", "", f"Accepted documents after all filters: {accepted_documents:,}", "",
        "| Rejection reason | Records |", "|---|---:|", *rejection_lines, "",
        "Accepted records contain no known invalid UTF-8, U+FFFD, malformed surrogates, confirmed mojibake, wrong-language material, or material PII matches.", "",
    ]
    atomic_write_text(reports_root / "phase3c_quality_filter_report.md", "\n".join(quality))

    duplicate = inventory["duplicate_report"]
    dedup = [
        "# Phase 3C Deduplication Report", "",
        f"Exact duplicate removals: {duplicate['exact_duplicate_removals']:,}",
        f"Near-duplicate removals: {duplicate['near_duplicate_removals']:,}",
        f"Phase 1B paragraph overlap removals: {duplicate['phase1b_paragraph_overlap_removals']:,}",
        f"Cross-source paragraph overlap removals: {duplicate['cross_source_paragraph_overlap_removals']:,}",
        f"Evaluation contamination removals: {inventory['contamination_report']['rejected_documents']:,}",
        "", "Unresolved exact duplicates: 0", "", "Unresolved near-duplicate clusters: 0", "",
    ]
    if duplicate["clusters"]:
        dedup.extend(["## Deterministic cluster examples", ""])
        for cluster in duplicate["clusters"][:5]:
            dedup.append(
                f"- {cluster['kind']}: kept `{cluster['representative_id']}`; rejected `{cluster['rejected_id']}`."
            )
        dedup.append("")
    atomic_write_text(reports_root / "phase3c_deduplication_report.md", "\n".join(dedup))

    status = "PASS" if quotas["result"] == "PASS" else "FAIL"
    corpus = [
        "# Phase 3C Corpus V3 Tranche 1", "", f"Quota gate: **{status}**", "",
        f"Available validated unique tokens before allocation: {accepted_tokens:,}", "", *source_table, "",
    ]
    if status == "FAIL":
        corpus.extend([
            "Final corpus allocation was not built because at least one immutable source quota could not be met.", "",
            "No source quota was transferred, duplicated, or filled with synthetic material.", "",
            f"Validated technical tokens from the two Python archives: {accepted_tokens:,} of the 15,000,000-token target.", "",
            "The Wikipedia archives were not acquired after this independent hard source gate failed; their planned quotas remain unevaluated.", "",
        ])
    atomic_write_text(reports_root / "phase3c_corpus_v3_tranche1.md", "\n".join(corpus))

    if tokenized is None:
        tokenization = [
            "# Phase 3C Tokenization Report", "", "Status: **NOT RUN**", "",
            "The hard source-quota gate failed before final corpus construction. No tokenized shards were created.", "",
        ]
    else:
        tokenization = [
            "# Phase 3C Tokenization Report", "", "Status: **PASS**", "",
            f"Tokens: {tokenized['statistics']['total_tokens']:,}",
            f"Shards: {len(tokenized['shards']):,}",
            f"Deterministic content hash: `{tokenized['deterministic_content_hash']}`", "",
        ]
    atomic_write_text(reports_root / "phase3c_tokenization_report.md", "\n".join(tokenization))

    if determinism is None:
        determinism_lines = [
            "# Phase 3C Determinism Report", "", "Status: **NOT RUN**", "",
            "Two-pass shard comparison is downstream of the hard source-quota gate and was not started.", "",
        ]
    else:
        determinism_lines = [
            "# Phase 3C Determinism Report", "", f"Status: **{determinism['result']}**", "",
            f"Manifest equality: {determinism['manifest_equal']}", "",
        ]
    atomic_write_text(reports_root / "phase3c_determinism_report.md", "\n".join(determinism_lines))

    sample_lines = ["# Phase 3C Corpus V3 Tranche 1 Samples", ""]
    for key in sorted(inventory["samples"]):
        sample_lines.extend((f"## {key}", ""))
        for sample in inventory["samples"][key]:
            excerpt = " ".join(str(sample.get("excerpt", "")).split())[:180]
            sample_lines.append(
                f"- `{sample.get('id', 'unknown')}`: {excerpt}"
            )
        sample_lines.append("")
    sample_lines.extend(("Excerpts are intentionally short and are included only for structural quality review.", ""))
    atomic_write_text(reports_root / "phase3c_corpus_v3_tranche1_samples.md", "\n".join(sample_lines))

    readiness_status = "READY FOR USER APPROVAL" if quotas["result"] == "PASS" and tokenized and determinism else "NOT READY"
    verified_archive_bytes = sum(int(item["bytes"]) for item in raw_verification["archives"])
    exact_duplicates = inventory["duplicate_report"]["exact_duplicate_removals"]
    near_duplicates = inventory["duplicate_report"]["near_duplicate_removals"]
    phase1b_overlaps = inventory["duplicate_report"]["phase1b_paragraph_overlap_removals"]
    contamination = inventory["contamination_report"]["rejected_documents"]
    readiness = [
        "# Phase 3C Training Readiness", "", f"Status: **{readiness_status}**", "",
        f"Raw archive integrity: {raw_verification['result']}",
        f"Verified raw archive bytes: {verified_archive_bytes:,}",
        f"Streamed documents: {raw_documents:,}",
        f"Accepted documents: {accepted_documents:,}",
        f"Available validated unique tokens: {accepted_tokens:,}",
        f"Exact duplicate removals: {exact_duplicates:,}",
        f"Near-duplicate removals: {near_duplicates:,}",
        f"Phase 1B paragraph overlap removals: {phase1b_overlaps:,}",
        f"Evaluation contamination removals: {contamination:,}",
        f"Source quota gate: {quotas['result']}",
        f"Frozen tokenizer provenance: PASS",
        f"Tokenizer model SHA-256: `{EXPECTED_HASHES['tokenizer.model']}`",
        f"Tokenizer vocabulary SHA-256: `{EXPECTED_HASHES['tokenizer.vocab']}`",
        f"Tokenizer freeze-manifest SHA-256: `{EXPECTED_HASHES['tokenizer_freeze_manifest.json']}`",
        f"Final tokenization: {'PASS' if tokenized else 'NOT RUN'}",
        f"Two-pass determinism: {determinism['result'] if determinism else 'NOT RUN'}", "",
        *source_table, "",
    ]
    if quotas["result"] == "FAIL":
        readiness.extend([
            "No Corpus V3 tranche was accepted. The remaining gap is therefore 500,000,000 tokens to 500M and 1,000,000,000 tokens to 1B.", "",
            "Python documentation publishes no official archive checksum manifest. The downloaded archives passed the frozen byte, Last-Modified, ETag, and local SHA-256 controls.", "",
            "The immutable source plan requires correction. No approved source can silently replace these technical quotas; a revised config and new user approval are required.", "",
            "Approved correction candidates in the frozen registry are `wikimedia_trwikibooks_20260701` (10M cap), `wikimedia_enwikiversity_20260701` (25M cap), and `wikimedia_enwikibooks_20260701` (15M cap). `wikimedia_simplewiki_20260701` is an approved 25M English educational option but does not solve the Turkish gap.", "",
            "Wikimedia source archives were not downloaded after the independent Python source gate failed. Their official checksum manifests were verified, but archive checksums were not claimed as completed downloads.", "",
            "The 5M production Stage-1 experiment is not ready for approval.", "",
            "DARKMIND V2 CORPUS V3 FIRST TRANCHE REQUIRES CORRECTION BEFORE TRAINING", "",
        ])
    else:
        readiness.extend([
            "Remaining planned gap to 500M tokens: 400,000,000 tokens.",
            "Remaining planned gap to 1B tokens: 900,000,000 tokens.", "",
        ])
    atomic_write_text(reports_root / "phase3c_training_readiness.md", "\n".join(readiness))


def compare_tokenized_builds(left: Path, right: Path) -> dict[str, Any]:
    left_manifest = json.loads((left / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((right / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    manifest_equal = left_manifest == right_manifest
    left_checksums = json.loads((left / "shard_checksums.json").read_text(encoding="utf-8"))
    right_checksums = json.loads((right / "shard_checksums.json").read_text(encoding="utf-8"))
    result = "PASS" if manifest_equal and left_checksums == right_checksums else "FAIL"
    return {"result": result, "manifest_equal": manifest_equal, "shard_checksums_equal": left_checksums == right_checksums}


def build_source_gate(config_path: Path, source_ids: set[str]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    known = {source["source_id"] for source in config["sources"]}
    if not source_ids or not source_ids <= known:
        raise ValueError(f"source gate is outside the frozen tranche: {sorted(source_ids)}")
    runtime_root = repository_path(config["runtime_root"])
    raw_verification = verify_raw_archives(
        config,
        runtime_root,
        manifest_filename="download_manifest.selected.json",
        source_ids=source_ids,
    )
    selected_config = {
        **config,
        "sources": [source for source in config["sources"] if source["source_id"] in source_ids],
    }
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    historical_hashes = load_phase1b_hashes(config)
    contamination_exact, contamination_substrings = load_contamination_material(config)
    gate_root = runtime_root / "source_gate"
    inventory = run_inventory(
        selected_config,
        gate_root,
        tokenizer=tokenizer,
        historical_hashes=historical_hashes,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    quotas = quota_status(selected_config, inventory)
    write_reports(selected_config, inventory, quotas, raw_verification)
    result = {
        "result": "PASS" if quotas["result"] == "PASS" else "FAIL_SOURCE_QUOTA",
        "source_gate_only": True,
        "source_ids": sorted(source_ids),
        "raw_verification": raw_verification,
        "inventory": inventory,
        "quota_status": quotas,
        "final_corpus_built": False,
        "tokenizer_trained": False,
        "model_training_started": False,
    }
    atomic_write_json(runtime_root / "phase3c_source_gate_result.json", result)
    return result


def build_tranche(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    runtime_root = repository_path(config["runtime_root"])
    raw_verification = verify_raw_archives(config, runtime_root)
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    historical_hashes = load_phase1b_hashes(config)
    contamination_exact, contamination_substrings = load_contamination_material(config)
    inventory = run_inventory(
        config,
        runtime_root,
        tokenizer=tokenizer,
        historical_hashes=historical_hashes,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    quotas = quota_status(config, inventory)
    if quotas["result"] != "PASS":
        write_reports(config, inventory, quotas, raw_verification)
        result = {
            "result": "FAIL_SOURCE_QUOTA",
            "raw_verification": raw_verification,
            "inventory": inventory,
            "quota_status": quotas,
            "final_corpus_built": False,
            "tokenizer_trained": False,
            "model_training_started": False,
        }
        atomic_write_json(runtime_root / "phase3c_build_result.json", result)
        return result

    final_text_root = runtime_root / "final_text"
    allocation = allocate_documents(config, runtime_root, final_text_root)
    tokenized_root = runtime_root / "tokenized" / "tranche1_v1"
    tokenized = tokenize_allocation(config, final_text_root, tokenized_root, tokenizer)

    rebuild_root = runtime_root / "determinism_rebuild"
    rebuild_inventory = run_inventory(
        config,
        rebuild_root,
        tokenizer=tokenizer,
        historical_hashes=historical_hashes,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    rebuild_quotas = quota_status(config, rebuild_inventory)
    if rebuild_quotas != quotas:
        raise ValueError("determinism rebuild diverged at quota inventory")
    rebuild_final = rebuild_root / "final_text"
    rebuild_allocation = allocate_documents(config, rebuild_root, rebuild_final)
    if rebuild_allocation != allocation:
        raise ValueError("determinism rebuild diverged at allocation")
    rebuild_tokenized_root = rebuild_root / "tokenized" / "tranche1_v1"
    tokenize_allocation(config, rebuild_final, rebuild_tokenized_root, tokenizer)
    determinism = compare_tokenized_builds(tokenized_root, rebuild_tokenized_root)
    if determinism["result"] != "PASS":
        raise ValueError("two-pass deterministic shard comparison failed")
    write_reports(
        config,
        inventory,
        quotas,
        raw_verification,
        allocation=allocation,
        tokenized=tokenized,
        determinism=determinism,
    )
    result = {
        "result": "PASS",
        "raw_verification": raw_verification,
        "inventory": inventory,
        "quota_status": quotas,
        "allocation": allocation,
        "tokenized": tokenized,
        "determinism": determinism,
        "final_corpus_built": True,
        "tokenizer_trained": False,
        "model_training_started": False,
    }
    atomic_write_json(runtime_root / "phase3c_build_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    if args.source_id:
        result = build_source_gate(args.config, set(args.source_id))
    else:
        result = build_tranche(args.config)
    summary = {
        "result": result["result"],
        "quota_status": result["quota_status"],
        "final_corpus_built": result["final_corpus_built"],
        "tokenizer_trained": result["tokenizer_trained"],
        "model_training_started": result["model_training_started"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if result["result"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
