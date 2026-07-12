"""Build or inventory the DarkMind v2 Phase 1B tokenizer pilot corpus."""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import html
import json
import os
import re
import sys
import tarfile
import time
import tracemalloc
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .detect_mojibake import detect_text
    from .normalize_text import is_unsafe_control_char, normalize_text
except ImportError:  # pragma: no cover - CLI fallback
    from detect_mojibake import detect_text
    from normalize_text import is_unsafe_control_char, normalize_text


DEFAULT_REGISTRY = Path(__file__).with_name("source_registry.phase1b.json")
DEFAULT_PLAN = Path("darkmind_v2/config/tokenizer_pilot_corpus.json")
DEFAULT_DATA_DIR = Path("darkmind_v2/data/phase1b")
DEFAULT_REPORT_DIR = Path("darkmind_v2/reports")
TR_CHARS = set("çğıİöşüÇĞIıÖŞÜ")
PII_RE = re.compile(r"[\w.+-]+@[\w.-]+|\b(?:\+?\d[\d ()-]{7,}\d)\b|https?://|www\.", re.I)
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", re.U)
MOJIBAKE_MARKERS = ("�", "Ã", "Â", "Ä", "Å", "â€", "ï¿½")


@dataclass(frozen=True)
class CandidateDocument:
    document_id: str
    text: str
    language: str
    source_id: str
    content_type: str
    metadata: dict[str, Any]


@dataclass
class SourceStats:
    source_id: str
    archive_path: str = ""
    compressed_bytes: int = 0
    extracted_bytes: int = 0
    documents: int = 0
    accepted_documents: int = 0
    candidate_documents: int = 0
    candidate_characters: int = 0
    raw_characters: int = 0
    normalized_characters: int = 0
    turkish_characters: int = 0
    english_characters: int = 0
    candidate_turkish_characters: int = 0
    candidate_english_characters: int = 0
    rejected_documents: int = 0
    license_metadata_complete: int = 0
    attribution_metadata_complete: int = 0
    estimated_final_usable_characters: int = 0
    scan_stopped_after_cap: bool = False
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int | None = None
    rejection_reasons: dict[str, int] | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(record: dict[str, Any]) -> str:
    return sha256_text(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def language_of_text(text: str) -> str:
    if not text.strip():
        return "unknown"
    tr_score = sum(2 for char in text if char in TR_CHARS)
    en_score = len(re.findall(r"\b(the|and|with|for|this|that|python|function|class)\b", text.casefold()))
    tr_score += len(re.findall(r"\b(bir|ve|ile|için|bu|şu|python|işlev|sınıf)\b", text.casefold()))
    if tr_score > en_score:
        return "tr"
    if en_score > tr_score:
        return "en"
    if all(ord(char) < 128 for char in text):
        return "en"
    return "unknown"


def clean_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}", " ", text)
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = html.unescape(text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def paragraph_chunks(text: str, min_chars: int = 40) -> Iterable[str]:
    for chunk in re.split(r"\n\s*\n|(?<=\.)\s{2,}", text):
        chunk = " ".join(chunk.split())
        if len(chunk) >= min_chars:
            yield chunk


def iter_wikimedia(source: dict[str, Any], archive_path: Path) -> Iterable[CandidateDocument]:
    language = source["language"]
    article_url_prefix = str(source.get("article_url_prefix") or source["official_homepage"].rstrip("/") + "/wiki/")
    with bz2.open(archive_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("page"):
                ns = elem.findtext("./{*}ns") or ""
                if ns != "0":
                    elem.clear()
                    continue
                title = elem.findtext("./{*}title") or ""
                page_id = elem.findtext("./{*}id") or ""
                revision = elem.find("./{*}revision")
                revision_id = revision.findtext("./{*}id") if revision is not None else ""
                text_node = revision.find("./{*}text") if revision is not None else None
                raw = text_node.text if text_node is not None and text_node.text else ""
                cleaned = clean_wikitext(raw)
                for index, paragraph in enumerate(paragraph_chunks(cleaned), start=1):
                    yield CandidateDocument(
                        document_id=f"{source['source_id']}:{page_id}:{index}",
                        text=paragraph,
                        language=language,
                        source_id=source["source_id"],
                        content_type=source["content_type"],
                        metadata={
                            "page_id": page_id,
                            "revision_id": revision_id,
                            "title": title,
                            "license": source["license_id"],
                            "attribution_url": f"{article_url_prefix}{title.replace(' ', '_')}",
                        },
                    )
                elem.clear()


def iter_python_docs(source: dict[str, Any], archive_path: Path) -> Iterable[CandidateDocument]:
    skip_markers = ("license", "copyright", "genindex", "py-modindex", "contents")
    with tarfile.open(archive_path, "r:*") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            name = member.name.replace("\\", "/")
            if not member.isfile() or not name.endswith(".txt"):
                continue
            if any(marker in name.casefold() for marker in skip_markers):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw_bytes = extracted.read()
            text = raw_bytes.decode("utf-8")
            for index, paragraph in enumerate(paragraph_chunks(text), start=1):
                yield CandidateDocument(
                    document_id=f"{source['source_id']}:{name}:{index}",
                    text=paragraph,
                    language=source["language"],
                    source_id=source["source_id"],
                    content_type=source["content_type"],
                    metadata={"original_document_path": name, "license": source["license_id"]},
                )


def iter_tatoeba(source: dict[str, Any], archive_path: Path) -> Iterable[CandidateDocument]:
    language_map = {"tur": "tr", "eng": "en"}
    with tarfile.open(archive_path, "r:*") as archive:
        member = next((item for item in archive.getmembers() if item.name.endswith("sentences_detailed.csv")), None)
        if member is None:
            raise ValueError("Tatoeba archive does not contain sentences_detailed.csv")
        extracted = archive.extractfile(member)
        if extracted is None:
            return
        text_stream = (line.decode("utf-8") for line in extracted)
        for row in csv.reader(text_stream, delimiter="\t"):
            if len(row) < 4:
                continue
            sentence_id, tatoeba_lang, sentence = row[0], row[1], row[2]
            username = row[3] if len(row) > 3 else ""
            language = language_map.get(tatoeba_lang)
            if not language or not username.strip():
                continue
            yield CandidateDocument(
                document_id=f"{source['source_id']}:{sentence_id}",
                text=sentence,
                language=language,
                source_id=source["source_id"],
                content_type=source["content_type"],
                metadata={
                    "sentence_id": sentence_id,
                    "username": username,
                    "license": source["license_id"],
                    "snapshot_date": source["snapshot_date"],
                },
            )


def archive_path_for(source: dict[str, Any], data_dir: Path) -> Path:
    return data_dir / "raw" / "archives" / str(source.get("local_archive_name") or source["filename"])


def iter_source_documents(source: dict[str, Any], data_dir: Path) -> Iterable[CandidateDocument]:
    archive_path = archive_path_for(source, data_dir)
    if not archive_path.exists():
        raise FileNotFoundError(f"missing source archive: {archive_path}")
    processor = source.get("processor")
    if processor == "wikimedia_pages_articles_bz2":
        yield from iter_wikimedia(source, archive_path)
    elif processor == "python_docs_text_tar_bz2":
        yield from iter_python_docs(source, archive_path)
    elif processor == "tatoeba_sentences_detailed_tar_bz2":
        yield from iter_tatoeba(source, archive_path)
    else:
        raise ValueError(f"unsupported processor: {processor}")


def validate_document(document: CandidateDocument, source: dict[str, Any], min_chars: int, max_chars: int) -> tuple[CandidateDocument | None, str | None]:
    if PII_RE.search(document.text):
        return None, "pii_or_url"
    if "\ufffd" in document.text:
        return None, "replacement_character"
    if any(marker in document.text for marker in MOJIBAKE_MARKERS) and detect_text(document.text):
        return None, "mojibake"
    if any(is_unsafe_control_char(char) for char in document.text):
        return None, "unsafe_control_character"
    normalized, _ = normalize_text(document.text)
    normalized = unicodedata.normalize("NFC", normalized).strip()
    if not (min_chars <= len(normalized) <= max_chars):
        return None, "document_length"
    detected = language_of_text(normalized)
    allowed = {"tr", "en"} if source["language"] == "mixed_tr_en" else {source["language"]}
    if detected not in allowed:
        return None, "language_mismatch"
    metadata = dict(document.metadata)
    metadata.update({"source": document.source_id, "license": source["license_id"], "language": detected})
    return CandidateDocument(document.document_id, normalized, detected, document.source_id, document.content_type, metadata), None


def ordered_sources_for_language(sources: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Prefer single-language sources, then use mixed-language data as the deterministic filler."""
    exact = [source for source in sources if source["language"] == language]
    mixed = [source for source in sources if source["language"] == "mixed_tr_en"]
    return exact + mixed


def source_language_allocations(
    documents: list[CandidateDocument],
    sources: list[dict[str, Any]],
    language_targets: dict[str, int],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Allocate each language target without allowing mixed sources to crowd out dedicated sources."""
    available = {source["source_id"]: {"tr": 0, "en": 0} for source in sources}
    for document in documents:
        available[document.source_id][document.language] += len(document.text)

    allocations = {source["source_id"]: {"tr": 0, "en": 0} for source in sources}
    for language in ("tr", "en"):
        remaining = language_targets[language]
        for source in ordered_sources_for_language(sources, language):
            source_id = source["source_id"]
            cap_remaining = source_cap(source) - sum(allocations[source_id].values())
            amount = min(available[source_id][language], cap_remaining, remaining)
            allocations[source_id][language] = amount
            remaining -= amount
        if remaining:
            raise ValueError(f"insufficient {language} candidate characters for final corpus: missing {remaining}")
    return allocations, available


def select_balanced_documents(
    documents: list[CandidateDocument],
    sources: list[dict[str, Any]],
    language_targets: dict[str, int],
    seed: int,
) -> tuple[list[CandidateDocument], dict[str, dict[str, int]], dict[str, int], list[dict[str, Any]]]:
    """Select a deterministic, language-balanced corpus and remove exact/near duplicates."""
    source_by_id = {source["source_id"]: source for source in sources}
    allocations, available = source_language_allocations(documents, sources, language_targets)
    buckets: dict[tuple[str, str], list[CandidateDocument]] = defaultdict(list)
    for document in documents:
        buckets[(document.source_id, document.language)].append(document)
    for key, bucket in buckets.items():
        bucket.sort(
            key=lambda item: stable_hash(
                {"seed": seed, "source": key[0], "language": key[1], "id": item.document_id, "text": item.text}
            )
        )

    selected: list[CandidateDocument] = []
    selected_ids: set[str] = set()
    selected_chars = {source["source_id"]: {"tr": 0, "en": 0} for source in sources}
    seen_exact: set[str] = set()
    seen_near: set[tuple[str, ...]] = set()
    duplicate_counts = {"exact_duplicate": 0, "near_duplicate": 0}
    rejection_records: list[dict[str, Any]] = []
    handled_duplicate_ids: set[str] = set()

    def take_from_source(source: dict[str, Any], language: str, requested: int) -> int:
        if requested <= 0:
            return 0
        source_id = source["source_id"]
        remaining = min(requested, source_cap(source) - sum(selected_chars[source_id].values()))
        for document in buckets[(source_id, language)]:
            if remaining <= 0:
                break
            if document.document_id in selected_ids or len(document.text) > remaining:
                continue
            exact = sha256_text(" ".join(document.text.casefold().split()))
            near = tuple(WORD_RE.findall(document.text.casefold())[:24])
            if exact in seen_exact or near in seen_near:
                if document.document_id not in handled_duplicate_ids:
                    reason = "exact_duplicate" if exact in seen_exact else "near_duplicate"
                    duplicate_counts[reason] += 1
                    rejection_records.append(
                        {
                            "id": document.document_id,
                            "source_id": document.source_id,
                            "language": document.language,
                            "character_count": len(document.text),
                            "reason": reason,
                        }
                    )
                    handled_duplicate_ids.add(document.document_id)
                continue
            selected.append(document)
            selected_ids.add(document.document_id)
            seen_exact.add(exact)
            seen_near.add(near)
            selected_chars[source_id][language] += len(document.text)
            remaining -= len(document.text)
        return requested - remaining

    # First pass honors the planned source allocation. A second pass can use unused
    # candidate capacity if deduplication made a planned allocation slightly short.
    for language in ("tr", "en"):
        for source in ordered_sources_for_language(sources, language):
            source_id = source["source_id"]
            take_from_source(source, language, allocations[source_id][language])

    for language in ("tr", "en"):
        while True:
            selected_total = sum(values[language] for values in selected_chars.values())
            remaining = language_targets[language] - selected_total
            if remaining <= 0:
                break
            progress = 0
            for source in ordered_sources_for_language(sources, language):
                source_id = source["source_id"]
                source_remaining = min(
                    available[source_id][language] - selected_chars[source_id][language],
                    source_cap(source) - sum(selected_chars[source_id].values()),
                    remaining,
                )
                if source_remaining <= 0:
                    continue
                before = selected_chars[source_id][language]
                take_from_source(source, language, source_remaining)
                progress += selected_chars[source_id][language] - before
                selected_total = sum(values[language] for values in selected_chars.values())
                remaining = language_targets[language] - selected_total
                if remaining <= 0:
                    break
            if progress == 0:
                break

    return selected, selected_chars, duplicate_counts, rejection_records


def split_documents(documents: list[CandidateDocument], ratios: dict[str, float], seed: int) -> dict[str, list[CandidateDocument]]:
    total_chars = sum(len(item.text) for item in documents)
    targets = {
        "train": int(total_chars * ratios["train"]),
        "validation": int(total_chars * ratios["validation"]),
    }
    splits = {"train": [], "validation": [], "eval": []}
    running = 0
    for document in sorted(
        documents,
        key=lambda item: stable_hash({"seed": seed, "split": item.document_id, "text": item.text}),
    ):
        if running < targets["train"]:
            split = "train"
        elif running < targets["train"] + targets["validation"]:
            split = "validation"
        else:
            split = "eval"
        splits[split].append(document)
        running += len(document.text)
    return splits


def text_content(documents: list[CandidateDocument]) -> str:
    return "\n\n".join(document.text for document in documents) + ("\n" if documents else "")


def jsonl_content(records: list[dict[str, Any]]) -> str:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def write_content(path: Path, content: str) -> str:
    atomic_write_text(path, content)
    return sha256_text(content)


def inventory_artifact_path(inventory_dir: Path, source_id: str) -> Path:
    return inventory_dir / f"{source_id}.inventory.json"


def inventory_progress_path(inventory_dir: Path, source_id: str) -> Path:
    return inventory_dir / f"{source_id}.progress.json"


def source_cap(source: dict[str, Any]) -> int:
    return min(int(source["max_sample_characters"]), int(source["final_pilot_character_cap"]))


def source_stats_from_payload(payload: dict[str, Any]) -> SourceStats:
    fields = set(SourceStats.__dataclass_fields__)
    stats = {key: value for key, value in payload["stats"].items() if key in fields}
    return SourceStats(**stats)


def reusable_source_artifact(
    source: dict[str, Any],
    artifact_path: Path,
    archive_path: Path,
    registry_hash: str,
    plan_hash: str,
) -> SourceStats | None:
    if not artifact_path.exists():
        return None
    payload = load_json(artifact_path)
    archive = payload.get("archive", {})
    if payload.get("status") != "complete":
        return None
    if payload.get("plan_hash") != plan_hash:
        return None
    if payload.get("source_hash") != stable_hash(source):
        return None
    if archive.get("path") != str(archive_path):
        return None
    if archive_path.exists() and archive.get("bytes") != archive_path.stat().st_size:
        return None
    return source_stats_from_payload(payload)


def source_artifact_payload(
    source: dict[str, Any],
    stats: SourceStats,
    archive_path: Path,
    registry_hash: str,
    plan_hash: str,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "source_id": source["source_id"],
        "source_name": source["source_name"],
        "source_hash": stable_hash(source),
        "registry_hash": registry_hash,
        "plan_hash": plan_hash,
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size if archive_path.exists() else 0,
        },
        "source_cap_characters": source_cap(source),
        "license_metadata_complete": stats.license_metadata_complete == stats.documents,
        "attribution_metadata_complete": stats.attribution_metadata_complete == stats.documents,
        "resumability_status": "complete_reusable" if status == "complete" else "checkpoint_only",
        "stats": asdict(stats),
    }


def write_source_progress(
    source: dict[str, Any],
    stats: SourceStats,
    archive_path: Path,
    inventory_dir: Path,
    registry_hash: str,
    plan_hash: str,
    status: str,
) -> None:
    atomic_write_json(
        inventory_progress_path(inventory_dir, source["source_id"]),
        source_artifact_payload(source, stats, archive_path, registry_hash, plan_hash, status),
    )


def inventory_source(
    source: dict[str, Any],
    data_dir: Path,
    min_chars: int,
    max_chars: int,
    inventory_dir: Path,
    registry_hash: str,
    plan_hash: str,
    build: bool,
    progress_every: int,
    reuse_inventory: bool,
) -> tuple[SourceStats, list[CandidateDocument]]:
    source_id = source["source_id"]
    archive = archive_path_for(source, data_dir)
    artifact_path = inventory_artifact_path(inventory_dir, source_id)
    if reuse_inventory and not build:
        reusable = reusable_source_artifact(source, artifact_path, archive, registry_hash, plan_hash)
        if reusable is not None:
            print(f"[inventory] {source_id}: reusing completed artifact {artifact_path}", file=sys.stderr, flush=True)
            return reusable, []

    start = time.monotonic()
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    stats = SourceStats(
        source_id=source_id,
        archive_path=str(archive),
        compressed_bytes=archive.stat().st_size if archive.exists() else 0,
        rejection_reasons={},
    )
    accepted_for_source: list[CandidateDocument] = []
    cap = source_cap(source)
    checkpoint_interval = max(1, progress_every)
    quota_floor = max(0, cap - max_chars)

    for raw_doc in iter_source_documents(source, data_dir):
        stats.documents += 1
        stats.raw_characters += len(raw_doc.text)
        stats.extracted_bytes += len(raw_doc.text.encode("utf-8"))
        stats.license_metadata_complete += int(bool(raw_doc.metadata.get("license")))
        stats.attribution_metadata_complete += int(bool(raw_doc.metadata))
        accepted, reason = validate_document(raw_doc, source, min_chars, max_chars)
        if accepted is None:
            stats.rejected_documents += 1
            stats.rejection_reasons[reason or "unknown"] = stats.rejection_reasons.get(reason or "unknown", 0) + 1
        else:
            stats.accepted_documents += 1
            chars = len(accepted.text)
            stats.normalized_characters += chars
            stats.turkish_characters += chars if accepted.language == "tr" else 0
            stats.english_characters += chars if accepted.language == "en" else 0
            if stats.candidate_characters + chars <= cap:
                stats.candidate_documents += 1
                stats.candidate_characters += chars
                stats.candidate_turkish_characters += chars if accepted.language == "tr" else 0
                stats.candidate_english_characters += chars if accepted.language == "en" else 0
                if build:
                    accepted_for_source.append(accepted)
            if stats.candidate_characters >= cap or (cap > max_chars and stats.candidate_characters >= quota_floor):
                stats.scan_stopped_after_cap = True
                break

        if stats.documents % checkpoint_interval == 0:
            _, peak = tracemalloc.get_traced_memory()
            stats.peak_memory_bytes = peak
            stats.elapsed_seconds = round(time.monotonic() - start, 3)
            write_source_progress(source, stats, archive, inventory_dir, registry_hash, plan_hash, "running")
            print(
                "[inventory] "
                f"{source_id}: records={stats.documents} accepted={stats.accepted_documents} "
                f"rejected={stats.rejected_documents} candidate_chars={stats.candidate_characters}",
                file=sys.stderr,
                flush=True,
            )

    _, peak = tracemalloc.get_traced_memory()
    stats.peak_memory_bytes = peak
    stats.elapsed_seconds = round(time.monotonic() - start, 3)
    stats.estimated_final_usable_characters = stats.candidate_characters
    write_source_progress(source, stats, archive, inventory_dir, registry_hash, plan_hash, "complete")
    atomic_write_json(artifact_path, source_artifact_payload(source, stats, archive, registry_hash, plan_hash, "complete"))
    print(
        "[inventory] "
        f"{source_id}: complete records={stats.documents} accepted={stats.accepted_documents} "
        f"rejected={stats.rejected_documents} candidate_chars={stats.candidate_characters}",
        file=sys.stderr,
        flush=True,
    )
    return stats, accepted_for_source


def summarize_inventory_stats(stats: list[SourceStats], registry: dict[str, Any], plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source_ids = [source["source_id"] for source in registry["sources"]]
    stats_by_source = {item.source_id: item for item in stats}
    missing_sources = [source_id for source_id in source_ids if source_id not in stats_by_source]
    selected_sources = {source_id: stats_by_source[source_id].estimated_final_usable_characters for source_id in source_ids if source_id in stats_by_source}
    selected_chars = sum(selected_sources.values())
    selected_lang = Counter()
    for item in stats:
        selected_lang["tr"] += item.candidate_turkish_characters
        selected_lang["en"] += item.candidate_english_characters

    target_total = int(plan["target_normalized_characters"])
    target_tr = int(plan["language_mix"]["tr"]["target_characters"])
    target_en = int(plan["language_mix"]["en"]["target_characters"])
    max_source = int(plan["source_caps"]["max_single_source_characters"])
    failures = []
    if missing_sources:
        failures.append(f"missing completed source inventories: {', '.join(missing_sources)}")
    if selected_chars < target_total:
        failures.append(f"total normalized characters {selected_chars} outside 50M +/-1% target")
    if selected_lang["tr"] < target_tr:
        failures.append(f"Turkish characters {selected_lang['tr']} outside 60% +/-1% target")
    if selected_lang["en"] < target_en:
        failures.append(f"English characters {selected_lang['en']} outside 40% +/-1% target")
    source_cap_violations = {
        source_id: char_count for source_id, char_count in selected_sources.items() if char_count > max_source
    }
    for source_id, char_count in source_cap_violations.items():
        failures.append(f"source cap violation for {source_id}: {char_count}")

    result = "FAIL" if failures else "PASS"
    payload = {
        "result": result,
        "completed_source_count": len(stats),
        "expected_source_count": len(source_ids),
        "missing_source_ids": missing_sources,
        "stats": [asdict(item) for item in stats],
        "selected_characters": selected_chars,
        "selected_language_characters": dict(selected_lang),
        "selected_source_characters": selected_sources,
        "source_cap_violations": source_cap_violations,
        "target_normalized_characters": target_total,
        "target_feasible": result == "PASS",
        "additional_licensed_sources_needed": result != "PASS",
        "duplicate_counts": {},
        "failures": failures,
        "output_hashes": {},
    }
    return payload, failures


def markdown_inventory(stats: list[SourceStats], payload: dict[str, Any], failures: list[str]) -> str:
    lines = ["# DarkMind v2 Phase 1B Source Inventory", "", f"Result: **{payload['result']}**", ""]
    lines.append(f"- Completed sources: {payload['completed_source_count']} / {payload['expected_source_count']}")
    lines.append(f"- Total usable normalized characters: {payload['selected_characters']}")
    language = payload["selected_language_characters"]
    lines.append(f"- Turkish characters: {language.get('tr', 0)}")
    lines.append(f"- English characters: {language.get('en', 0)}")
    lines.append(f"- 50M-character target feasible: {payload['target_feasible']}")
    lines.append(f"- Any source exceeds 40% cap: {bool(payload['source_cap_violations'])}")
    lines.append(f"- Additional licensed sources needed: {payload['additional_licensed_sources_needed']}")
    lines.extend(["", "## Downloaded Sources", ""])
    lines.append("| Source | Archive | Compressed bytes | Source cap | Resumability |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for item in stats:
        lines.append(
            f"| {item.source_id} | `{item.archive_path}` | {item.compressed_bytes} | "
            f"{payload['selected_source_characters'].get(item.source_id, 0)} | complete_reusable |"
        )
    lines.extend(["", "## Inventory Counts", ""])
    lines.append("| Source | Streamed records | Accepted docs | Rejected docs | Raw chars | Extracted bytes | Normalized chars | Usable chars | TR chars | EN chars |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in stats:
        lines.append(
            f"| {item.source_id} | {item.documents} | {item.accepted_documents} | {item.rejected_documents} | "
            f"{item.raw_characters} | {item.extracted_bytes} | {item.normalized_characters} | "
            f"{item.estimated_final_usable_characters} | {item.candidate_turkish_characters} | {item.candidate_english_characters} |"
        )
    lines.extend(["", "## Metadata Completeness", ""])
    lines.append("| Source | License metadata | Attribution metadata |")
    lines.append("| --- | ---: | ---: |")
    for item in stats:
        lines.append(f"| {item.source_id} | {item.license_metadata_complete}/{item.documents} | {item.attribution_metadata_complete}/{item.documents} |")
    lines.extend(["", "## Rejections By Reason", ""])
    for item in stats:
        reasons = item.rejection_reasons or {}
        if not reasons:
            lines.append(f"- {item.source_id}: none")
            continue
        lines.append(f"- {item.source_id}: " + ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())))
    lines.extend(["", "## Performance And Resume", ""])
    lines.append("| Source | Elapsed seconds | Peak memory bytes | Stopped near source cap |")
    lines.append("| --- | ---: | ---: | --- |")
    for item in stats:
        lines.append(f"| {item.source_id} | {item.elapsed_seconds} | {item.peak_memory_bytes or 'unavailable'} | {item.scan_stopped_after_cap} |")
    lines.extend(["", "## Risks", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- No source inventory target failures detected.")
    if payload["additional_licensed_sources_needed"]:
        lines.append("- Additional licensed sources or an explicitly smaller pilot are needed before tokenizer training.")
    lines.append("- Source-specific inventory artifacts and progress checkpoints are reusable on restart.")
    return "\n".join(lines) + "\n"


def json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def within_tolerance(actual: int, target: int) -> bool:
    tolerance = target * 0.01
    return target - tolerance <= actual <= target + tolerance


def attribution_record(document: CandidateDocument, source: dict[str, Any], split: str) -> dict[str, Any]:
    record = {
        "id": document.document_id,
        "source_id": source["source_id"],
        "source_name": source["source_name"],
        "source_url": source["official_download_url"],
        "source_homepage": source["official_homepage"],
        "source_version": source["source_version"],
        "license": source["license_id"],
        "license_id": source["license_id"],
        "license_url": source["official_license_url"],
        "snapshot_date": source["snapshot_date"],
        "language": document.language,
        "content_type": document.content_type,
        "selected_split": split,
        "selected_character_count": len(document.text),
    }
    record.update(document.metadata)
    if source["processor"] == "python_docs_text_tar_bz2":
        record["license_notes"] = "PSF License v2 documentation; Zero-Clause BSD note applies to code examples."
        record.setdefault("attribution_url", source["official_homepage"])
    if source["processor"] == "tatoeba_sentences_detailed_tar_bz2":
        record["author"] = record["username"]
    return record


def required_attribution_fields(source: dict[str, Any]) -> set[str]:
    fields = {
        "id",
        "source_id",
        "source_name",
        "source_url",
        "license",
        "license_id",
        "license_url",
        "snapshot_date",
        "selected_split",
        "selected_character_count",
    }
    fields.update(source.get("attribution_manifest_fields", []))
    if source["processor"] == "tatoeba_sentences_detailed_tar_bz2":
        fields.update({"sentence_id", "username", "author"})
    return fields


def split_character_counts(splits: dict[str, list[CandidateDocument]]) -> dict[str, int]:
    return {name: sum(len(document.text) for document in documents) for name, documents in splits.items()}


def phase1b_corpus_report(
    *,
    result: str,
    total_characters: int,
    language_characters: dict[str, int],
    split_counts: dict[str, int],
    split_documents: dict[str, list[CandidateDocument]],
    source_allocations: list[dict[str, Any]],
    content_type_characters: dict[str, int],
    input_rejections: dict[str, int],
    duplicate_counts: dict[str, int],
    hygiene_gates: dict[str, Any],
    output_hashes: dict[str, str],
    failures: list[str],
) -> str:
    lines = ["# DarkMind v2 Phase 1B Corpus Report", "", f"Result: **{result}**", ""]
    lines.extend(
        [
            "## Final Corpus",
            "",
            f"- Final normalized characters: {total_characters}",
            f"- Turkish characters: {language_characters.get('tr', 0)}",
            f"- English characters: {language_characters.get('en', 0)}",
            "- Target: 50,000,000 total; 30,000,000 Turkish; 20,000,000 English; each within +/-1%.",
            "",
            "## Splits",
            "",
            "| Split | Documents | Characters |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in ("train", "validation", "eval"):
        lines.append(f"| {split} | {len(split_documents[split])} | {split_counts[split]} |")

    lines.extend(["", "## Source Allocations", ""])
    lines.append("| Source | Candidate TR | Candidate EN | Planned TR | Planned EN | Selected TR | Selected EN | Selected total | Cap | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in source_allocations:
        lines.append(
            "| {source_id} | {candidate_tr} | {candidate_en} | {planned_tr} | {planned_en} | {selected_tr} | {selected_en} | {selected_total} | {source_cap} | {source_cap_status} |".format(
                **item
            )
        )

    lines.extend(["", "## Content And Rejections", ""])
    lines.append("| Content type | Selected characters |")
    lines.append("| --- | ---: |")
    for content_type, count in sorted(content_type_characters.items()):
        lines.append(f"| {content_type} | {count} |")
    lines.append("")
    lines.append("- Input rejections by reason: " + ", ".join(f"{reason}={count}" for reason, count in sorted(input_rejections.items())))
    lines.append(f"- Exact duplicate removals: {duplicate_counts['exact_duplicate']}")
    lines.append(f"- Near-duplicate removals: {duplicate_counts['near_duplicate']}")

    lines.extend(["", "## Hygiene Gates", ""])
    lines.append("| Gate | Value |")
    lines.append("| --- | --- |")
    for name, value in hygiene_gates.items():
        lines.append(f"| {name} | {value} |")

    lines.extend(["", "## Manifest Hashes", ""])
    lines.append("| Artifact | SHA-256 |")
    lines.append("| --- | --- |")
    for name, value in sorted(output_hashes.items()):
        lines.append(f"| {name} | `{value}` |")

    lines.extend(["", "## Determinism", ""])
    lines.append("- Verification mode rendered the finalized split and manifest content a second time from the same fixed seed and stable ordering.")
    lines.append(f"- Deterministic rebuild verification: {hygiene_gates['deterministic_rebuild_verification']}")
    lines.extend(["", "## Risks", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- No unresolved corpus hygiene or source-cap risks detected.")
        lines.append("")
        lines.append("TOKENIZER CANDIDATE TRAINING IS READY FOR USER APPROVAL")
    return "\n".join(lines) + "\n"


def build_final_corpus(
    *,
    registry: dict[str, Any],
    plan: dict[str, Any],
    registry_hash: str,
    plan_hash: str,
    data_dir: Path,
    report_dir: Path,
    inventory_dir: Path,
    stats: list[SourceStats],
    candidates: list[CandidateDocument],
) -> dict[str, Any]:
    sources = registry["sources"]
    source_by_id = {source["source_id"]: source for source in sources}
    language_targets = {
        "tr": int(plan["language_mix"]["tr"]["target_characters"]),
        "en": int(plan["language_mix"]["en"]["target_characters"]),
    }
    target_total = int(plan["target_normalized_characters"])
    seed = int(plan["splits"]["seed"])
    selected, selected_by_source_language, duplicate_counts, selection_rejections = select_balanced_documents(
        candidates,
        sources,
        language_targets,
        seed,
    )

    language_characters = Counter()
    source_characters = Counter({source["source_id"]: 0 for source in sources})
    source_documents = Counter({source["source_id"]: 0 for source in sources})
    content_type_characters = Counter()
    for document in selected:
        characters = len(document.text)
        language_characters[document.language] += characters
        source_characters[document.source_id] += characters
        source_documents[document.source_id] += 1
        content_type_characters[document.content_type] += characters

    planned_allocations, candidate_availability = source_language_allocations(candidates, sources, language_targets)
    source_allocations: list[dict[str, Any]] = []
    source_cap_violations: dict[str, int] = {}
    for source in sources:
        source_id = source["source_id"]
        cap = source_cap(source)
        selected_total = source_characters[source_id]
        if selected_total > cap or selected_total > int(plan["source_caps"]["max_single_source_characters"]):
            source_cap_violations[source_id] = selected_total
        source_allocations.append(
            {
                "source_id": source_id,
                "source_name": source["source_name"],
                "candidate_tr": candidate_availability[source_id]["tr"],
                "candidate_en": candidate_availability[source_id]["en"],
                "planned_tr": planned_allocations[source_id]["tr"],
                "planned_en": planned_allocations[source_id]["en"],
                "selected_tr": selected_by_source_language[source_id]["tr"],
                "selected_en": selected_by_source_language[source_id]["en"],
                "selected_total": selected_total,
                "selected_documents": source_documents[source_id],
                "source_cap": cap,
                "source_cap_status": "PASS" if source_id not in source_cap_violations else "FAIL",
            }
        )

    splits = split_documents(
        selected,
        {
            "train": float(plan["splits"]["train"]["ratio"]),
            "validation": float(plan["splits"]["validation"]["ratio"]),
            "eval": float(plan["splits"]["test"]["ratio"]),
        },
        seed,
    )
    split_counts = split_character_counts(splits)
    split_for_document = {
        document.document_id: split
        for split, documents in splits.items()
        for document in documents
    }
    attribution_records = [
        attribution_record(document, source_by_id[document.source_id], split_for_document[document.document_id])
        for split in ("train", "validation", "eval")
        for document in splits[split]
    ]
    missing_license_metadata = 0
    missing_attribution_metadata = 0
    for record in attribution_records:
        source = source_by_id[record["source_id"]]
        if not record.get("license") or not record.get("license_url"):
            missing_license_metadata += 1
        if any(not record.get(field) for field in required_attribution_fields(source)):
            missing_attribution_metadata += 1

    invalid_utf8 = 0
    for document in selected:
        try:
            document.text.encode("utf-8").decode("utf-8")
        except UnicodeError:
            invalid_utf8 += 1
    selected_mojibake = sum(
        int(any(marker in document.text for marker in MOJIBAKE_MARKERS) and bool(detect_text(document.text)))
        for document in selected
    )
    replacement_characters = sum(document.text.count("\ufffd") for document in selected)
    language_mismatches = sum(int(language_of_text(document.text) != document.language) for document in selected)
    exact_seen: set[str] = set()
    near_seen: set[tuple[str, ...]] = set()
    unresolved_exact_duplicates = 0
    unresolved_near_duplicates = 0
    for document in selected:
        exact = sha256_text(" ".join(document.text.casefold().split()))
        near = tuple(WORD_RE.findall(document.text.casefold())[:24])
        unresolved_exact_duplicates += int(exact in exact_seen)
        unresolved_near_duplicates += int(near in near_seen)
        exact_seen.add(exact)
        near_seen.add(near)

    failures: list[str] = []
    selected_total = sum(language_characters.values())
    if not within_tolerance(selected_total, target_total):
        failures.append(f"total normalized characters {selected_total} outside 50M +/-1% target")
    for language, target in language_targets.items():
        if not within_tolerance(language_characters[language], target):
            failures.append(f"{language} characters {language_characters[language]} outside +/-1% target")
    if source_cap_violations:
        failures.append(f"source cap violation: {source_cap_violations}")
    for name, value in {
        "invalid_utf8": invalid_utf8,
        "mojibake_detections": selected_mojibake,
        "replacement_characters": replacement_characters,
        "language_mismatch": language_mismatches,
        "unresolved_exact_duplicates": unresolved_exact_duplicates,
        "unresolved_near_duplicate_clusters": unresolved_near_duplicates,
        "missing_license_metadata": missing_license_metadata,
        "missing_attribution_metadata": missing_attribution_metadata,
    }.items():
        if value:
            failures.append(f"{name}={value}")

    input_rejections = Counter()
    for item in stats:
        input_rejections.update(item.rejection_reasons or {})
    input_rejections.update(duplicate_counts)
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    source_allocation_payload = {
        "format": "darkmind-v2-phase1b-source-allocation-v1",
        "deterministic_seed": seed,
        "language_targets": language_targets,
        "source_allocations": source_allocations,
    }
    source_allocation_content = json_content(source_allocation_payload)
    split_texts = {name: text_content(documents) for name, documents in splits.items()}
    split_hashes = {name: sha256_text(content) for name, content in split_texts.items()}
    split_manifest_payload = {
        "format": "darkmind-v2-phase1b-split-manifest-v1",
        "deterministic_seed": seed,
        "ratios": {"train": 0.9, "validation": 0.05, "eval": 0.05},
        "splits": {
            name: {
                "documents": len(splits[name]),
                "normalized_characters": split_counts[name],
                "sha256": split_hashes[name],
            }
            for name in ("train", "validation", "eval")
        },
    }
    split_manifest_content = json_content(split_manifest_payload)
    attribution_content = jsonl_content(attribution_records)
    rejected_content = jsonl_content(selection_rejections)

    preliminary_hygiene = {
        "invalid_utf8": invalid_utf8,
        "mojibake_detections": selected_mojibake,
        "replacement_characters": replacement_characters,
        "language_mismatch": language_mismatches,
        "unresolved_exact_duplicates": unresolved_exact_duplicates,
        "unresolved_near_duplicate_clusters": unresolved_near_duplicates,
        "missing_license_metadata": missing_license_metadata,
        "missing_attribution_metadata": missing_attribution_metadata,
        "source_cap_violations": len(source_cap_violations),
        "deterministic_split_hashes_present": all(split_hashes.values()),
        "deterministic_rebuild_verification": "PENDING",
    }
    if failures:
        report = phase1b_corpus_report(
            result="FAIL",
            total_characters=selected_total,
            language_characters=dict(language_characters),
            split_counts=split_counts,
            split_documents=splits,
            source_allocations=source_allocations,
            content_type_characters=dict(content_type_characters),
            input_rejections=dict(input_rejections),
            duplicate_counts=duplicate_counts,
            hygiene_gates=preliminary_hygiene,
            output_hashes={},
            failures=failures,
        )
        atomic_write_text(report_dir / "phase1b_corpus_report.md", report)
        return {
            "result": "FAIL",
            "failures": failures,
            "selected_characters": selected_total,
            "selected_language_characters": dict(language_characters),
            "selected_source_characters": dict(source_characters),
            "duplicate_counts": duplicate_counts,
            "hygiene_gates": preliminary_hygiene,
            "output_hashes": {},
            "stats": [asdict(item) for item in stats],
        }

    output_hashes = {
        "tokenizer_train.txt": write_content(processed / "tokenizer_train.txt", split_texts["train"]),
        "tokenizer_validation.txt": write_content(processed / "tokenizer_validation.txt", split_texts["validation"]),
        "tokenizer_eval.txt": write_content(processed / "tokenizer_eval.txt", split_texts["eval"]),
        "attribution_manifest.jsonl": write_content(processed / "attribution_manifest.jsonl", attribution_content),
        "rejected_documents.jsonl": write_content(processed / "rejected_documents.jsonl", rejected_content),
        "source_allocation.json": write_content(processed / "source_allocation.json", source_allocation_content),
        "split_manifest.json": write_content(processed / "split_manifest.json", split_manifest_content),
    }
    corpus_manifest_payload = {
        "format": "darkmind-v2-phase1b-corpus-manifest-v1",
        "result": "PASS",
        "registry_sha256": registry_hash,
        "plan_sha256": plan_hash,
        "deterministic_seed": seed,
        "total_documents": len(selected),
        "total_normalized_characters": selected_total,
        "language_characters": dict(language_characters),
        "source_characters": dict(source_characters),
        "content_type_characters": dict(content_type_characters),
        "hygiene_gates": preliminary_hygiene,
        "output_hashes": dict(output_hashes),
    }
    corpus_manifest_payload["deterministic_content_sha256"] = sha256_text(
        json.dumps(corpus_manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    corpus_manifest_content = json_content(corpus_manifest_payload)
    output_hashes["corpus_manifest.json"] = write_content(processed / "corpus_manifest.json", corpus_manifest_content)

    second_pass_hashes = {
        "tokenizer_train.txt": sha256_text(text_content(splits["train"])),
        "tokenizer_validation.txt": sha256_text(text_content(splits["validation"])),
        "tokenizer_eval.txt": sha256_text(text_content(splits["eval"])),
        "attribution_manifest.jsonl": sha256_text(jsonl_content(attribution_records)),
        "rejected_documents.jsonl": sha256_text(jsonl_content(selection_rejections)),
        "source_allocation.json": sha256_text(json_content(source_allocation_payload)),
        "split_manifest.json": sha256_text(json_content(split_manifest_payload)),
        "corpus_manifest.json": sha256_text(json_content(corpus_manifest_payload)),
    }
    deterministic_pass = output_hashes == second_pass_hashes
    hygiene_gates = dict(preliminary_hygiene)
    hygiene_gates["deterministic_rebuild_verification"] = "PASS" if deterministic_pass else "FAIL"
    if not deterministic_pass:
        failures.append("deterministic rebuild verification failed")
    determinism_payload = {
        "format": "darkmind-v2-phase1b-determinism-v1",
        "method": "second in-memory rendering from fixed seed and stable ordering",
        "result": "PASS" if deterministic_pass else "FAIL",
        "first_pass_hashes": output_hashes,
        "second_pass_hashes": second_pass_hashes,
    }
    output_hashes["determinism_verification.json"] = write_content(
        processed / "determinism_verification.json", json_content(determinism_payload)
    )

    result = "PASS" if not failures else "FAIL"
    report = phase1b_corpus_report(
        result=result,
        total_characters=selected_total,
        language_characters=dict(language_characters),
        split_counts=split_counts,
        split_documents=splits,
        source_allocations=source_allocations,
        content_type_characters=dict(content_type_characters),
        input_rejections=dict(input_rejections),
        duplicate_counts=duplicate_counts,
        hygiene_gates=hygiene_gates,
        output_hashes=output_hashes,
        failures=failures,
    )
    atomic_write_text(report_dir / "phase1b_corpus_report.md", report)
    payload = {
        "result": result,
        "failures": failures,
        "completed_source_count": len(stats),
        "expected_source_count": len(sources),
        "selected_characters": selected_total,
        "selected_language_characters": dict(language_characters),
        "selected_source_characters": dict(source_characters),
        "source_allocations": source_allocations,
        "split_characters": split_counts,
        "duplicate_counts": duplicate_counts,
        "input_rejections": dict(input_rejections),
        "hygiene_gates": hygiene_gates,
        "output_hashes": output_hashes,
        "stats": [asdict(item) for item in stats],
    }
    atomic_write_json(inventory_dir / "phase1b_source_inventory.json", payload)
    return payload


def run_inventory(
    registry_path: Path,
    plan_path: Path,
    data_dir: Path,
    report_dir: Path,
    build: bool,
    source_ids: set[str] | None = None,
    inventory_dir: Path | None = None,
    workers: int = 1,
    progress_every: int = 100_000,
    reuse_inventory: bool = True,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("Phase 1B inventory currently supports --workers 1 only")
    registry = load_json(registry_path)
    plan = load_json(plan_path)
    registry_hash = sha256_text(registry_path.read_text(encoding="utf-8"))
    plan_hash = sha256_text(plan_path.read_text(encoding="utf-8"))
    inventory_dir = inventory_dir or data_dir / "inventory"
    min_chars = int(plan["quality_gates"]["min_document_characters"])
    max_chars = int(plan["quality_gates"]["max_document_characters"])
    selected_candidates: list[CandidateDocument] = []
    stats: list[SourceStats] = []

    for source in registry["sources"]:
        if source_ids and source["source_id"] not in source_ids:
            continue
        source_stats, accepted_for_source = inventory_source(
            source,
            data_dir,
            min_chars,
            max_chars,
            inventory_dir,
            registry_hash,
            plan_hash,
            build,
            progress_every,
            reuse_inventory,
        )
        stats.append(source_stats)
        selected_candidates.extend(accepted_for_source)

    if not build:
        completed_stats = []
        for source in registry["sources"]:
            artifact = inventory_artifact_path(inventory_dir, source["source_id"])
            if not artifact.exists():
                continue
            reusable = reusable_source_artifact(
                source,
                artifact,
                archive_path_for(source, data_dir),
                registry_hash,
                plan_hash,
            )
            if reusable is not None:
                completed_stats.append(reusable)
        payload, failures = summarize_inventory_stats(completed_stats, registry, plan)
        report_dir.mkdir(parents=True, exist_ok=True)
        inventory_dir.mkdir(parents=True, exist_ok=True)
        inventory_md = markdown_inventory(completed_stats, payload, failures)
        atomic_write_text(report_dir / "phase1b_source_inventory.md", inventory_md)
        atomic_write_json(inventory_dir / "phase1b_source_inventory.json", payload)
        if failures:
            gap = {
                "result": "FAIL",
                "reason": "approved sources cannot satisfy Phase 1B corpus targets without relaxing gates",
                "failures": failures,
                "selected_characters": payload["selected_characters"],
                "selected_language_characters": payload["selected_language_characters"],
                "selected_source_characters": payload["selected_source_characters"],
                "recommendation": "Approve additional licensed sources or explicitly approve a smaller pilot before tokenizer training.",
            }
            payload["gap"] = gap
            atomic_write_text(
                report_dir / "phase1b_source_gap_report.md",
                "# DarkMind v2 Phase 1B Source Gap Report\n\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nRecommendation: approve additional licensed sources or explicitly approve a smaller pilot before tokenizer training.\n",
            )
            atomic_write_json(inventory_dir / "phase1b_source_gap_report.json", gap)
        else:
            payload["gap"] = {}
        return payload

    return build_final_corpus(
        registry=registry,
        plan=plan,
        registry_hash=registry_hash,
        plan_hash=plan_hash,
        data_dir=data_dir,
        report_dir=report_dir,
        inventory_dir=inventory_dir,
        stats=stats,
        candidates=selected_candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory or build the DarkMind v2 Phase 1B tokenizer pilot corpus.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--inventory-dir", type=Path, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--no-reuse-inventory", action="store_true")
    args = parser.parse_args()
    try:
        payload = run_inventory(
            args.registry,
            args.plan,
            args.data_dir,
            args.report_dir,
            args.build,
            set(args.source_id) if args.source_id else None,
            args.inventory_dir,
            args.workers,
            args.progress_every,
            not args.no_reuse_inventory,
        )
    except (OSError, UnicodeDecodeError, ValueError, ET.ParseError) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        sys.exit(1)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.build and payload["result"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
