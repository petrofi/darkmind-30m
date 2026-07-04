"""Deterministic exact and near-duplicate handling for DarkMind v2 corpus text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+", re.UNICODE)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RejectedDocument:
    document_id: str
    reason: str
    representative_id: str
    similarity: float


def normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(text.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_hash(text: str) -> str:
    return sha256_text(normalize_for_hash(text))


def paragraph_hashes(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    return [exact_hash(paragraph) for paragraph in paragraphs]


def token_shingles(text: str, *, shingle_size: int = 5) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in WORD_RE.findall(unicodedata.normalize("NFC", text))]
    if not tokens:
        return set()
    if len(tokens) <= shingle_size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + shingle_size]) for index in range(len(tokens) - shingle_size + 1)}


def jaccard_similarity(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def read_documents(path: Path) -> list[Document]:
    if path.suffix.lower() == ".jsonl":
        documents: list[Document] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text") or record.get("content") or "")
            document_id = str(record.get("id") or f"{path.name}:{index}")
            metadata = {key: value for key, value in record.items() if key not in {"text", "content"}}
            documents.append(Document(document_id, text, metadata))
        return documents

    text = path.read_text(encoding="utf-8")
    documents = []
    for index, chunk in enumerate(re.split(r"\n\s*\n", text), start=1):
        if chunk.strip():
            documents.append(Document(f"{path.name}:{index}", chunk.strip(), {"source": str(path)}))
    return documents


def deduplicate_documents(
    documents: list[Document],
    *,
    threshold: float = 0.85,
    shingle_size: int = 5,
) -> tuple[list[Document], list[RejectedDocument], dict[str, str]]:
    accepted: list[Document] = []
    rejected: list[RejectedDocument] = []
    mapping: dict[str, str] = {}
    exact_seen: dict[str, str] = {}
    shingles_by_id: dict[str, set[tuple[str, ...]]] = {}

    for document in sorted(documents, key=lambda item: item.document_id):
        doc_hash = exact_hash(document.text)
        if doc_hash in exact_seen:
            representative_id = exact_seen[doc_hash]
            rejected.append(RejectedDocument(document.document_id, "exact_duplicate", representative_id, 1.0))
            mapping[document.document_id] = representative_id
            continue

        document_shingles = token_shingles(document.text, shingle_size=shingle_size)
        duplicate: tuple[str, float] | None = None
        for accepted_document in accepted:
            similarity = jaccard_similarity(document_shingles, shingles_by_id[accepted_document.document_id])
            if similarity >= threshold:
                duplicate = (accepted_document.document_id, similarity)
                break

        if duplicate is not None:
            representative_id, similarity = duplicate
            rejected.append(RejectedDocument(document.document_id, "near_duplicate", representative_id, similarity))
            mapping[document.document_id] = representative_id
            continue

        accepted.append(document)
        exact_seen[doc_hash] = document.document_id
        shingles_by_id[document.document_id] = document_shingles
        mapping[document.document_id] = document.document_id

    return accepted, rejected, mapping


def write_jsonl(path: Path, documents: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            payload = {"id": document.document_id, "text": document.text, **document.metadata}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_deduplication(
    input_path: Path,
    *,
    accepted_out: Path,
    rejected_out: Path,
    report_out: Path,
    mapping_out: Path,
    threshold: float = 0.85,
    shingle_size: int = 5,
) -> dict[str, Any]:
    documents = read_documents(input_path)
    accepted, rejected, mapping = deduplicate_documents(documents, threshold=threshold, shingle_size=shingle_size)
    write_jsonl(accepted_out, accepted)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out.write_text(
        "\n".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
        newline="\n",
    )
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "input_path": str(input_path),
        "total_documents": len(documents),
        "accepted_documents": len(accepted),
        "rejected_documents": len(rejected),
        "exact_duplicates": sum(1 for item in rejected if item.reason == "exact_duplicate"),
        "near_duplicates": sum(1 for item in rejected if item.reason == "near_duplicate"),
        "threshold": threshold,
        "shingle_size": shingle_size,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate DarkMind v2 corpus text without deleting source data.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--accepted-out", type=Path, required=True)
    parser.add_argument("--rejected-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--mapping-out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--shingle-size", type=int, default=5)
    args = parser.parse_args()

    report = run_deduplication(
        args.input,
        accepted_out=args.accepted_out,
        rejected_out=args.rejected_out,
        report_out=args.report_out,
        mapping_out=args.mapping_out,
        threshold=args.threshold,
        shingle_size=args.shingle_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
