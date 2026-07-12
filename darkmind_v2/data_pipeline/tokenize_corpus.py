"""Stream a controlled JSONL corpus into deterministic uint16 token shards."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from darkmind_v2.data_pipeline.tokenized_manifest import (
    MANIFEST_SCHEMA,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_hash,
    sha256_bytes,
    sha256_file,
)
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer


SPLITS = ("train", "validation", "eval")
DEFAULT_OUTPUT = Path("darkmind_v2/data/phase2a/tokenized")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} must be an object")
            yield line_number, record


def _encode_uint16_le(token_ids: list[int]) -> bytes:
    if any(token_id < 0 or token_id > 65535 for token_id in token_ids):
        raise ValueError("token ID cannot be represented as uint16")
    return struct.pack(f"<{len(token_ids)}H", *token_ids)


def tokenize_corpus(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    tokenizer: FrozenTokenizer | None = None,
    add_bos: bool = False,
    max_tokens_per_shard: int = 1_000_000,
) -> dict[str, Any]:
    if max_tokens_per_shard <= 0:
        raise ValueError("max_tokens_per_shard must be positive")
    tokenizer = tokenizer or FrozenTokenizer()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_text_hashes: dict[str, str] = {}

    for line_number, record in iter_jsonl(input_path):
        document_id = str(record.get("id", "")).strip()
        split = str(record.get("split", "")).strip()
        language = str(record.get("language", "")).strip()
        text = record.get("text")
        reason = None
        if not document_id:
            reason = "missing_id"
        elif split not in SPLITS:
            reason = "invalid_split"
        elif language not in {"tr", "en"}:
            reason = "invalid_language"
        elif not isinstance(text, str) or not text.strip():
            reason = "empty_text"
        elif document_id in seen_ids:
            reason = "duplicate_id"
        text_hash = sha256_bytes(text.encode("utf-8")) if isinstance(text, str) else ""
        if reason is None and text_hash in seen_text_hashes and seen_text_hashes[text_hash] != split:
            reason = "cross_split_duplicate_text"
        if reason:
            rejected.append({"line_number": line_number, "id": document_id, "reason": reason})
            continue

        token_ids = tokenizer.encode_document(text, add_bos=add_bos)
        if len(token_ids) > max_tokens_per_shard:
            rejected.append({"line_number": line_number, "id": document_id, "reason": "document_exceeds_shard_cap"})
            continue
        seen_ids[document_id] = split
        seen_text_hashes[text_hash] = split
        accepted.append(
            {
                "id": document_id,
                "split": split,
                "language": language,
                "text_hash": text_hash,
                "characters": len(text),
                "token_ids": token_ids,
                "source_order": line_number,
            }
        )

    accepted.sort(key=lambda item: (SPLITS.index(item["split"]), item["source_order"], item["id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    document_records: list[dict[str, Any]] = []
    language_tokens: Counter[str] = Counter()
    language_characters: Counter[str] = Counter()

    for split in SPLITS:
        split_documents = [item for item in accepted if item["split"] == split]
        shard_index = 0
        pending_tokens: list[int] = []
        pending_documents: list[dict[str, Any]] = []

        def flush_shard() -> None:
            nonlocal shard_index, pending_tokens, pending_documents
            if not pending_tokens:
                return
            filename = f"{split}-{shard_index:05d}.bin"
            payload = _encode_uint16_le(pending_tokens)
            atomic_write_bytes(output_dir / filename, payload)
            checksum = sha256_bytes(payload)
            shard_records.append(
                {
                    "split": split,
                    "filename": filename,
                    "sha256": checksum,
                    "tokens": len(pending_tokens),
                    "bytes": len(payload),
                    "document_ids": [item["id"] for item in pending_documents],
                }
            )
            for item in pending_documents:
                document_records.append({**item, "shard": filename})
            shard_index += 1
            pending_tokens = []
            pending_documents = []

        for document in split_documents:
            token_ids = document.pop("token_ids")
            if pending_tokens and len(pending_tokens) + len(token_ids) > max_tokens_per_shard:
                flush_shard()
            start_offset = len(pending_tokens)
            pending_tokens.extend(token_ids)
            end_offset = len(pending_tokens)
            pending_documents.append(
                {
                    "id": document["id"],
                    "split": split,
                    "language": document["language"],
                    "text_sha256": document["text_hash"],
                    "characters": document["characters"],
                    "tokens": len(token_ids),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }
            )
            language_tokens[document["language"]] += len(token_ids)
            language_characters[document["language"]] += document["characters"]
        flush_shard()

    checksums = {record["filename"]: record["sha256"] for record in shard_records}
    statistics = {
        "accepted_documents": len(accepted),
        "rejected_documents": len(rejected),
        "total_tokens": sum(record["tokens"] for record in shard_records),
        "total_bytes": sum(record["bytes"] for record in shard_records),
        "language_tokens": dict(sorted(language_tokens.items())),
        "language_characters": dict(sorted(language_characters.items())),
        "document_boundary_eos_tokens": len(accepted),
        "bos_tokens_added": len(accepted) if add_bos else 0,
    }
    manifest_core = {
        "schema_version": MANIFEST_SCHEMA,
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "add_bos": add_bos,
        "eos_token_id": tokenizer.eos_token_id,
        "source": {
            "path": input_path.name,
            "sha256": sha256_file(input_path),
            "ordering": "split order train/validation/eval, then source manifest order",
        },
        "tokenizer": {
            "name": tokenizer.manifest["tokenizer_name"],
            "model_sha256": EXPECTED_HASHES["tokenizer.model"],
            "vocab_sha256": EXPECTED_HASHES["tokenizer.vocab"],
            "freeze_manifest_sha256": EXPECTED_HASHES["tokenizer_freeze_manifest.json"],
        },
        "shards": shard_records,
        "documents": document_records,
        "statistics": statistics,
    }
    manifest = {**manifest_core, "deterministic_content_hash": canonical_json_hash(manifest_core)}
    atomic_write_json(output_dir / "tokenized_corpus_manifest.json", manifest)
    atomic_write_json(output_dir / "shard_checksums.json", checksums)
    atomic_write_json(output_dir / "tokenization_statistics.json", statistics)
    atomic_write_jsonl(output_dir / "rejected_records.jsonl", rejected)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--max-tokens-per-shard", type=int, default=1_000_000)
    args = parser.parse_args()
    manifest = tokenize_corpus(
        args.input,
        args.output_dir,
        add_bos=args.add_bos,
        max_tokens_per_shard=args.max_tokens_per_shard,
    )
    print(json.dumps({"result": "PASS", "statistics": manifest["statistics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
