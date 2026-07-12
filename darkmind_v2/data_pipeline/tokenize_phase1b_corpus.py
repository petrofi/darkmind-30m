"""Stream the validated Phase 1B corpus into deterministic uint16 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from darkmind_v2.data_pipeline.tokenized_manifest import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_hash,
    sha256_file,
)
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer


SPLITS = ("train", "validation", "eval")
SPLIT_FILES = {
    "train": "tokenizer_train.txt",
    "validation": "tokenizer_validation.txt",
    "eval": "tokenizer_eval.txt",
}
DEFAULT_PROCESSED_DIR = Path("darkmind_v2/data/phase1b/processed")
DEFAULT_OUTPUT_DIR = Path("darkmind_v2/data/phase2a/tokenized/full_v1")


def iter_split_documents(path: Path) -> Iterator[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.removesuffix("\n").removesuffix("\r")
            if line:
                lines.append(line)
            elif lines:
                yield "\n".join(lines)
                lines = []
    if lines:
        yield "\n".join(lines)


def iter_attribution(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"attribution record {line_number} is not an object")
            yield record


def uint16_bytes(token_ids: list[int]) -> bytes:
    values = array("H", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


class ShardWriter:
    def __init__(self, output_dir: Path, split: str, token_cap: int) -> None:
        self.output_dir = output_dir
        self.split = split
        self.token_cap = token_cap
        self.index = 0
        self.handle = None
        self.hasher = None
        self.filename = ""
        self.tokens = 0
        self.documents = 0
        self.records: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.filename = f"{self.split}-{self.index:05d}.bin"
        self.handle = (self.output_dir / f".{self.filename}.tmp").open("xb")
        self.hasher = hashlib.sha256()
        self.tokens = 0
        self.documents = 0

    def add(self, token_ids: list[int]) -> tuple[str, int, int]:
        if not token_ids:
            raise ValueError("cannot write an empty tokenized document")
        if self.handle is None:
            self._open()
        if self.tokens and self.tokens + len(token_ids) > self.token_cap:
            self.close_current()
            self._open()
        start = self.tokens
        payload = uint16_bytes(token_ids)
        self.handle.write(payload)
        self.hasher.update(payload)
        self.tokens += len(token_ids)
        self.documents += 1
        return self.filename, start, self.tokens

    def close_current(self) -> None:
        if self.handle is None:
            return
        temporary = Path(self.handle.name)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        final = self.output_dir / self.filename
        os.replace(temporary, final)
        self.records.append(
            {
                "split": self.split,
                "filename": self.filename,
                "sha256": self.hasher.hexdigest(),
                "tokens": self.tokens,
                "bytes": self.tokens * 2,
                "documents": self.documents,
            }
        )
        self.index += 1
        self.handle = None
        self.hasher = None

    def close(self) -> list[dict[str, Any]]:
        self.close_current()
        return self.records


def tokenize_phase1b_corpus(
    processed_dir: Path,
    output_dir: Path,
    *,
    shard_token_cap: int = 2_000_000,
) -> dict[str, Any]:
    if shard_token_cap <= 0:
        raise ValueError("shard_token_cap must be positive")
    required = [
        *SPLIT_FILES.values(),
        "corpus_manifest.json",
        "split_manifest.json",
        "attribution_manifest.jsonl",
    ]
    missing = [name for name in required if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 1B processed inputs: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite tokenized output: {output_dir}")
    incomplete = output_dir.with_name(f".{output_dir.name}.incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete output requires manual inspection: {incomplete}")
    incomplete.mkdir(parents=True)

    tokenizer = FrozenTokenizer()
    attribution = iter_attribution(processed_dir / "attribution_manifest.jsonl")
    boundary_path = incomplete / "document_boundaries.jsonl"
    rejected_path = incomplete / "rejected_records.jsonl"
    boundary_tmp = boundary_path.with_name(f".{boundary_path.name}.tmp")
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    language_tokens: Counter[str] = Counter()
    language_characters: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_documents: Counter[str] = Counter()
    split_characters: Counter[str] = Counter()
    unknown_tokens = 0
    shard_records: list[dict[str, Any]] = []
    boundaries_hasher = hashlib.sha256()

    with boundary_tmp.open("xb") as boundaries:
        for split in SPLITS:
            writer = ShardWriter(incomplete, split, shard_token_cap)
            for source_order, text in enumerate(iter_split_documents(processed_dir / SPLIT_FILES[split]), start=1):
                try:
                    record = next(attribution)
                except StopIteration as exc:
                    raise ValueError(f"attribution ended before {split} document {source_order}") from exc
                document_id = str(record.get("id", ""))
                language = str(record.get("language", ""))
                if record.get("selected_split") != split:
                    raise ValueError(f"attribution split mismatch for {document_id}")
                if int(record.get("selected_character_count", -1)) != len(text):
                    raise ValueError(f"attribution character count mismatch for {document_id}")
                if not document_id or document_id in seen_ids:
                    raise ValueError(f"duplicate or empty document ID: {document_id!r}")
                if language not in {"tr", "en"}:
                    raise ValueError(f"invalid document language for {document_id}: {language!r}")
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_hash in seen_text_hashes:
                    raise ValueError(f"cross-split or unresolved duplicate text: {document_id}")
                token_ids = tokenizer.encode_document(text)
                if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in token_ids):
                    raise ValueError(f"token outside vocabulary for {document_id}")
                if token_ids[-1] != tokenizer.eos_token_id:
                    raise ValueError(f"document lacks EOS for {document_id}")
                unknown_tokens += token_ids.count(tokenizer.unk_token_id)
                filename, start, end = writer.add(token_ids)
                boundary = {
                    "id": document_id,
                    "split": split,
                    "language": language,
                    "source_order": source_order,
                    "text_sha256": text_hash,
                    "characters": len(text),
                    "tokens": len(token_ids),
                    "shard": filename,
                    "start_offset": start,
                    "end_offset": end,
                }
                encoded = json.dumps(boundary, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
                boundaries.write(encoded)
                boundaries_hasher.update(encoded)
                seen_ids.add(document_id)
                seen_text_hashes.add(text_hash)
                language_tokens[language] += len(token_ids)
                language_characters[language] += len(text)
                split_tokens[split] += len(token_ids)
                split_documents[split] += 1
                split_characters[split] += len(text)
                if source_order % 25_000 == 0:
                    print(
                        f"progress split={split} documents={source_order} "
                        f"tokens={split_tokens[split]}",
                        flush=True,
                    )
            shard_records.extend(writer.close())
    os.replace(boundary_tmp, boundary_path)

    try:
        extra = next(attribution)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(f"unconsumed attribution record: {extra.get('id')}")
    if unknown_tokens:
        raise ValueError(f"frozen tokenizer emitted {unknown_tokens} unknown tokens")

    split_manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    expected_documents = {
        split: int(split_manifest["splits"][split]["documents"])
        for split in SPLITS
    }
    if dict(split_documents) != expected_documents:
        raise ValueError(f"split document counts changed: actual={dict(split_documents)} expected={expected_documents}")

    atomic_write_jsonl(rejected_path, rejected)
    statistics = {
        "accepted_documents": sum(split_documents.values()),
        "rejected_documents": len(rejected),
        "total_tokens": sum(split_tokens.values()),
        "total_bytes": sum(item["bytes"] for item in shard_records),
        "split_tokens": dict(split_tokens),
        "split_documents": dict(split_documents),
        "split_characters": dict(split_characters),
        "language_tokens": dict(language_tokens),
        "language_characters": dict(language_characters),
        "document_boundary_eos_tokens": sum(split_documents.values()),
        "unknown_tokens": unknown_tokens,
    }
    checksums = {record["filename"]: record["sha256"] for record in shard_records}
    source_files = {
        filename: {"sha256": sha256_file(processed_dir / filename), "bytes": (processed_dir / filename).stat().st_size}
        for filename in required
    }
    manifest_core = {
        "schema_version": "darkmind-v2-tokenized-corpus-v2",
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_added": False,
        "ordering": "Phase 1B split order train/validation/eval and source order within each split",
        "shard_token_cap": shard_token_cap,
        "source": {
            "processed_directory": "darkmind_v2/data/phase1b/processed",
            "files": source_files,
            "corpus_manifest_deterministic_hash": json.loads(
                (processed_dir / "corpus_manifest.json").read_text(encoding="utf-8")
            )["deterministic_content_sha256"],
        },
        "tokenizer": {
            "name": tokenizer.manifest["tokenizer_name"],
            "model_sha256": EXPECTED_HASHES["tokenizer.model"],
            "vocab_sha256": EXPECTED_HASHES["tokenizer.vocab"],
            "freeze_manifest_sha256": EXPECTED_HASHES["tokenizer_freeze_manifest.json"],
        },
        "document_boundaries": {
            "filename": boundary_path.name,
            "sha256": boundaries_hasher.hexdigest(),
            "records": statistics["accepted_documents"],
        },
        "rejected_records": {
            "filename": rejected_path.name,
            "sha256": sha256_file(rejected_path),
            "records": len(rejected),
        },
        "shards": shard_records,
        "statistics": statistics,
    }
    manifest = {**manifest_core, "deterministic_content_hash": canonical_json_hash(manifest_core)}
    atomic_write_json(incomplete / "tokenized_corpus_manifest.json", manifest)
    atomic_write_json(incomplete / "shard_checksums.json", checksums)
    atomic_write_json(incomplete / "tokenization_statistics.json", statistics)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-token-cap", type=int, default=2_000_000)
    args = parser.parse_args()
    manifest = tokenize_phase1b_corpus(
        args.processed_dir,
        args.output_dir,
        shard_token_cap=args.shard_token_cap,
    )
    print(json.dumps({"result": "PASS", "statistics": manifest["statistics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
