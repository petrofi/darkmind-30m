"""Validate deterministic token shards and document boundaries."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash, sha256_file


def read_uint16_le(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 2:
        raise ValueError(f"uint16 shard has an odd byte count: {path}")
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def validate_tokenized_shards(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "tokenized_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    core = {key: value for key, value in manifest.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != manifest.get("deterministic_content_hash"):
        failures.append("manifest deterministic content hash mismatch")
    if manifest.get("dtype") != "uint16-le":
        failures.append("unsupported token dtype")
    vocab_size = int(manifest.get("vocab_size", 0))
    eos_token_id = int(manifest.get("eos_token_id", -1))
    documents_by_shard: dict[str, list[dict[str, Any]]] = {}
    document_splits: dict[str, str] = {}
    text_splits: dict[str, str] = {}

    for document in manifest.get("documents", []):
        document_id = document["id"]
        split = document["split"]
        text_hash = document["text_sha256"]
        if document_id in document_splits and document_splits[document_id] != split:
            failures.append(f"document ID crosses splits: {document_id}")
        if text_hash in text_splits and text_splits[text_hash] != split:
            failures.append(f"document content crosses splits: {document_id}")
        document_splits[document_id] = split
        text_splits[text_hash] = split
        documents_by_shard.setdefault(document["shard"], []).append(document)

    total_tokens = 0
    for shard in manifest.get("shards", []):
        path = output_dir / shard["filename"]
        if not path.is_file():
            failures.append(f"missing shard: {shard['filename']}")
            continue
        if sha256_file(path) != shard["sha256"]:
            failures.append(f"shard checksum mismatch: {shard['filename']}")
        tokens = read_uint16_le(path)
        total_tokens += len(tokens)
        if len(tokens) != shard["tokens"]:
            failures.append(f"shard token count mismatch: {shard['filename']}")
        if tokens and max(tokens) >= vocab_size:
            failures.append(f"token outside vocabulary: {shard['filename']}")
        for document in documents_by_shard.get(shard["filename"], []):
            start = int(document["start_offset"])
            end = int(document["end_offset"])
            if start < 0 or end > len(tokens) or end <= start:
                failures.append(f"invalid offsets for document: {document['id']}")
            elif tokens[end - 1] != eos_token_id:
                failures.append(f"document lacks EOS boundary: {document['id']}")

    expected_total = manifest.get("statistics", {}).get("total_tokens")
    if total_tokens != expected_total:
        failures.append("aggregate token count mismatch")
    return {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "shards": len(manifest.get("shards", [])),
        "documents": len(manifest.get("documents", [])),
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = validate_tokenized_shards(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
