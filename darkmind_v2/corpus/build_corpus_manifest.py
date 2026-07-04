"""Build deterministic DarkMind v2 corpus manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .deduplicate_text import read_documents, sha256_text
    from .detect_language import detect_language
except ImportError:  # pragma: no cover - CLI fallback
    from deduplicate_text import read_documents, sha256_text
    from detect_language import detect_language


PIPELINE_VERSION = "darkmind-v2-phase0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def deterministic_split_hashes(document_ids: list[str], *, seed: int) -> dict[str, str]:
    keyed = sorted((sha256_text(f"{seed}:{document_id}"), document_id) for document_id in document_ids)
    total = len(keyed)
    train_cut = int(total * 0.98)
    val_cut = train_cut + int(total * 0.01)
    splits = {
        "train": [document_id for _, document_id in keyed[:train_cut]],
        "validation": [document_id for _, document_id in keyed[train_cut:val_cut]],
        "test": [document_id for _, document_id in keyed[val_cut:]],
    }
    return {name: stable_json_hash(ids) for name, ids in splits.items()}


def build_manifest(
    input_paths: list[Path],
    *,
    normalized_paths: list[Path] | None = None,
    config: dict[str, Any] | None = None,
    timestamp: str | None = None,
    split_seed: int = 20260703,
) -> dict[str, Any]:
    documents = []
    for path in input_paths:
        documents.extend(read_documents(path))

    language_counts = Counter(detect_language(document.text) for document in documents)
    character_counts = {document.document_id: len(document.text) for document in documents}
    source_names = sorted({str(document.metadata.get("source", "")) for document in documents if document.metadata.get("source")})
    source_urls = sorted({str(document.metadata.get("source_url", "")) for document in documents if document.metadata.get("source_url")})
    licenses = sorted({str(document.metadata.get("license", "")) for document in documents if document.metadata.get("license")})
    filtering_statistics = dict(config.get("filtering_statistics", {})) if config else {}
    dedup_settings = dict(config.get("deduplication", {})) if config else {}
    document_ids = [document.document_id for document in documents]

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generation_timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file_hashes": {str(path): sha256_file(path) for path in input_paths},
        "normalized_output_hashes": {
            str(path): sha256_file(path) for path in (normalized_paths or []) if path.exists()
        },
        "source_names": source_names,
        "source_urls": source_urls,
        "license_identifiers": licenses,
        "language_counts": dict(sorted(language_counts.items())),
        "character_counts": {
            "total": sum(character_counts.values()),
            "by_document": dict(sorted(character_counts.items())),
        },
        "document_counts": {"total": len(documents)},
        "filtering_statistics": filtering_statistics,
        "deduplication_settings": dedup_settings,
        "deterministic_split_seed": split_seed,
        "train_validation_test_hashes": deterministic_split_hashes(document_ids, seed=split_seed),
    }
    content_payload = {key: value for key, value in manifest.items() if key != "generation_timestamp"}
    manifest["deterministic_content_hash"] = stable_json_hash(content_payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic DarkMind v2 corpus manifest.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--normalized-output", type=Path, nargs="*", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=20260703)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    manifest = build_manifest(
        args.input,
        normalized_paths=args.normalized_output,
        config=config,
        timestamp=args.timestamp,
        split_seed=args.split_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
