"""Reopen and validate every full-corpus token shard and provenance hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash, sha256_file
from darkmind_v2.tokenizer.load_frozen_tokenizer import DEFAULT_FROZEN_DIR, EXPECTED_HASHES


def validate_full_tokenized_corpus(
    output_dir: Path,
    processed_dir: Path,
    *,
    plausibility_estimate_tokens: int = 13_054_542,
) -> dict[str, Any]:
    manifest = json.loads((output_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    core = {key: value for key, value in manifest.items() if key != "deterministic_content_hash"}
    if canonical_json_hash(core) != manifest.get("deterministic_content_hash"):
        failures.append("tokenized manifest deterministic hash mismatch")
    if manifest.get("dtype") != "uint16-le" or manifest.get("vocab_size") != 24000:
        failures.append("token dtype or vocabulary mismatch")
    if manifest.get("eos_token_id") != 3:
        failures.append("EOS token mismatch")

    for filename, metadata in manifest.get("source", {}).get("files", {}).items():
        path = processed_dir / filename
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            failures.append(f"source provenance mismatch: {filename}")
    for filename, expected in EXPECTED_HASHES.items():
        if sha256_file(DEFAULT_FROZEN_DIR / filename) != expected:
            failures.append(f"frozen tokenizer hash mismatch: {filename}")

    checksums = json.loads((output_dir / "shard_checksums.json").read_text(encoding="utf-8"))
    shard_by_name = {item["filename"]: item for item in manifest.get("shards", [])}
    split_tokens: Counter[str] = Counter()
    split_documents: Counter[str] = Counter()
    total_tokens = 0
    token_range_violations = 0
    shard_arrays: dict[str, np.memmap] = {}
    for filename, shard in shard_by_name.items():
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing shard: {filename}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != shard["sha256"] or checksums.get(filename) != actual_hash:
            failures.append(f"shard checksum mismatch: {filename}")
        if path.stat().st_size != shard["bytes"] or path.stat().st_size % 2:
            failures.append(f"shard byte size mismatch: {filename}")
        values = np.memmap(path, mode="r", dtype="<u2")
        shard_arrays[filename] = values
        if len(values) != shard["tokens"]:
            failures.append(f"shard token count mismatch: {filename}")
        if len(values):
            token_range_violations += int(np.count_nonzero(values >= 24000))
        total_tokens += len(values)
        split_tokens[shard["split"]] += len(values)

    boundary_metadata = manifest["document_boundaries"]
    boundary_path = output_dir / boundary_metadata["filename"]
    if sha256_file(boundary_path) != boundary_metadata["sha256"]:
        failures.append("document boundary manifest hash mismatch")
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    boundary_records = 0
    previous_end_by_shard: dict[str, int] = {}
    with boundary_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            boundary_records += 1
            document_id = record["id"]
            text_hash = record["text_sha256"]
            shard_name = record["shard"]
            if document_id in seen_ids or text_hash in seen_text:
                failures.append(f"duplicate boundary identity at line {line_number}")
            seen_ids.add(document_id)
            seen_text.add(text_hash)
            if shard_name not in shard_arrays:
                failures.append(f"boundary references missing shard: {shard_name}")
                continue
            shard = shard_by_name[shard_name]
            if record["split"] != shard["split"]:
                failures.append(f"boundary split mismatch: {document_id}")
            start = int(record["start_offset"])
            end = int(record["end_offset"])
            expected_start = previous_end_by_shard.get(shard_name, 0)
            if start != expected_start or end <= start or end > len(shard_arrays[shard_name]):
                failures.append(f"invalid or non-contiguous boundary: {document_id}")
            elif int(shard_arrays[shard_name][end - 1]) != 3:
                failures.append(f"missing EOS boundary: {document_id}")
            previous_end_by_shard[shard_name] = end
            split_documents[record["split"]] += 1

    for filename, shard in shard_by_name.items():
        if previous_end_by_shard.get(filename, 0) != shard["tokens"]:
            failures.append(f"boundary coverage incomplete: {filename}")
        if split_documents[shard["split"]] < 0:
            failures.append("unreachable split count guard")
    if boundary_records != boundary_metadata["records"]:
        failures.append("boundary record count mismatch")
    if token_range_violations:
        failures.append(f"token range violations: {token_range_violations}")

    statistics = manifest["statistics"]
    if total_tokens != statistics["total_tokens"] or dict(split_tokens) != statistics["split_tokens"]:
        failures.append("aggregate token statistics mismatch")
    if dict(split_documents) != statistics["split_documents"]:
        failures.append("aggregate document statistics mismatch")
    rejected_lines = sum(
        1 for line in (output_dir / "rejected_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if rejected_lines != statistics["rejected_documents"]:
        failures.append("rejected record count mismatch")
    if plausibility_estimate_tokens <= 0:
        raise ValueError("plausibility estimate must be positive")
    estimate_ratio = total_tokens / plausibility_estimate_tokens
    if not 0.75 <= estimate_ratio <= 1.25:
        failures.append(f"token total is implausible relative to Phase 2A estimate: ratio={estimate_ratio:.4f}")

    report_core = {
        "result": "FAIL" if failures else "PASS",
        "failures": failures,
        "total_tokens": total_tokens,
        "split_tokens": dict(split_tokens),
        "split_documents": dict(split_documents),
        "shards": len(shard_by_name),
        "boundary_records": boundary_records,
        "token_range_violations": token_range_violations,
        "unknown_tokens": statistics.get("unknown_tokens"),
        "estimate_ratio": round(estimate_ratio, 6),
        "manifest_hash": manifest["deterministic_content_hash"],
        "boundaries_hash": boundary_metadata["sha256"],
        "shard_checksums_hash": sha256_file(output_dir / "shard_checksums.json"),
    }
    return {**report_core, "validation_content_hash": canonical_json_hash(report_core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("darkmind_v2/data/phase1b/processed"))
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    report = validate_full_tokenized_corpus(args.output_dir, args.processed_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
