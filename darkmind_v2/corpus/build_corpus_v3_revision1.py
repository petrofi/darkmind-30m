"""Import the Phase 1B seed and measure aggregate Corpus V3 revision-1 capacity."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from darkmind_v2.corpus.build_corpus_v3_tranche import (
    deterministic_split,
    iter_paragraphs,
    load_contamination_material,
    normalized_hash_text,
    paragraph_hash,
    run_inventory,
)
from darkmind_v2.corpus.download_corpus_v3_revision1 import (
    DEFAULT_CONFIG,
    SUPPLEMENTAL_SOURCE_IDS,
    atomic_write_json,
    atomic_write_text,
    file_hashes,
    repository_path,
    validate_config,
)
from darkmind_v2.data_pipeline.tokenize_phase1b_corpus import iter_split_documents
from darkmind_v2.data_pipeline.validate_full_tokenized_corpus import validate_full_tokenized_corpus
from darkmind_v2.tokenizer.load_frozen_tokenizer import load_frozen_tokenizer


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "eval")
SPLIT_FILES = {
    "train": "tokenizer_train.txt",
    "validation": "tokenizer_validation.txt",
    "eval": "tokenizer_eval.txt",
}
PYTHON_ARCHIVES = {
    "python_docs_tr_3_14_6": "python-3.14-docs-text-tr-3.14.6.tar.bz2",
    "python_docs_en_3_14_6": "python-3.14-docs-text-en-3.14.6.tar.bz2",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            yield value


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return file_hashes(path)["sha256"]


def classify_seed_category(content_type: str, technical_types: set[str]) -> str:
    return "technical_educational" if content_type in technical_types else "general_prose"


def seed_split(document_id: str, normalized_hash: str, split_policy: dict[str, Any]) -> str:
    return deterministic_split(document_id, normalized_hash, split_policy)


def _verify_processed_hashes(processed_dir: Path, tokenized_manifest: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for filename, expected in tokenized_manifest["source"]["files"].items():
        path = processed_dir / filename
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"Phase 1B processed file byte mismatch: {filename}")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Phase 1B processed file hash mismatch: {filename}")
        verified[filename] = actual
    return verified


def import_phase1b_seed(
    config: dict[str, Any],
    processed_dir: Path,
    tokenized_dir: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    seed_root = runtime_root / "seed"
    seed_root.mkdir(parents=True, exist_ok=True)
    tokenized_manifest_path = tokenized_dir / "tokenized_corpus_manifest.json"
    tokenized_manifest = json.loads(tokenized_manifest_path.read_text(encoding="utf-8"))
    expected = config["seed_inputs"]
    if tokenized_manifest.get("deterministic_content_hash") != expected["expected_content_hash"]:
        raise ValueError("Phase 1B tokenized manifest content hash changed")
    if tokenized_manifest["document_boundaries"]["sha256"] != expected["expected_boundary_hash"]:
        raise ValueError("Phase 1B boundary hash changed")
    if int(tokenized_manifest["statistics"]["accepted_documents"]) != int(expected["expected_documents"]):
        raise ValueError("Phase 1B document count changed")
    if int(tokenized_manifest["statistics"]["total_tokens"]) != int(expected["expected_total_tokens"]):
        raise ValueError("Phase 1B token count changed")
    processed_hashes = _verify_processed_hashes(processed_dir, tokenized_manifest)

    validation = validate_full_tokenized_corpus(tokenized_dir, processed_dir)
    if validation["result"] != "PASS" or validation["failures"]:
        raise ValueError(f"Phase 1B full-tokenization validation failed: {validation['failures']}")
    expected_validation_hash = "999bc95f53559e8a9f16aed40b1e9ac2678e49577aae6a4fd6298dfaf16bb25b"
    if validation["validation_content_hash"] != expected_validation_hash:
        raise ValueError("Phase 1B validation content hash changed")

    attribution = iter_jsonl(processed_dir / "attribution_manifest.jsonl")
    boundaries = iter_jsonl(tokenized_dir / tokenized_manifest["document_boundaries"]["filename"])
    technical_types = set(expected["technical_content_types"])
    language_tokens: collections.Counter[str] = collections.Counter()
    category_tokens: collections.Counter[str] = collections.Counter()
    source_tokens: collections.Counter[str] = collections.Counter()
    split_tokens: collections.Counter[str] = collections.Counter()
    split_documents: collections.Counter[str] = collections.Counter()
    old_split_tokens: collections.Counter[str] = collections.Counter()
    documents = 0
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    output_path = seed_root / "documents.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for old_split in SPLITS:
            text_path = processed_dir / SPLIT_FILES[old_split]
            for source_order, text in enumerate(iter_split_documents(text_path), start=1):
                try:
                    metadata = next(attribution)
                    boundary = next(boundaries)
                except StopIteration as exc:
                    raise ValueError(f"seed metadata ended before {old_split}:{source_order}") from exc
                document_id = str(metadata.get("id", ""))
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if metadata.get("selected_split") != old_split or boundary.get("split") != old_split:
                    raise ValueError(f"seed split mismatch for {document_id}")
                if boundary.get("id") != document_id or boundary.get("text_sha256") != text_hash:
                    raise ValueError(f"seed identity/hash mismatch for {document_id}")
                if int(metadata.get("selected_character_count", -1)) != len(text):
                    raise ValueError(f"seed character count mismatch for {document_id}")
                if document_id in seen_ids or text_hash in seen_hashes:
                    raise ValueError(f"duplicate seed identity/content: {document_id}")
                language = str(metadata.get("language", ""))
                content_type = str(metadata.get("content_type", ""))
                if language not in {"tr", "en"} or not content_type:
                    raise ValueError(f"seed lacks language/category metadata: {document_id}")
                if not metadata.get("source_id") or not metadata.get("license") or not metadata.get("license_url"):
                    raise ValueError(f"seed lacks provenance/license metadata: {document_id}")
                token_count = int(boundary["tokens"])
                category = classify_seed_category(content_type, technical_types)
                normalized_hash = hashlib.sha256(normalized_hash_text(text).encode("utf-8")).hexdigest()
                new_split = seed_split(document_id, normalized_hash, config["split_policy"])
                record = {
                    "id": document_id,
                    "source_id": metadata["source_id"],
                    "source_native_id": document_id.split(":", 1)[-1],
                    "snapshot": metadata.get("snapshot_date", metadata.get("source_version", "phase1b")),
                    "language": language,
                    "category": category,
                    "content_type": content_type,
                    "quality_tier": "A",
                    "license": metadata["license"],
                    "license_url": metadata["license_url"],
                    "source_url": metadata.get("source_url", metadata.get("source_homepage")),
                    "attribution": metadata,
                    "text": text,
                    "text_sha256": text_hash,
                    "normalized_content_sha256": normalized_hash,
                    "duplicate_cluster_id": normalized_hash,
                    "token_count": token_count,
                    "original_split": old_split,
                    "split": new_split,
                }
                write_jsonl_record(output, record)
                documents += 1
                seen_ids.add(document_id)
                seen_hashes.add(text_hash)
                language_tokens[language] += token_count
                category_tokens[category] += token_count
                source_tokens[metadata["source_id"]] += token_count
                split_tokens[new_split] += token_count
                split_documents[new_split] += 1
                old_split_tokens[old_split] += token_count
                if documents % 50_000 == 0:
                    print(f"seed import progress documents={documents} tokens={sum(language_tokens.values())}", flush=True)
    os.replace(temporary, output_path)

    try:
        extra_attribution = next(attribution)
    except StopIteration:
        extra_attribution = None
    try:
        extra_boundary = next(boundaries)
    except StopIteration:
        extra_boundary = None
    if extra_attribution is not None or extra_boundary is not None:
        raise ValueError("seed metadata has unexplained trailing records")
    if documents != int(expected["expected_documents"]):
        raise ValueError("seed import document count differs from frozen manifest")
    total_tokens = sum(language_tokens.values())
    if total_tokens != int(expected["expected_total_tokens"]):
        raise ValueError("seed import token count differs from frozen manifest")

    local_inputs = {
        "phase1b_processed": str(processed_dir.resolve()),
        "phase2a_tokenized": str(tokenized_dir.resolve()),
        "read_only": True,
    }
    atomic_write_json(seed_root / "local_inputs.json", local_inputs)
    result = {
        "schema_version": "darkmind-v2-corpus-v3-seed-import-v1",
        "result": "PASS",
        "documents": documents,
        "total_tokens": total_tokens,
        "language_tokens": dict(sorted(language_tokens.items())),
        "category_tokens": dict(sorted(category_tokens.items())),
        "source_tokens": dict(sorted(source_tokens.items())),
        "new_split_tokens": dict(sorted(split_tokens.items())),
        "new_split_documents": dict(sorted(split_documents.items())),
        "original_split_tokens": dict(sorted(old_split_tokens.items())),
        "documents_sha256": sha256_file(output_path),
        "processed_hashes": processed_hashes,
        "tokenized_manifest_content_hash": tokenized_manifest["deterministic_content_hash"],
        "boundary_hash": tokenized_manifest["document_boundaries"]["sha256"],
        "validation_content_hash": validation["validation_content_hash"],
        "token_range_violations": validation["token_range_violations"],
        "missing_attribution": 0,
        "missing_license": 0,
        "duplicate_ids": 0,
        "duplicate_text_hashes": 0,
        "resplit_policy": config["split_policy"],
    }
    atomic_write_json(seed_root / "seed_import_manifest.json", result)
    report = [
        "# Phase 3C.1 Seed Import",
        "",
        "Status: **PASS**",
        "",
        f"Imported documents: {documents:,}",
        f"Imported tokens: {total_tokens:,}",
        f"Turkish tokens: {language_tokens['tr']:,}",
        f"English tokens: {language_tokens['en']:,}",
        f"Technical/educational tokens: {category_tokens['technical_educational']:,}",
        f"General prose tokens: {category_tokens['general_prose']:,}",
        "",
        f"Tokenized manifest content hash: `{tokenized_manifest['deterministic_content_hash']}`",
        f"Boundary hash: `{tokenized_manifest['document_boundaries']['sha256']}`",
        f"Validation content hash: `{validation['validation_content_hash']}`",
        "",
        "Document IDs, languages, content categories, source provenance, licenses, normalized content hashes, token counts, EOS boundaries, and token ranges were verified.",
        "",
        "All seed documents were reassigned deterministically to the Corpus V3 98/1/1 document split. No source worktree was modified.",
        "",
    ]
    atomic_write_text(repository_path(config["reports_root"]) / "phase3c1_seed_import.md", "\n".join(report))
    return result


def verify_python_inputs(config: dict[str, Any], phase3c_runtime: Path) -> dict[str, Any]:
    expected = config["existing_python_inputs"]["required_archive_sha256"]
    archive_hashes: dict[str, str] = {}
    for source_id, filename in PYTHON_ARCHIVES.items():
        path = phase3c_runtime / "raw" / source_id / filename
        actual = sha256_file(path)
        if actual != expected[source_id]:
            raise ValueError(f"verified Python archive changed: {source_id}")
        archive_hashes[source_id] = actual
    accepted_path = phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"]
    source_tokens: collections.Counter[str] = collections.Counter()
    language_tokens: collections.Counter[str] = collections.Counter()
    documents = 0
    for record in iter_jsonl(accepted_path):
        if record["source_id"] not in PYTHON_ARCHIVES:
            raise ValueError(f"unexpected Phase 3C accepted source: {record['source_id']}")
        source_tokens[record["source_id"]] += int(record["token_count"])
        language_tokens[record["language"]] += int(record["token_count"])
        documents += 1
    result = {
        "result": "PASS",
        "documents": documents,
        "total_tokens": sum(source_tokens.values()),
        "source_tokens": dict(sorted(source_tokens.items())),
        "language_tokens": dict(sorted(language_tokens.items())),
        "accepted_documents_sha256": sha256_file(accepted_path),
        "archive_sha256": archive_hashes,
    }
    if result["total_tokens"] != 2_907_272:
        raise ValueError("new unique Python token total changed")
    return result


def load_historical_paragraph_hashes(
    document_paths: Iterable[Path],
    *,
    minimum_characters: int,
) -> set[str]:
    hashes: set[str] = set()
    for path in document_paths:
        for record in iter_jsonl(path):
            for paragraph in iter_paragraphs(record["text"]):
                if len(paragraph) >= minimum_characters:
                    hashes.add(paragraph_hash(paragraph))
    return hashes


def verify_stage_archives(config: dict[str, Any], runtime_root: Path, stage: str) -> dict[str, Any]:
    manifest_path = runtime_root / f"{stage}_download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("result") != "PASS" or manifest.get("official_metadata_verification", {}).get("result") != "PASS":
        raise ValueError(f"{stage} download manifest is not verified")
    records = {record["filename"]: record for record in manifest["downloads"]}
    checked: list[dict[str, Any]] = []
    selected_ids = set(manifest["source_ids"])
    for source in config["sources"]:
        if source["source_id"] not in selected_ids:
            continue
        for item in source.get("files", []):
            record = records.get(item["filename"])
            if record is None:
                raise ValueError(f"verified archive missing from manifest: {item['filename']}")
            path = ROOT / record["path"]
            hashes = file_hashes(path, ("sha256", "md5", "sha1"))
            if path.stat().st_size != int(item["expected_bytes"]):
                raise ValueError(f"archive bytes changed: {item['filename']}")
            if hashes["md5"] != item["md5"] or hashes["sha1"] != item["sha1"]:
                raise ValueError(f"archive official checksum changed: {item['filename']}")
            if hashes["sha256"] != record["sha256"]:
                raise ValueError(f"archive local SHA-256 changed: {item['filename']}")
            checked.append({"filename": item["filename"], "bytes": path.stat().st_size, **hashes})
    return {"result": "PASS", "archives": checked}


def technical_capacity_gate(
    seed_technical_tokens: int,
    python_tokens: int,
    supplemental_tokens: dict[str, int],
    source_caps: dict[str, int],
    *,
    target: int = 15_000_000,
    tolerance_percent: float = 2.0,
) -> dict[str, Any]:
    eligible = {
        source_id: min(int(tokens), int(source_caps[source_id]))
        for source_id, tokens in supplemental_tokens.items()
    }
    total = int(seed_technical_tokens) + int(python_tokens) + sum(eligible.values())
    minimum = int(target * (1 - tolerance_percent / 100))
    return {
        "result": "PASS" if total >= minimum else "FAIL",
        "target_tokens": target,
        "minimum_tokens": minimum,
        "seed_technical_tokens": int(seed_technical_tokens),
        "new_python_tokens": int(python_tokens),
        "supplemental_accepted_tokens": dict(sorted(supplemental_tokens.items())),
        "supplemental_cap_eligible_tokens": dict(sorted(eligible.items())),
        "combined_unique_technical_capacity": total,
        "shortfall_to_minimum": max(0, minimum - total),
    }


def next_inventory_root(runtime_root: Path) -> Path:
    base = runtime_root / "supplemental_inventory"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = runtime_root / f"supplemental_inventory_retry{index}"
        if not candidate.exists():
            return candidate
        index += 1


def supplemental_capacity(
    config: dict[str, Any],
    phase3c_runtime: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    seed_manifest = json.loads((runtime_root / "seed" / "seed_import_manifest.json").read_text(encoding="utf-8"))
    if seed_manifest.get("result") != "PASS":
        raise ValueError("seed import did not pass")
    python = verify_python_inputs(config, phase3c_runtime)
    raw_verification = verify_stage_archives(config, runtime_root, "supplemental")
    supplemental_sources = [
        source for source in config["sources"] if source["source_id"] in SUPPLEMENTAL_SOURCE_IDS
    ]
    selected_config = {**config, "sources": supplemental_sources}
    minimum_historical = int(config["deduplication_policy"]["minimum_historical_paragraph_characters"])
    historical_hashes = load_historical_paragraph_hashes(
        (
            runtime_root / "seed" / "documents.jsonl",
            phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"],
        ),
        minimum_characters=minimum_historical,
    )
    contamination_exact, contamination_substrings = load_contamination_material(config)
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    inventory_root = next_inventory_root(runtime_root)
    inventory = run_inventory(
        selected_config,
        inventory_root,
        tokenizer=tokenizer,
        historical_hashes=historical_hashes,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    supplemental_tokens = {
        source_id: int(stats["accepted_tokens"])
        for source_id, stats in inventory["source_statistics"].items()
    }
    source_caps = {
        source["source_id"]: int(source["maximum_source_cap_tokens"])
        for source in supplemental_sources
    }
    gate = technical_capacity_gate(
        seed_manifest["category_tokens"]["technical_educational"],
        python["total_tokens"],
        supplemental_tokens,
        source_caps,
    )
    result = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-supplemental-capacity-v1",
        "result": "PASS" if gate["result"] == "PASS" else "FAIL_TECHNICAL_CAPACITY",
        "raw_verification": raw_verification,
        "seed": seed_manifest,
        "new_python": python,
        "inventory": inventory,
        "inventory_runtime_directory": str(inventory_root.relative_to(ROOT)),
        "technical_capacity_gate": gate,
        "wikipedia_acquisition_allowed": gate["result"] == "PASS",
        "final_corpus_built": False,
        "model_training_started": False,
    }
    atomic_write_json(runtime_root / "supplemental_capacity_result.json", result)

    lines = [
        "# Phase 3C.1 Supplemental Capacity",
        "",
        f"Status: **{result['result']}**",
        "",
        f"Phase 1B technical/educational tokens: {gate['seed_technical_tokens']:,}",
        f"New unique Python tokens: {gate['new_python_tokens']:,}",
        "",
        "| Source | Accepted documents | Accepted unique tokens | Cap-eligible tokens | Source cap |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in supplemental_sources:
        source_id = source["source_id"]
        stats = inventory["source_statistics"][source_id]
        lines.append(
            f"| {source_id} | {stats['accepted_documents']:,} | {stats['accepted_tokens']:,} | "
            f"{gate['supplemental_cap_eligible_tokens'][source_id]:,} | {source['maximum_source_cap_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"Combined unique technical/educational capacity: {gate['combined_unique_technical_capacity']:,}",
            f"Required minimum: {gate['minimum_tokens']:,}",
            f"Shortfall: {gate['shortfall_to_minimum']:,}",
            "",
            f"Exact duplicate removals: {inventory['duplicate_report']['exact_duplicate_removals']:,}",
            f"Near-duplicate removals: {inventory['duplicate_report']['near_duplicate_removals']:,}",
            f"Seed/Python paragraph overlap removals: {inventory['duplicate_report']['phase1b_paragraph_overlap_removals']:,}",
            f"Evaluation contamination removals: {inventory['contamination_report']['rejected_documents']:,}",
            "",
            "Wikipedia acquisition is permitted only when this aggregate technical-capacity gate passes.",
            "",
        ]
    )
    if gate["result"] != "PASS":
        lines.extend(
            [
                "No Wikipedia archive was acquired or counted after this failed gate.",
                "",
                "DARKMIND V2 CORPUS V3 AGGREGATE 100M REQUIRES FURTHER SOURCE CORRECTION",
                "",
            ]
        )
    atomic_write_text(repository_path(config["reports_root"]) / "phase3c1_supplemental_capacity.md", "\n".join(lines))
    return result


def index_jsonl_for_selection(path: Path) -> list[tuple[str, str, int, int, int]]:
    index: list[tuple[str, str, int, int, int]] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            index.append(
                (
                    str(record["normalized_content_sha256"]),
                    str(record["id"]),
                    int(record["token_count"]),
                    offset,
                    len(line),
                )
            )
    return sorted(index, key=lambda item: (item[0], item[1]))


def select_jsonl_to_target(
    path: Path,
    target_tokens: int,
    source_cap_tokens: int,
) -> tuple[list[tuple[str, str, int, int, int]], int]:
    if target_tokens <= 0 or target_tokens > source_cap_tokens:
        raise ValueError("selection target must be positive and within the source cap")
    selected: list[tuple[str, str, int, int, int]] = []
    tokens = 0
    for entry in index_jsonl_for_selection(path):
        token_count = entry[2]
        if tokens + token_count > source_cap_tokens:
            continue
        selected.append(entry)
        tokens += token_count
        if tokens >= target_tokens:
            break
    if tokens < target_tokens:
        raise ValueError(f"source cannot meet deterministic selection target: {path.name}")
    return selected, tokens


def copy_selected_jsonl(
    source_path: Path,
    selections: Iterable[tuple[str, str, int, int, int]],
    output_handle: Any,
) -> dict[str, Any]:
    language_tokens: collections.Counter[str] = collections.Counter()
    documents = 0
    tokens = 0
    with source_path.open("rb") as source:
        for _, _, token_count, offset, length in selections:
            source.seek(offset)
            line = source.read(length)
            record = json.loads(line.decode("utf-8"))
            write_jsonl_record(output_handle, record)
            documents += 1
            tokens += token_count
            language_tokens[record["language"]] += token_count
    return {
        "documents": documents,
        "tokens": tokens,
        "language_tokens": dict(sorted(language_tokens.items())),
    }


def compute_revised_allocation(
    config: dict[str, Any],
    phase3c_runtime: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    capacity = json.loads((runtime_root / "supplemental_capacity_result.json").read_text(encoding="utf-8"))
    if capacity.get("result") != "PASS" or not capacity.get("wikipedia_acquisition_allowed"):
        raise ValueError("technical capacity gate did not pass")
    inventory_root = ROOT / capacity["inventory_runtime_directory"]
    seed_documents = runtime_root / "seed" / "documents.jsonl"
    python_documents = phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"]

    seed_language: collections.Counter[str] = collections.Counter()
    seed_category: collections.Counter[str] = collections.Counter()
    seed_language_category: collections.Counter[str] = collections.Counter()
    seed_documents_count = 0
    for record in iter_jsonl(seed_documents):
        tokens = int(record["token_count"])
        seed_language[record["language"]] += tokens
        seed_category[record["category"]] += tokens
        seed_language_category[f"{record['language']}:{record['category']}"] += tokens
        seed_documents_count += 1

    python_language: collections.Counter[str] = collections.Counter()
    python_source: collections.Counter[str] = collections.Counter()
    python_documents_count = 0
    for record in iter_jsonl(python_documents):
        tokens = int(record["token_count"])
        python_language[record["language"]] += tokens
        python_source[record["source_id"]] += tokens
        python_documents_count += 1

    technical_target = int(config["targets"]["categories"]["technical_educational"])
    fixed_technical = seed_category["technical_educational"] + sum(python_language.values())
    supplemental_needed = technical_target - fixed_technical
    if supplemental_needed <= 0:
        raise ValueError("preserved seed/Python technical tokens already exceed the target")
    source_by_id = {source["source_id"]: source for source in config["sources"]}
    tr_source_id = "wikimedia_trwikibooks_20260701"
    en_primary_id = "wikimedia_enwikiversity_20260201"
    en_secondary_id = "wikimedia_enwikibooks_20260701"
    accepted = capacity["inventory"]["source_statistics"]
    tr_target = min(
        supplemental_needed,
        int(accepted[tr_source_id]["accepted_tokens"]),
        int(source_by_id[tr_source_id]["maximum_source_cap_tokens"]),
    )
    remaining_after_tr = supplemental_needed - tr_target
    if remaining_after_tr < 2:
        raise ValueError("technical allocation lacks room for approved English source diversity")
    en_primary_target = remaining_after_tr // 2

    selection_path = runtime_root / "allocation" / "technical_selected_documents.jsonl"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selection_path.with_suffix(".jsonl.tmp")
    selections: dict[str, dict[str, Any]] = {}
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        tr_path = inventory_root / "deduplicated" / f"{tr_source_id}.jsonl"
        tr_entries, tr_tokens = select_jsonl_to_target(
            tr_path, tr_target, int(source_by_id[tr_source_id]["maximum_source_cap_tokens"])
        )
        selections[tr_source_id] = copy_selected_jsonl(tr_path, tr_entries, output)

        primary_path = inventory_root / "deduplicated" / f"{en_primary_id}.jsonl"
        primary_entries, primary_tokens = select_jsonl_to_target(
            primary_path,
            en_primary_target,
            int(source_by_id[en_primary_id]["maximum_source_cap_tokens"]),
        )
        selections[en_primary_id] = copy_selected_jsonl(primary_path, primary_entries, output)

        secondary_target = max(1, technical_target - fixed_technical - tr_tokens - primary_tokens)
        secondary_path = inventory_root / "deduplicated" / f"{en_secondary_id}.jsonl"
        secondary_entries, _ = select_jsonl_to_target(
            secondary_path,
            secondary_target,
            int(source_by_id[en_secondary_id]["maximum_source_cap_tokens"]),
        )
        selections[en_secondary_id] = copy_selected_jsonl(secondary_path, secondary_entries, output)
    os.replace(temporary, selection_path)

    supplemental_language: collections.Counter[str] = collections.Counter()
    supplemental_total = 0
    supplemental_documents = 0
    for value in selections.values():
        supplemental_total += int(value["tokens"])
        supplemental_documents += int(value["documents"])
        supplemental_language.update(value["language_tokens"])
    technical_total = fixed_technical + supplemental_total
    technical_tolerance = int(technical_target * config["targets"]["technical_tolerance_percent"] / 100)
    if not technical_target - technical_tolerance <= technical_total <= technical_target + technical_tolerance:
        raise ValueError(f"technical allocation outside aggregate tolerance: {technical_total}")

    base_language = seed_language + python_language + supplemental_language
    base_total = sum(base_language.values())
    language_targets = config["targets"]["languages"]
    wikipedia_targets = {
        language: int(language_targets[language]) - int(base_language[language])
        for language in ("tr", "en")
    }
    if any(value <= 0 for value in wikipedia_targets.values()):
        raise ValueError("preserved allocation exceeds a language target")
    remaining_total = int(config["targets"]["total_tokens"]) - base_total
    if sum(wikipedia_targets.values()) != remaining_total:
        raise ValueError("derived Wikipedia language deficits do not add up to the total deficit")
    final_general = seed_category["general_prose"] + remaining_total
    general_target = int(config["targets"]["categories"]["general_prose"])
    general_tolerance = int(general_target * config["targets"]["total_tolerance_percent"] / 100)
    if not general_target - general_tolerance <= final_general <= general_target + general_tolerance:
        raise ValueError("derived prose allocation is outside aggregate tolerance")

    result = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-allocation-v1",
        "result": "PASS",
        "seed": {
            "documents": seed_documents_count,
            "tokens": sum(seed_language.values()),
            "language_tokens": dict(sorted(seed_language.items())),
            "category_tokens": dict(sorted(seed_category.items())),
            "language_category_tokens": dict(sorted(seed_language_category.items())),
        },
        "new_python": {
            "documents": python_documents_count,
            "tokens": sum(python_language.values()),
            "language_tokens": dict(sorted(python_language.items())),
            "source_tokens": dict(sorted(python_source.items())),
        },
        "supplemental_selection": selections,
        "supplemental_selected_tokens": supplemental_total,
        "supplemental_selected_documents": supplemental_documents,
        "technical_tokens": technical_total,
        "base_language_tokens": dict(sorted(base_language.items())),
        "base_total_tokens": base_total,
        "wikipedia_target_tokens": wikipedia_targets,
        "wikipedia_total_tokens": remaining_total,
        "expected_final_tokens": base_total + remaining_total,
        "expected_final_category_tokens": {
            "general_prose": final_general,
            "technical_educational": technical_total,
        },
        "technical_selection_sha256": sha256_file(selection_path),
        "technical_selection_path": str(selection_path.relative_to(ROOT)),
    }
    atomic_write_json(runtime_root / "revised_allocation_manifest.json", result)
    lines = [
        "# Phase 3C.1 Revised Allocation",
        "",
        "Status: **PASS**",
        "",
        f"Phase 1B seed contribution: {result['seed']['tokens']:,} tokens",
        f"New unique Python contribution: {result['new_python']['tokens']:,} tokens",
        "",
        "| Technical source | Selected documents | Selected tokens | Source cap |",
        "|---|---:|---:|---:|",
    ]
    for source_id in (tr_source_id, en_primary_id, en_secondary_id):
        value = selections[source_id]
        lines.append(
            f"| {source_id} | {value['documents']:,} | {value['tokens']:,} | "
            f"{source_by_id[source_id]['maximum_source_cap_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            f"Aggregate technical/educational tokens: {technical_total:,}",
            f"Tokens before Wikipedia: {base_total:,}",
            f"Remaining Turkish deficit: {wikipedia_targets['tr']:,}",
            f"Remaining English deficit: {wikipedia_targets['en']:,}",
            f"Remaining prose/total deficit: {remaining_total:,}",
            "",
            f"Exact planned Turkish Wikipedia selection cap: {wikipedia_targets['tr']:,}",
            f"Exact planned English Wikipedia selection cap: {wikipedia_targets['en']:,}",
            f"Expected total after deterministic selection: {result['expected_final_tokens']:,}",
            "",
            "The obsolete 55M/30M source quotas were not reused. Wikipedia requirements were derived from the accepted seed and technical allocation.",
            "",
        ]
    )
    atomic_write_text(repository_path(config["reports_root"]) / "phase3c1_revised_allocation.md", "\n".join(lines))
    return result


def wikipedia_capacity(
    config: dict[str, Any],
    phase3c_runtime: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    allocation = json.loads((runtime_root / "revised_allocation_manifest.json").read_text(encoding="utf-8"))
    if allocation.get("result") != "PASS":
        raise ValueError("revised allocation did not pass")
    raw_verification = verify_stage_archives(config, runtime_root, "wikipedia")
    capacity = json.loads((runtime_root / "supplemental_capacity_result.json").read_text(encoding="utf-8"))
    supplemental_inventory_root = ROOT / capacity["inventory_runtime_directory"]
    minimum_historical = int(config["deduplication_policy"]["minimum_historical_paragraph_characters"])
    historical_hashes = load_historical_paragraph_hashes(
        (
            runtime_root / "seed" / "documents.jsonl",
            phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"],
            supplemental_inventory_root / "deduplicated" / "accepted_documents.jsonl",
        ),
        minimum_characters=minimum_historical,
    )
    wikipedia_ids = {"wikimedia_trwiki_20260701", "wikimedia_enwiki_20260701"}
    wikipedia_sources = [source for source in config["sources"] if source["source_id"] in wikipedia_ids]
    selected_config = {**config, "sources": wikipedia_sources}
    contamination_exact, contamination_substrings = load_contamination_material(config)
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    inventory_root = runtime_root / "wikipedia_inventory"
    if inventory_root.exists():
        index = 1
        while (runtime_root / f"wikipedia_inventory_retry{index}").exists():
            index += 1
        inventory_root = runtime_root / f"wikipedia_inventory_retry{index}"
    inventory = run_inventory(
        selected_config,
        inventory_root,
        tokenizer=tokenizer,
        historical_hashes=historical_hashes,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    source_for_language = {
        "tr": "wikimedia_trwiki_20260701",
        "en": "wikimedia_enwiki_20260701",
    }
    targets = allocation["wikipedia_target_tokens"]
    source_gate: dict[str, Any] = {}
    passed = True
    for language, source_id in source_for_language.items():
        available = int(inventory["source_statistics"][source_id]["accepted_tokens"])
        target = int(targets[language])
        source_gate[source_id] = {
            "language": language,
            "target_tokens": target,
            "available_unique_tokens": available,
            "shortfall": max(0, target - available),
            "pass": available >= target,
        }
        passed = passed and available >= target
    result = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-wikipedia-capacity-v1",
        "result": "PASS" if passed else "FAIL_WIKIPEDIA_CAPACITY",
        "raw_verification": raw_verification,
        "inventory_runtime_directory": str(inventory_root.relative_to(ROOT)),
        "inventory": inventory,
        "source_gate": source_gate,
        "final_corpus_built": False,
        "model_training_started": False,
    }
    atomic_write_json(runtime_root / "wikipedia_capacity_result.json", result)
    lines = [
        "# Phase 3C.1 Corpus Build",
        "",
        f"Wikipedia capacity status: **{result['result']}**",
        "",
        "| Source | Required tokens | Accepted unique tokens | Shortfall |",
        "|---|---:|---:|---:|",
    ]
    for source_id in ("wikimedia_trwiki_20260701", "wikimedia_enwiki_20260701"):
        gate = source_gate[source_id]
        lines.append(
            f"| {source_id} | {gate['target_tokens']:,} | {gate['available_unique_tokens']:,} | {gate['shortfall']:,} |"
        )
    lines.extend(
        [
            "",
            f"Exact duplicate removals: {inventory['duplicate_report']['exact_duplicate_removals']:,}",
            f"Near-duplicate removals: {inventory['duplicate_report']['near_duplicate_removals']:,}",
            f"Historical paragraph overlap removals: {inventory['duplicate_report']['phase1b_paragraph_overlap_removals']:,}",
            f"Evaluation contamination removals: {inventory['contamination_report']['rejected_documents']:,}",
            "",
        ]
    )
    if not passed:
        lines.extend(
            [
                "The final 100M corpus was not built because a language-specific Wikipedia capacity gate failed.",
                "",
                "DARKMIND V2 CORPUS V3 AGGREGATE 100M REQUIRES FURTHER SOURCE CORRECTION",
                "",
            ]
        )
    atomic_write_text(repository_path(config["reports_root"]) / "phase3c1_corpus_build.md", "\n".join(lines))
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("seed-import", "supplemental-capacity", "revised-allocation", "wikipedia-capacity"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase1b-processed", type=Path)
    parser.add_argument("--phase2a-tokenized", type=Path)
    parser.add_argument("--phase3c-runtime", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    runtime_root = repository_path(config["runtime_root"])
    if args.command == "seed-import":
        if args.phase1b_processed is None or args.phase2a_tokenized is None:
            parser.error("seed-import requires --phase1b-processed and --phase2a-tokenized")
        result = import_phase1b_seed(config, args.phase1b_processed, args.phase2a_tokenized, runtime_root)
    elif args.command == "supplemental-capacity":
        if args.phase3c_runtime is None:
            parser.error("supplemental-capacity requires --phase3c-runtime")
        result = supplemental_capacity(config, args.phase3c_runtime, runtime_root)
    elif args.command == "revised-allocation":
        if args.phase3c_runtime is None:
            parser.error("revised-allocation requires --phase3c-runtime")
        result = compute_revised_allocation(config, args.phase3c_runtime, runtime_root)
    else:
        if args.phase3c_runtime is None:
            parser.error("wikipedia-capacity requires --phase3c-runtime")
        result = wikipedia_capacity(config, args.phase3c_runtime, runtime_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
