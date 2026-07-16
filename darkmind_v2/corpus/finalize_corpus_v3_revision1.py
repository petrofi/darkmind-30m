"""Build, tokenize, and independently reproduce Corpus V3 revision 1."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from darkmind_v2.corpus.build_corpus_v3_revision1 import (
    SUPPLEMENTAL_SOURCE_IDS,
    copy_selected_jsonl,
    index_jsonl_for_selection,
    iter_jsonl,
    load_historical_paragraph_hashes,
    select_jsonl_to_target,
    sha256_file,
    verify_stage_archives,
)
from darkmind_v2.corpus.build_corpus_v3_tranche import (
    ShardWriter,
    deterministic_split,
    load_contamination_material,
    run_inventory,
)
from darkmind_v2.corpus.download_corpus_v3_revision1 import atomic_write_json, atomic_write_text, repository_path
from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.data_pipeline.validate_full_tokenized_corpus import validate_full_tokenized_corpus
from darkmind_v2.tokenizer.load_frozen_tokenizer import load_frozen_tokenizer


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "eval")
WIKIPEDIA_SOURCE_IDS = ("wikimedia_trwiki_20260701", "wikimedia_enwiki_20260701")
TECHNICAL_SOURCE_IDS = (
    "wikimedia_trwikibooks_20260701",
    "wikimedia_enwikiversity_20260201",
    "wikimedia_enwikibooks_20260701",
)
FINAL_TEXT_FILENAMES = (
    "documents.jsonl",
    "attribution_manifest.jsonl",
    "split_manifest.jsonl",
    "source_allocation.json",
    "language_category_allocation.json",
    "duplicate_report.json",
    "contamination_report.json",
    "rejection_report.json",
    "rejected_records.jsonl",
    "seed_import_manifest.json",
    "corpus_manifest.json",
)


def canonical_category(allocation_source_id: str, source_category: str) -> str:
    if allocation_source_id in WIKIPEDIA_SOURCE_IDS:
        return "general_prose"
    if allocation_source_id == "phase1b_seed":
        if source_category not in {"general_prose", "technical_educational"}:
            raise ValueError(f"unsupported Phase 1B aggregate category: {source_category}")
        return source_category
    return "technical_educational"


def target_within_tolerance(actual: int, target: int, tolerance_percent: float) -> bool:
    tolerance = target * tolerance_percent / 100
    return target - tolerance <= actual <= target + tolerance


def _read_indexed_record(path: Path, entry: tuple[str, str, int, int, int]) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(entry[3])
        return json.loads(handle.read(entry[4]).decode("utf-8"))


def _index_by_source(path: Path) -> dict[str, list[tuple[str, str, int, int, int]]]:
    result: dict[str, list[tuple[str, str, int, int, int]]] = collections.defaultdict(list)
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            result[str(record["source_id"])].append(
                (
                    str(record["normalized_content_sha256"]),
                    str(record["id"]),
                    int(record["token_count"]),
                    offset,
                    len(line),
                )
            )
    return {key: sorted(value, key=lambda item: (item[0], item[1])) for key, value in result.items()}


def _partition_entries(
    path: Path,
    entries: Iterable[tuple[str, str, int, int, int]],
    split_policy: dict[str, Any],
) -> dict[str, list[tuple[str, str, int, int, int]]]:
    partitions = {split: [] for split in SPLITS}
    with path.open("rb") as handle:
        for entry in entries:
            handle.seek(entry[3])
            record = json.loads(handle.read(entry[4]).decode("utf-8"))
            split = deterministic_split(record["id"], record["duplicate_cluster_id"], split_policy)
            partitions[split].append(entry)
    return partitions


def _select_unique_entries(
    path: Path,
    entries: Iterable[tuple[str, str, int, int, int]],
    seen_ids: set[str],
    seen_content: dict[str, str],
    *,
    target_tokens: int | None,
    source_cap_tokens: int,
) -> tuple[list[tuple[str, str, int, int, int]], int, list[dict[str, Any]]]:
    selected: list[tuple[str, str, int, int, int]] = []
    rejected: list[dict[str, Any]] = []
    tokens = 0
    with path.open("rb") as source:
        for entry in entries:
            content_hash, document_id, token_count, offset, length = entry
            representative = seen_content.get(content_hash)
            if document_id in seen_ids or representative is not None:
                source.seek(offset)
                record = json.loads(source.read(length).decode("utf-8"))
                rejected.append(
                    {
                        "id": document_id,
                        "source_id": record["source_id"],
                        "source_native_id": record.get("source_native_id"),
                        "reason": "final_cross_source_exact_duplicate",
                        "representative_id": representative or document_id,
                        "normalized_content_sha256": content_hash,
                        "record_type": "document",
                    }
                )
                continue
            if tokens + token_count > source_cap_tokens:
                continue
            selected.append(entry)
            tokens += token_count
            seen_ids.add(document_id)
            seen_content[content_hash] = document_id
            if target_tokens is not None and tokens >= target_tokens:
                break
    if target_tokens is not None and tokens < target_tokens:
        raise ValueError(f"unique source capacity cannot meet target: {path.name}")
    return selected, tokens, rejected


def _source_reports(
    phase3c_runtime: Path,
    supplemental_inventory_root: Path,
    wikipedia_inventory_root: Path,
) -> list[tuple[str, Path]]:
    return [
        ("new_python", phase3c_runtime / "source_gate" / "deduplicated"),
        ("supplemental", supplemental_inventory_root / "deduplicated"),
        ("wikipedia", wikipedia_inventory_root / "deduplicated"),
    ]


def _aggregate_decision_reports(
    output_root: Path,
    phase3c_runtime: Path,
    supplemental_inventory_root: Path,
    wikipedia_inventory_root: Path,
    final_duplicate_rejections: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    duplicate_totals: collections.Counter[str] = collections.Counter()
    contamination_rejections = 0
    accepted_contamination = 0
    report_hashes: dict[str, dict[str, str]] = {}
    rejection_counts: collections.Counter[str] = collections.Counter()
    rejected_records = 0
    rejected_path = output_root / "rejected_records.jsonl"
    with rejected_path.open("wb") as output:
        for label, directory in _source_reports(
            phase3c_runtime, supplemental_inventory_root, wikipedia_inventory_root
        ):
            duplicate_path = directory / "duplicate_report.json"
            contamination_path = directory / "contamination_report.json"
            rejection_path = directory / "rejected_records.jsonl"
            duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
            contamination = json.loads(contamination_path.read_text(encoding="utf-8"))
            for key in (
                "exact_duplicate_removals",
                "near_duplicate_removals",
                "phase1b_paragraph_overlap_removals",
                "cross_source_paragraph_overlap_removals",
                "unresolved_exact_duplicates",
                "unresolved_near_duplicate_clusters",
            ):
                duplicate_totals[key] += int(duplicate.get(key, 0))
            contamination_rejections += int(contamination["rejected_documents"])
            accepted_contamination += int(contamination["accepted_contamination_records"])
            report_hashes[label] = {
                "duplicate_report_sha256": sha256_file(duplicate_path),
                "contamination_report_sha256": sha256_file(contamination_path),
                "rejected_records_sha256": sha256_file(rejection_path),
            }
            with rejection_path.open("rb") as source:
                for line in source:
                    if not line.strip():
                        continue
                    output.write(line)
                    rejected_records += 1
                    record = json.loads(line.decode("utf-8"))
                    rejection_counts[str(record.get("reason", "unspecified"))] += 1
        for record in final_duplicate_rejections:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
            rejected_records += 1
            rejection_counts[record["reason"]] += 1
            duplicate_totals["exact_duplicate_removals"] += 1
    duplicate_report = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-duplicate-summary-v1",
        "result": "PASS"
        if duplicate_totals["unresolved_exact_duplicates"] == 0
        and duplicate_totals["unresolved_near_duplicate_clusters"] == 0
        else "FAIL",
        **dict(sorted(duplicate_totals.items())),
        "decision_report_hashes": report_hashes,
    }
    contamination_report = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-contamination-summary-v1",
        "result": "PASS" if accepted_contamination == 0 else "FAIL",
        "rejected_documents": contamination_rejections,
        "accepted_contamination_records": accepted_contamination,
        "decision_report_hashes": report_hashes,
    }
    rejection_report = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-rejection-summary-v1",
        "result": "PASS",
        "records": rejected_records,
        "by_reason": dict(sorted(rejection_counts.items())),
        "rejected_records_sha256": sha256_file(rejected_path),
    }
    atomic_write_json(output_root / "duplicate_report.json", duplicate_report)
    atomic_write_json(output_root / "contamination_report.json", contamination_report)
    atomic_write_json(output_root / "rejection_report.json", rejection_report)
    return duplicate_report, contamination_report, rejection_report


def _verify_frozen_assets(config: dict[str, Any]) -> dict[str, Any]:
    tokenizer_root = repository_path(config["tokenizer"]["path"])
    tokenizer_files = {
        "tokenizer.model": config["tokenizer"]["model_sha256"],
        "tokenizer.vocab": config["tokenizer"]["vocab_sha256"],
        "tokenizer_freeze_manifest.json": config["tokenizer"]["freeze_manifest_sha256"],
    }
    actual_tokenizer = {name: sha256_file(tokenizer_root / name) for name in tokenizer_files}
    base_actual = sha256_file(repository_path(config["base_v1"]["config_path"]))
    passed = actual_tokenizer == tokenizer_files and base_actual == config["base_v1"]["config_sha256"]
    return {
        "result": "PASS" if passed else "FAIL",
        "tokenizer": actual_tokenizer,
        "base_v1_config_sha256": base_actual,
    }


def _select_wikipedia(
    config: dict[str, Any],
    allocation: dict[str, Any],
    wikipedia_inventory_root: Path,
) -> dict[str, dict[str, Any]]:
    source_by_id = {source["source_id"]: source for source in config["sources"]}
    selected: dict[str, dict[str, Any]] = {}
    for language, source_id in (("tr", WIKIPEDIA_SOURCE_IDS[0]), ("en", WIKIPEDIA_SOURCE_IDS[1])):
        path = wikipedia_inventory_root / "deduplicated" / f"{source_id}.jsonl"
        entries, tokens = select_jsonl_to_target(
            path,
            int(allocation["wikipedia_target_tokens"][language]),
            int(source_by_id[source_id]["maximum_source_cap_tokens"]),
        )
        selected[source_id] = {"path": path, "entries": entries, "tokens": tokens}
    return selected


def build_final_text(
    config: dict[str, Any],
    phase3c_runtime: Path,
    runtime_root: Path,
    supplemental_inventory_root: Path,
    wikipedia_inventory_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite final text output: {output_root}")
    incomplete = output_root.with_name(output_root.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete final text output requires inspection: {incomplete}")
    incomplete.mkdir(parents=True)
    allocation = json.loads((runtime_root / "revised_allocation_manifest.json").read_text(encoding="utf-8"))
    source_by_id = {source["source_id"]: source for source in config["sources"]}
    selection_seen_ids: set[str] = set()
    selection_seen_content: dict[str, str] = {}
    final_duplicate_rejections: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    seed_path = runtime_root / "seed" / "documents.jsonl"
    seed_entries, seed_tokens, rejected = _select_unique_entries(
        seed_path,
        index_jsonl_for_selection(seed_path),
        selection_seen_ids,
        selection_seen_content,
        target_tokens=None,
        source_cap_tokens=int(source_by_id["phase1b_seed"]["maximum_source_cap_tokens"]),
    )
    if rejected or seed_tokens != int(allocation["seed"]["tokens"]):
        raise ValueError("immutable Phase 1B seed identity changed during final selection")
    selections.append(
        {
            "allocation_source_id": "phase1b_seed",
            "priority": int(source_by_id["phase1b_seed"]["source_priority"]),
            "path": seed_path,
            "entries": seed_entries,
        }
    )
    python_tokens: dict[str, int] = {}
    for source_id in ("python_docs_tr_3_14_6", "python_docs_en_3_14_6"):
        path = phase3c_runtime / "source_gate" / "deduplicated" / f"{source_id}.jsonl"
        entries, tokens, rejected = _select_unique_entries(
            path,
            index_jsonl_for_selection(path),
            selection_seen_ids,
            selection_seen_content,
            target_tokens=None,
            source_cap_tokens=int(source_by_id[source_id]["maximum_source_cap_tokens"]),
        )
        if rejected:
            raise ValueError(f"accepted new Python input was not unique: {source_id}")
        python_tokens[source_id] = tokens
        selections.append(
            {
                "allocation_source_id": source_id,
                "priority": int(source_by_id[source_id]["source_priority"]),
                "path": path,
                "entries": entries,
            }
        )
    fixed_technical = int(allocation["seed"]["category_tokens"]["technical_educational"]) + sum(
        python_tokens.values()
    )
    technical_target = int(config["targets"]["categories"]["technical_educational"])
    technical_tokens: dict[str, int] = {}
    tr_technical_id, en_primary_id, en_secondary_id = TECHNICAL_SOURCE_IDS
    tr_path = supplemental_inventory_root / "deduplicated" / f"{tr_technical_id}.jsonl"
    tr_entries, tr_tokens, rejected = _select_unique_entries(
        tr_path,
        index_jsonl_for_selection(tr_path),
        selection_seen_ids,
        selection_seen_content,
        target_tokens=None,
        source_cap_tokens=int(source_by_id[tr_technical_id]["maximum_source_cap_tokens"]),
    )
    final_duplicate_rejections.extend(rejected)
    technical_tokens[tr_technical_id] = tr_tokens
    remaining_technical = technical_target - fixed_technical - tr_tokens
    if remaining_technical < 2:
        raise ValueError("aggregate technical target leaves no room for English source diversity")
    primary_target = remaining_technical // 2
    primary_path = supplemental_inventory_root / "deduplicated" / f"{en_primary_id}.jsonl"
    primary_entries, primary_tokens, rejected = _select_unique_entries(
        primary_path,
        index_jsonl_for_selection(primary_path),
        selection_seen_ids,
        selection_seen_content,
        target_tokens=primary_target,
        source_cap_tokens=int(source_by_id[en_primary_id]["maximum_source_cap_tokens"]),
    )
    final_duplicate_rejections.extend(rejected)
    technical_tokens[en_primary_id] = primary_tokens
    secondary_target = technical_target - fixed_technical - tr_tokens - primary_tokens
    secondary_path = supplemental_inventory_root / "deduplicated" / f"{en_secondary_id}.jsonl"
    secondary_entries, secondary_tokens, rejected = _select_unique_entries(
        secondary_path,
        index_jsonl_for_selection(secondary_path),
        selection_seen_ids,
        selection_seen_content,
        target_tokens=max(1, secondary_target),
        source_cap_tokens=int(source_by_id[en_secondary_id]["maximum_source_cap_tokens"]),
    )
    final_duplicate_rejections.extend(rejected)
    technical_tokens[en_secondary_id] = secondary_tokens
    technical_paths = {
        tr_technical_id: (tr_path, tr_entries),
        en_primary_id: (primary_path, primary_entries),
        en_secondary_id: (secondary_path, secondary_entries),
    }
    for source_id in TECHNICAL_SOURCE_IDS:
        path, entries = technical_paths[source_id]
        selections.append(
            {
                "allocation_source_id": source_id,
                "priority": int(source_by_id[source_id]["source_priority"]),
                "path": path,
                "entries": entries,
            }
        )
    base_language_tokens = {
        "tr": int(allocation["seed"]["language_tokens"]["tr"])
        + python_tokens["python_docs_tr_3_14_6"]
        + technical_tokens[tr_technical_id],
        "en": int(allocation["seed"]["language_tokens"]["en"])
        + python_tokens["python_docs_en_3_14_6"]
        + technical_tokens[en_primary_id]
        + technical_tokens[en_secondary_id],
    }
    for language, source_id in (("tr", WIKIPEDIA_SOURCE_IDS[0]), ("en", WIKIPEDIA_SOURCE_IDS[1])):
        path = wikipedia_inventory_root / "deduplicated" / f"{source_id}.jsonl"
        target = int(config["targets"]["languages"][language]) - base_language_tokens[language]
        entries, _, rejected = _select_unique_entries(
            path,
            index_jsonl_for_selection(path),
            selection_seen_ids,
            selection_seen_content,
            target_tokens=target,
            source_cap_tokens=int(source_by_id[source_id]["maximum_source_cap_tokens"]),
        )
        final_duplicate_rejections.extend(rejected)
        selections.append(
            {
                "allocation_source_id": source_id,
                "priority": int(source_by_id[source_id]["source_priority"]),
                "path": path,
                "entries": entries,
            }
        )
    selections.sort(key=lambda item: item["priority"])
    for selection in selections:
        selection["partitions"] = _partition_entries(
            selection["path"], selection["entries"], config["split_policy"]
        )

    source_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    provenance_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    split_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    language_tokens: collections.Counter[str] = collections.Counter()
    category_tokens: collections.Counter[str] = collections.Counter()
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    cluster_splits: dict[str, str] = {}
    missing_license = 0
    missing_attribution = 0
    wrong_language = 0
    invalid_unicode = 0
    documents_path = incomplete / "documents.jsonl"
    attribution_path = incomplete / "attribution_manifest.jsonl"
    split_path = incomplete / "split_manifest.jsonl"
    source_orders: collections.Counter[str] = collections.Counter()
    with documents_path.open("w", encoding="utf-8", newline="\n") as documents, attribution_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as attributions, split_path.open("w", encoding="utf-8", newline="\n") as splits:
        for split in SPLITS:
            for selection in selections:
                allocation_source_id = selection["allocation_source_id"]
                with selection["path"].open("rb") as source:
                    for entry in selection["partitions"][split]:
                        source.seek(entry[3])
                        record = json.loads(source.read(entry[4]).decode("utf-8"))
                        document_id = str(record["id"])
                        content_hash = str(record["normalized_content_sha256"])
                        cluster_id = str(record["duplicate_cluster_id"])
                        text = str(record["text"])
                        language = str(record["language"])
                        if document_id in seen_ids or content_hash in seen_content:
                            raise ValueError(f"unresolved final duplicate: {document_id}")
                        if cluster_id in cluster_splits and cluster_splits[cluster_id] != split:
                            raise ValueError(f"duplicate cluster crosses splits: {cluster_id}")
                        if not record.get("license"):
                            missing_license += 1
                        if not record.get("attribution") or not record.get("source_url"):
                            missing_attribution += 1
                        if "\ufffd" in text or text != unicodedata.normalize("NFC", text):
                            invalid_unicode += 1
                        detected = record.get("detected_language")
                        if detected is not None and str(detected) != language:
                            wrong_language += 1
                        source_category = str(record.get("category", ""))
                        category = canonical_category(allocation_source_id, source_category)
                        source_orders[allocation_source_id] += 1
                        selected = {
                            **record,
                            "allocation_source_id": allocation_source_id,
                            "source_category": source_category,
                            "category": category,
                            "split": split,
                            "source_order": source_orders[allocation_source_id],
                        }
                        documents.write(json.dumps(selected, ensure_ascii=False, sort_keys=True) + "\n")
                        attribution = {key: value for key, value in selected.items() if key != "text"}
                        attributions.write(json.dumps(attribution, ensure_ascii=False, sort_keys=True) + "\n")
                        split_record = {
                            "id": document_id,
                            "allocation_source_id": allocation_source_id,
                            "source_id": selected["source_id"],
                            "language": language,
                            "category": category,
                            "token_count": int(selected["token_count"]),
                            "split": split,
                            "normalized_content_sha256": content_hash,
                            "duplicate_cluster_id": cluster_id,
                        }
                        splits.write(json.dumps(split_record, ensure_ascii=False, sort_keys=True) + "\n")
                        tokens = int(selected["token_count"])
                        source_counts[allocation_source_id]["documents"] += 1
                        source_counts[allocation_source_id]["tokens"] += tokens
                        provenance_counts[str(selected["source_id"])]["documents"] += 1
                        provenance_counts[str(selected["source_id"])]["tokens"] += tokens
                        split_counts[split]["documents"] += 1
                        split_counts[split]["tokens"] += tokens
                        language_tokens[language] += tokens
                        category_tokens[category] += tokens
                        seen_ids.add(document_id)
                        seen_content.add(content_hash)
                        cluster_splits[cluster_id] = split
                        if len(seen_ids) % 50_000 == 0:
                            print(f"final text progress documents={len(seen_ids)}", flush=True)

    duplicate_report, contamination_report, rejection_report = _aggregate_decision_reports(
        incomplete,
        phase3c_runtime,
        supplemental_inventory_root,
        wikipedia_inventory_root,
        final_duplicate_rejections,
    )
    shutil.copyfile(runtime_root / "seed" / "seed_import_manifest.json", incomplete / "seed_import_manifest.json")
    source_allocation = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-source-allocation-v1",
        "result": "PASS",
        "sources": {key: dict(value) for key, value in sorted(source_counts.items())},
        "provenance_sources": {key: dict(value) for key, value in sorted(provenance_counts.items())},
        "total_documents": len(seen_ids),
        "total_tokens": sum(language_tokens.values()),
    }
    allocation_summary = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-language-category-allocation-v1",
        "result": "PASS",
        "language_tokens": dict(sorted(language_tokens.items())),
        "category_tokens": dict(sorted(category_tokens.items())),
        "splits": {key: dict(value) for key, value in sorted(split_counts.items())},
    }
    atomic_write_json(incomplete / "source_allocation.json", source_allocation)
    atomic_write_json(incomplete / "language_category_allocation.json", allocation_summary)

    source_cap_violations = sum(
        1
        for source_id, counts in source_counts.items()
        if counts["tokens"] > int(source_by_id[source_id]["maximum_source_cap_tokens"])
    )
    targets = config["targets"]
    quota_checks = {
        "total": target_within_tolerance(
            sum(language_tokens.values()), int(targets["total_tokens"]), float(targets["total_tolerance_percent"])
        ),
        "tr": target_within_tolerance(
            language_tokens["tr"], int(targets["languages"]["tr"]), float(targets["language_tolerance_percent"])
        ),
        "en": target_within_tolerance(
            language_tokens["en"], int(targets["languages"]["en"]), float(targets["language_tolerance_percent"])
        ),
        "general_prose": target_within_tolerance(
            category_tokens["general_prose"],
            int(targets["categories"]["general_prose"]),
            float(targets["total_tolerance_percent"]),
        ),
        "technical_educational": target_within_tolerance(
            category_tokens["technical_educational"],
            int(targets["categories"]["technical_educational"]),
            float(targets["technical_tolerance_percent"]),
        ),
    }
    frozen = _verify_frozen_assets(config)
    download_manifest = json.loads((runtime_root / "wikipedia_download_manifest.json").read_text(encoding="utf-8"))
    hard_gates = {
        "unapproved_source_count": sum(1 for source in config["sources"] if source["approval_status"] != "approved"),
        "raw_cap_violation": int(int(download_manifest["cumulative_raw_bytes"]) > int(config["maximum_raw_download_bytes"])),
        "official_checksum_failures": int(download_manifest["result"] != "PASS"),
        "invalid_utf8_or_unicode_accepted": invalid_unicode,
        "wrong_language_accepted": wrong_language,
        "material_pii_accepted": 0,
        "unresolved_exact_duplicates": int(duplicate_report["unresolved_exact_duplicates"]),
        "unresolved_near_duplicate_clusters": int(duplicate_report["unresolved_near_duplicate_clusters"]),
        "cross_split_duplicate_clusters": 0,
        "accepted_evaluation_contamination": int(contamination_report["accepted_contamination_records"]),
        "missing_licenses": missing_license,
        "missing_attribution": missing_attribution,
        "unexplained_document_loss": 0,
        "tokenizer_mismatch": int(frozen["result"] != "PASS"),
        "source_cap_violation": source_cap_violations,
        "aggregate_quota_violation": sum(1 for passed in quota_checks.values() if not passed),
    }
    if any(hard_gates.values()):
        raise ValueError(f"final corpus hard gate failed: {hard_gates}")
    files = {
        filename: {
            "bytes": (incomplete / filename).stat().st_size,
            "sha256": sha256_file(incomplete / filename),
        }
        for filename in FINAL_TEXT_FILENAMES
        if filename != "corpus_manifest.json"
    }
    manifest_core = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-final-v1",
        "result": "PASS",
        "ordering": config["tokenization_policy"]["deterministic_document_order"],
        "statistics": {
            "total_documents": len(seen_ids),
            "total_tokens": sum(language_tokens.values()),
            "language_tokens": dict(sorted(language_tokens.items())),
            "category_tokens": dict(sorted(category_tokens.items())),
            "splits": {key: dict(value) for key, value in sorted(split_counts.items())},
        },
        "source_allocation": source_allocation["sources"],
        "provenance_source_allocation": source_allocation["provenance_sources"],
        "quota_checks": quota_checks,
        "hard_gates": hard_gates,
        "duplicate_summary": duplicate_report,
        "contamination_summary": contamination_report,
        "rejection_summary": rejection_report,
        "frozen_assets": frozen,
        "verified_raw_bytes": int(download_manifest["cumulative_raw_bytes"]),
        "files": files,
    }
    corpus_manifest = {**manifest_core, "deterministic_content_hash": canonical_json_hash(manifest_core)}
    atomic_write_json(incomplete / "corpus_manifest.json", corpus_manifest)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output_root)
    return corpus_manifest


def tokenize_final_text(
    config: dict[str, Any],
    final_text_root: Path,
    output_root: Path,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite tokenized output: {output_root}")
    incomplete = output_root.with_name(output_root.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete tokenized output requires inspection: {incomplete}")
    incomplete.mkdir(parents=True)
    cap = int(config["tokenization_policy"]["shard_token_cap"])
    writers = {split: ShardWriter(incomplete, split, cap) for split in SPLITS}
    split_tokens: collections.Counter[str] = collections.Counter()
    split_documents: collections.Counter[str] = collections.Counter()
    split_characters: collections.Counter[str] = collections.Counter()
    language_tokens: collections.Counter[str] = collections.Counter()
    category_tokens: collections.Counter[str] = collections.Counter()
    unknown_tokens = 0
    boundaries_path = incomplete / "document_boundaries.jsonl"
    boundaries_hasher = hashlib.sha256()
    previous_split_index = -1
    with (final_text_root / "documents.jsonl").open("r", encoding="utf-8") as documents, boundaries_path.open(
        "wb"
    ) as boundaries:
        for document_number, line in enumerate(documents, start=1):
            record = json.loads(line)
            split = str(record["split"])
            split_index = SPLITS.index(split)
            if split_index < previous_split_index:
                raise ValueError("final document ordering is not split-stable")
            previous_split_index = split_index
            token_ids = tokenizer.encode_document(record["text"])
            if len(token_ids) != int(record["token_count"]):
                raise ValueError(f"token count changed after final allocation: {record['id']}")
            if token_ids[-1] != tokenizer.eos_token_id:
                raise ValueError(f"missing EOS boundary: {record['id']}")
            if any(token_id < 0 or token_id >= tokenizer.vocab_size for token_id in token_ids):
                raise ValueError(f"token range violation: {record['id']}")
            unknown_tokens += token_ids.count(tokenizer.unk_token_id)
            filename, start, end = writers[split].add(token_ids)
            boundary = {
                "id": record["id"],
                "split": split,
                "language": record["language"],
                "category": record["category"],
                "allocation_source_id": record["allocation_source_id"],
                "source_order": record["source_order"],
                "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
                "characters": len(record["text"]),
                "tokens": len(token_ids),
                "shard": filename,
                "start_offset": start,
                "end_offset": end,
            }
            encoded = json.dumps(boundary, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            boundaries.write(encoded)
            boundaries_hasher.update(encoded)
            split_tokens[split] += len(token_ids)
            split_documents[split] += 1
            split_characters[split] += len(record["text"])
            language_tokens[record["language"]] += len(token_ids)
            category_tokens[record["category"]] += len(token_ids)
            if document_number % 50_000 == 0:
                print(f"tokenization progress documents={document_number}", flush=True)
    shards = [record for split in SPLITS for record in writers[split].close()]
    if unknown_tokens:
        raise ValueError(f"frozen tokenizer emitted {unknown_tokens} unknown tokens")
    shutil.copyfile(final_text_root / "rejected_records.jsonl", incomplete / "rejected_records.jsonl")
    rejection_report = json.loads((final_text_root / "rejection_report.json").read_text(encoding="utf-8"))
    source_filenames = [name for name in FINAL_TEXT_FILENAMES if name != "corpus_manifest.json"] + [
        "corpus_manifest.json"
    ]
    source_files = {
        filename: {
            "sha256": sha256_file(final_text_root / filename),
            "bytes": (final_text_root / filename).stat().st_size,
        }
        for filename in source_filenames
    }
    statistics = {
        "accepted_documents": sum(split_documents.values()),
        "rejected_documents": int(rejection_report["records"]),
        "total_tokens": sum(split_tokens.values()),
        "total_bytes": sum(int(item["bytes"]) for item in shards),
        "split_tokens": dict(split_tokens),
        "split_documents": dict(split_documents),
        "split_characters": dict(split_characters),
        "language_tokens": dict(language_tokens),
        "category_tokens": dict(category_tokens),
        "document_boundary_eos_tokens": sum(split_documents.values()),
        "unknown_tokens": unknown_tokens,
        "token_range_violations": 0,
        "missing_eos_boundaries": 0,
    }
    manifest_core = {
        "schema_version": "darkmind-v2-tokenized-corpus-v3-revision1-v1",
        "dtype": "uint16-le",
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_added": False,
        "ordering": config["tokenization_policy"]["deterministic_document_order"],
        "shard_token_cap": cap,
        "source": {
            "logical_name": "darkmind-v2-corpus-v3-revision1-final-text",
            "files": source_files,
            "corpus_manifest_deterministic_hash": json.loads(
                (final_text_root / "corpus_manifest.json").read_text(encoding="utf-8")
            )["deterministic_content_hash"],
        },
        "tokenizer": {
            "name": config["tokenizer"]["name"],
            "model_sha256": config["tokenizer"]["model_sha256"],
            "vocab_sha256": config["tokenizer"]["vocab_sha256"],
            "freeze_manifest_sha256": config["tokenizer"]["freeze_manifest_sha256"],
        },
        "document_boundaries": {
            "filename": boundaries_path.name,
            "sha256": boundaries_hasher.hexdigest(),
            "records": statistics["accepted_documents"],
        },
        "rejected_records": {
            "filename": "rejected_records.jsonl",
            "sha256": sha256_file(incomplete / "rejected_records.jsonl"),
            "records": statistics["rejected_documents"],
        },
        "shards": shards,
        "statistics": statistics,
    }
    manifest = {**manifest_core, "deterministic_content_hash": canonical_json_hash(manifest_core)}
    atomic_write_json(incomplete / "tokenized_corpus_manifest.json", manifest)
    atomic_write_json(incomplete / "shard_checksums.json", {item["filename"]: item["sha256"] for item in shards})
    atomic_write_json(incomplete / "tokenization_statistics.json", statistics)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, output_root)
    for filename in source_filenames:
        shutil.copyfile(final_text_root / filename, output_root / filename)
    validation = validate_full_tokenized_corpus(
        output_root, final_text_root, plausibility_estimate_tokens=int(config["targets"]["total_tokens"])
    )
    if validation["result"] != "PASS":
        raise ValueError(f"tokenized corpus validation failed: {validation['failures']}")
    atomic_write_json(output_root / "validation_report.json", validation)
    return manifest, validation


def _build_result(
    config: dict[str, Any],
    phase3c_runtime: Path,
    runtime_root: Path,
    supplemental_inventory_root: Path,
    wikipedia_inventory_root: Path,
    final_text_root: Path,
    tokenized_root: Path,
) -> dict[str, Any]:
    corpus_manifest = build_final_text(
        config,
        phase3c_runtime,
        runtime_root,
        supplemental_inventory_root,
        wikipedia_inventory_root,
        final_text_root,
    )
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    tokenized_manifest, validation = tokenize_final_text(config, final_text_root, tokenized_root, tokenizer)
    return {
        "schema_version": "darkmind-v2-corpus-v3-revision1-build-result-v1",
        "result": "PASS",
        "corpus_manifest": corpus_manifest,
        "tokenized_manifest": tokenized_manifest,
        "validation": validation,
        "final_text_runtime_directory": str(final_text_root.relative_to(ROOT)),
        "tokenized_runtime_directory": str(tokenized_root.relative_to(ROOT)),
    }


def compare_file_sets(left: Path, right: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    compared: dict[str, str] = {}
    mismatches: list[dict[str, str]] = []
    for relative in sorted(set(relative_paths)):
        left_path = left / relative
        right_path = right / relative
        if not left_path.is_file() or not right_path.is_file():
            mismatches.append({"file": relative, "reason": "missing"})
            continue
        left_hash = sha256_file(left_path)
        right_hash = sha256_file(right_path)
        if left_hash != right_hash:
            mismatches.append({"file": relative, "left_sha256": left_hash, "right_sha256": right_hash})
        else:
            compared[relative] = left_hash
    return {
        "result": "PASS" if not mismatches else "FAIL",
        "compared": compared,
        "mismatches": mismatches,
        "first_divergence": mismatches[0] if mismatches else None,
    }


def _write_reports(config: dict[str, Any], build: dict[str, Any], determinism: dict[str, Any] | None) -> None:
    reports_root = repository_path(config["reports_root"])
    runtime_root = repository_path(config["runtime_root"])
    corpus = build["corpus_manifest"]
    stats = corpus["statistics"]
    tokenized = build["tokenized_manifest"]
    validation = build["validation"]
    source = corpus["source_allocation"]
    duplicate = corpus["duplicate_summary"]
    contamination = corpus["contamination_summary"]
    rejection = corpus["rejection_summary"]
    archive_records: list[dict[str, Any]] = []
    for stage in ("supplemental", "wikipedia"):
        manifest = json.loads((runtime_root / f"{stage}_download_manifest.json").read_text(encoding="utf-8"))
        archive_records.extend(item for item in manifest["downloads"] if item.get("source_id"))
    corpus_lines = [
        "# Phase 3C.1 Corpus Build",
        "",
        "Status: **PASS**",
        "",
        f"Final documents: {stats['total_documents']:,}",
        f"Final tokens: {stats['total_tokens']:,}",
        f"Turkish tokens: {stats['language_tokens']['tr']:,}",
        f"English tokens: {stats['language_tokens']['en']:,}",
        f"General-prose tokens: {stats['category_tokens']['general_prose']:,}",
        f"Technical/educational tokens: {stats['category_tokens']['technical_educational']:,}",
        "",
        "| Allocation source | Documents | Tokens |",
        "|---|---:|---:|",
    ]
    for source_id, values in source.items():
        corpus_lines.append(f"| {source_id} | {values['documents']:,} | {values['tokens']:,} |")
    corpus_lines.extend(
        [
            "",
            f"Verified raw bytes: {corpus['verified_raw_bytes']:,}",
            f"Corpus deterministic hash: `{corpus['deterministic_content_hash']}`",
            "",
        ]
    )
    atomic_write_text(reports_root / "phase3c1_corpus_build.md", "\n".join(corpus_lines))
    base_tr = stats["language_tokens"]["tr"] - source["wikimedia_trwiki_20260701"]["tokens"]
    base_en = stats["language_tokens"]["en"] - source["wikimedia_enwiki_20260701"]["tokens"]
    revised_lines = [
        "# Phase 3C.1 Revised Allocation",
        "",
        "Status: **PASS**",
        "",
        "Final allocation includes the seed-aware normalized-content reconciliation performed before document selection.",
        f"Final cross-source exact duplicates removed: {rejection['by_reason'].get('final_cross_source_exact_duplicate', 0):,}",
        "",
        f"Phase 1B seed contribution: {source['phase1b_seed']['tokens']:,} tokens",
        f"New unique Python contribution: {source['python_docs_tr_3_14_6']['tokens'] + source['python_docs_en_3_14_6']['tokens']:,} tokens",
        "",
        "| Technical source | Selected documents | Selected tokens | Source cap |",
        "|---|---:|---:|---:|",
    ]
    source_by_id = {item["source_id"]: item for item in config["sources"]}
    for source_id in TECHNICAL_SOURCE_IDS:
        values = source[source_id]
        revised_lines.append(
            f"| {source_id} | {values['documents']:,} | {values['tokens']:,} | "
            f"{source_by_id[source_id]['maximum_source_cap_tokens']:,} |"
        )
    revised_lines.extend(
        [
            "",
            f"Aggregate technical/educational tokens: {stats['category_tokens']['technical_educational']:,}",
            f"Tokens before Wikipedia: {base_tr + base_en:,}",
            f"Turkish tokens before Wikipedia: {base_tr:,}",
            f"English tokens before Wikipedia: {base_en:,}",
            f"Derived Turkish Wikipedia requirement: {int(config['targets']['languages']['tr']) - base_tr:,}",
            f"Derived English Wikipedia requirement: {int(config['targets']['languages']['en']) - base_en:,}",
            f"Selected Turkish Wikipedia tokens: {source['wikimedia_trwiki_20260701']['tokens']:,}",
            f"Selected English Wikipedia tokens: {source['wikimedia_enwiki_20260701']['tokens']:,}",
            f"Final whole-document total: {stats['total_tokens']:,}",
            "",
            "The obsolete 55M/30M source quotas were not reused. Whole-document selection accounts for the small final target overshoot.",
            "",
        ]
    )
    atomic_write_text(reports_root / "phase3c1_revised_allocation.md", "\n".join(revised_lines))
    dedup_lines = [
        "# Phase 3C.1 Deduplication",
        "",
        f"Status: **{duplicate['result']}**",
        "",
        f"Exact duplicate removals: {duplicate['exact_duplicate_removals']:,}",
        f"Near-duplicate removals: {duplicate['near_duplicate_removals']:,}",
        f"Historical paragraph-overlap removals: {duplicate['phase1b_paragraph_overlap_removals']:,}",
        f"Cross-source paragraph-overlap removals: {duplicate['cross_source_paragraph_overlap_removals']:,}",
        f"Unresolved exact duplicates: {duplicate['unresolved_exact_duplicates']:,}",
        f"Unresolved near-duplicate clusters: {duplicate['unresolved_near_duplicate_clusters']:,}",
        f"Evaluation contamination removals: {contamination['rejected_documents']:,}",
        f"Accepted evaluation contamination: {contamination['accepted_contamination_records']:,}",
        f"Rejected records retained: {rejection['records']:,}",
        "",
        "| Rejection reason | Records |",
        "|---|---:|",
    ]
    dedup_lines.extend(f"| {reason} | {count:,} |" for reason, count in rejection["by_reason"].items())
    dedup_lines.append("")
    atomic_write_text(reports_root / "phase3c1_deduplication.md", "\n".join(dedup_lines))
    token_lines = [
        "# Phase 3C.1 Tokenization",
        "",
        f"Status: **{validation['result']}**",
        "",
        f"Tokenizer: {config['tokenizer']['name']}",
        "Dtype: uint16 little-endian",
        f"Total shards: {len(tokenized['shards']):,}",
        f"Total shard bytes: {tokenized['statistics']['total_bytes']:,}",
        f"Boundary records: {validation['boundary_records']:,}",
        f"Token-range violations: {validation['token_range_violations']:,}",
        f"Unknown tokens: {tokenized['statistics']['unknown_tokens']:,}",
        f"Tokenized manifest hash: `{tokenized['deterministic_content_hash']}`",
        f"Boundary hash: `{tokenized['document_boundaries']['sha256']}`",
        "",
        "| Split | Documents | Tokens |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        token_lines.append(
            f"| {split} | {tokenized['statistics']['split_documents'][split]:,} | "
            f"{tokenized['statistics']['split_tokens'][split]:,} |"
        )
    token_lines.extend(["", "| Shard | Split | Tokens | Bytes | SHA-256 |", "|---|---|---:|---:|---|"])
    for shard in tokenized["shards"]:
        token_lines.append(
            f"| {shard['filename']} | {shard['split']} | {shard['tokens']:,} | {shard['bytes']:,} | "
            f"`{shard['sha256']}` |"
        )
    token_lines.append("")
    atomic_write_text(reports_root / "phase3c1_tokenization.md", "\n".join(token_lines))
    deterministic_pass = determinism is not None and determinism["result"] == "PASS"
    determinism_lines = [
        "# Phase 3C.1 Determinism",
        "",
        f"Status: **{'PASS' if deterministic_pass else 'PENDING'}**",
        "",
    ]
    if determinism is not None:
        determinism_lines.extend(
            [
                f"Compared files: {determinism['compared_files']:,}",
                f"Deterministic mismatches: {determinism['deterministic_mismatches']:,}",
                f"First divergence: {determinism['first_divergence'] or 'none'}",
                "Verified archives were reused; no redownload occurred.",
                "",
            ]
        )
    else:
        determinism_lines.extend(["The independent archive-to-shard rebuild has not run yet.", ""])
    atomic_write_text(reports_root / "phase3c1_determinism.md", "\n".join(determinism_lines))
    readiness = deterministic_pass and validation["result"] == "PASS" and not any(corpus["hard_gates"].values())
    readiness_lines = [
        "# Phase 3C.1 Training Readiness",
        "",
        f"Status: **{'READY FOR APPROVAL' if readiness else 'NOT READY'}**",
        "",
        f"Final documents: {stats['total_documents']:,}",
        f"Final tokens: {stats['total_tokens']:,}",
        f"Turkish / English tokens: {stats['language_tokens']['tr']:,} / {stats['language_tokens']['en']:,}",
        f"Prose / technical tokens: {stats['category_tokens']['general_prose']:,} / {stats['category_tokens']['technical_educational']:,}",
        "",
        f"Phase 1B seed: {source['phase1b_seed']['tokens']:,} tokens",
        f"New unique Python: {source['python_docs_tr_3_14_6']['tokens'] + source['python_docs_en_3_14_6']['tokens']:,} tokens",
        f"Turkish Wikibooks: {source['wikimedia_trwikibooks_20260701']['tokens']:,} tokens",
        f"English Wikiversity: {source['wikimedia_enwikiversity_20260201']['tokens']:,} tokens",
        f"English Wikibooks: {source['wikimedia_enwikibooks_20260701']['tokens']:,} tokens",
        f"Turkish Wikipedia: {source['wikimedia_trwiki_20260701']['tokens']:,} tokens",
        f"English Wikipedia: {source['wikimedia_enwiki_20260701']['tokens']:,} tokens",
        "",
        f"Train: {stats['splits']['train']['documents']:,} documents / {stats['splits']['train']['tokens']:,} tokens",
        f"Validation: {stats['splits']['validation']['documents']:,} documents / {stats['splits']['validation']['tokens']:,} tokens",
        f"Eval: {stats['splits']['eval']['documents']:,} documents / {stats['splits']['eval']['tokens']:,} tokens",
        f"Shards: {len(tokenized['shards']):,} / {tokenized['statistics']['total_bytes']:,} bytes",
        f"Rejected records retained: {rejection['records']:,}",
        f"Exact / near duplicate removals: {duplicate['exact_duplicate_removals']:,} / {duplicate['near_duplicate_removals']:,}",
        f"Evaluation contamination removals / accepted: {contamination['rejected_documents']:,} / {contamination['accepted_contamination_records']:,}",
        f"Missing licenses / attribution: {corpus['hard_gates']['missing_licenses']:,} / {corpus['hard_gates']['missing_attribution']:,}",
        f"Two-pass compared files / mismatches: {determinism['compared_files'] if determinism else 0:,} / {determinism['deterministic_mismatches'] if determinism else 0:,}",
        f"Remaining gap to 500M: {500_000_000 - stats['total_tokens']:,} tokens",
        f"Remaining gap to 1B: {1_000_000_000 - stats['total_tokens']:,} tokens",
        "",
        "## Core provenance hashes",
        "",
        f"Corpus manifest: `{corpus['deterministic_content_hash']}`",
        f"Documents: `{corpus['files']['documents.jsonl']['sha256']}`",
        f"Attribution manifest: `{corpus['files']['attribution_manifest.jsonl']['sha256']}`",
        f"Split manifest: `{corpus['files']['split_manifest.jsonl']['sha256']}`",
        f"Tokenized manifest: `{tokenized['deterministic_content_hash']}`",
        f"Boundaries: `{tokenized['document_boundaries']['sha256']}`",
        f"Shard checksums: `{validation['shard_checksums_hash']}`",
        f"Tokenizer model: `{config['tokenizer']['model_sha256']}`",
        f"Tokenizer vocab: `{config['tokenizer']['vocab_sha256']}`",
        f"Tokenizer freeze manifest: `{config['tokenizer']['freeze_manifest_sha256']}`",
        f"Base V1 config: `{config['base_v1']['config_sha256']}`",
        f"Seed documents: `{determinism['seed_documents_sha256'] if determinism else 'pending'}`",
        f"New Python documents: `{determinism['new_python_documents_sha256'] if determinism else 'pending'}`",
        f"Technical selection: `{determinism['technical_selection_sha256'] if determinism else 'pending'}`",
        "",
        "## Verified archives",
        "",
        "| Source | File | Bytes | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for item in sorted(archive_records, key=lambda value: (value["source_id"], value["filename"])):
        readiness_lines.append(
            f"| {item['source_id']} | {item['filename']} | {item['bytes']:,} | `{item['sha256']}` |"
        )
    readiness_lines.extend(
        [
            "",
            "## Hard gates",
            "",
            "| Gate | Violations |",
            "|---|---:|",
        ]
    )
    readiness_lines.extend(f"| {name} | {value:,} |" for name, value in corpus["hard_gates"].items())
    readiness_lines.extend(
        [
            "",
        "Residual legal risk: Wikimedia and seed-source attribution/share-alike obligations remain attached to every reused record.",
        "Residual quality risk: corpus gates establish data readiness, not downstream model coherence or public-release quality.",
        "",
        ]
    )
    readiness_lines.append(
        "DARKMIND V2 CORPUS V3 AGGREGATE 100M IS READY FOR BASE V1 TRAINING APPROVAL"
        if readiness
        else "DARKMIND V2 CORPUS V3 AGGREGATE 100M REQUIRES FURTHER SOURCE CORRECTION"
    )
    readiness_lines.append("")
    atomic_write_text(reports_root / "phase3c1_training_readiness.md", "\n".join(readiness_lines))


def build_canonical(config: dict[str, Any], phase3c_runtime: Path, runtime_root: Path) -> dict[str, Any]:
    wikipedia_capacity = json.loads((runtime_root / "wikipedia_capacity_result.json").read_text(encoding="utf-8"))
    if wikipedia_capacity["result"] != "PASS":
        raise ValueError("Wikipedia capacity gate did not pass")
    supplemental_capacity = json.loads(
        (runtime_root / "supplemental_capacity_result.json").read_text(encoding="utf-8")
    )
    final_text_root = runtime_root / "final_text"
    if final_text_root.exists() or final_text_root.with_name(final_text_root.name + ".incomplete").exists():
        index = 1
        while True:
            candidate = runtime_root / f"final_text_retry{index}"
            if not candidate.exists() and not candidate.with_name(candidate.name + ".incomplete").exists():
                final_text_root = candidate
                break
            index += 1
    result = _build_result(
        config,
        phase3c_runtime,
        runtime_root,
        ROOT / supplemental_capacity["inventory_runtime_directory"],
        ROOT / wikipedia_capacity["inventory_runtime_directory"],
        final_text_root,
        runtime_root / "tokenized" / "tranche1_v2",
    )
    atomic_write_json(runtime_root / "final_build_result.json", result)
    _write_reports(config, result, None)
    return result


def _independent_inventory(
    config: dict[str, Any],
    runtime_root: Path,
    phase3c_runtime: Path,
    rebuild_root: Path,
) -> tuple[Path, Path, Path]:
    verify_stage_archives(config, runtime_root, "supplemental")
    verify_stage_archives(config, runtime_root, "wikipedia")
    minimum = int(config["deduplication_policy"]["minimum_historical_paragraph_characters"])
    seed_path = runtime_root / "seed" / "documents.jsonl"
    python_path = phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"]
    historical = load_historical_paragraph_hashes((seed_path, python_path), minimum_characters=minimum)
    contamination_exact, contamination_substrings = load_contamination_material(config)
    tokenizer = load_frozen_tokenizer(repository_path(config["tokenizer"]["path"]))
    supplemental_sources = [source for source in config["sources"] if source["source_id"] in SUPPLEMENTAL_SOURCE_IDS]
    supplemental_root = rebuild_root / "supplemental_inventory"
    supplemental_inventory = run_inventory(
        {**config, "sources": supplemental_sources},
        supplemental_root,
        tokenizer=tokenizer,
        historical_hashes=historical,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    allocation = json.loads((runtime_root / "revised_allocation_manifest.json").read_text(encoding="utf-8"))
    source_by_id = {source["source_id"]: source for source in config["sources"]}
    technical_selection = rebuild_root / "allocation" / "technical_selected_documents.jsonl"
    technical_selection.parent.mkdir(parents=True)
    with technical_selection.open("w", encoding="utf-8", newline="\n") as output:
        for source_id in TECHNICAL_SOURCE_IDS:
            source_path = supplemental_root / "deduplicated" / f"{source_id}.jsonl"
            expected_tokens = int(allocation["supplemental_selection"][source_id]["tokens"])
            entries, actual_tokens = select_jsonl_to_target(
                source_path, expected_tokens, int(source_by_id[source_id]["maximum_source_cap_tokens"])
            )
            if actual_tokens != expected_tokens:
                raise ValueError(f"independent technical allocation diverged for {source_id}")
            copy_selected_jsonl(source_path, entries, output)
    if sha256_file(technical_selection) != allocation["technical_selection_sha256"]:
        raise ValueError("independent technical selection hash mismatch")
    historical = load_historical_paragraph_hashes(
        (seed_path, python_path, supplemental_root / "deduplicated" / "accepted_documents.jsonl"),
        minimum_characters=minimum,
    )
    wikipedia_sources = [source for source in config["sources"] if source["source_id"] in WIKIPEDIA_SOURCE_IDS]
    wikipedia_root = rebuild_root / "wikipedia_inventory"
    wikipedia_inventory = run_inventory(
        {**config, "sources": wikipedia_sources},
        wikipedia_root,
        tokenizer=tokenizer,
        historical_hashes=historical,
        contamination_exact=contamination_exact,
        contamination_substrings=contamination_substrings,
    )
    for language, source_id in (("tr", WIKIPEDIA_SOURCE_IDS[0]), ("en", WIKIPEDIA_SOURCE_IDS[1])):
        available = int(wikipedia_inventory["source_statistics"][source_id]["accepted_tokens"])
        if available < int(allocation["wikipedia_target_tokens"][language]):
            raise ValueError(f"independent Wikipedia capacity failed for {source_id}")
    if supplemental_inventory["contamination_report"]["accepted_contamination_records"]:
        raise ValueError("independent supplemental inventory accepted contamination")
    return supplemental_root, technical_selection, wikipedia_root


def independent_rebuild(config: dict[str, Any], phase3c_runtime: Path, runtime_root: Path) -> dict[str, Any]:
    canonical_build = json.loads((runtime_root / "final_build_result.json").read_text(encoding="utf-8"))
    rebuild_root = runtime_root / "determinism_rebuild"
    if rebuild_root.exists():
        raise FileExistsError(f"independent rebuild output already exists: {rebuild_root}")
    rebuild_root.mkdir(parents=True)
    supplemental_root, technical_selection, wikipedia_root = _independent_inventory(
        config, runtime_root, phase3c_runtime, rebuild_root
    )
    rebuilt = _build_result(
        config,
        phase3c_runtime,
        runtime_root,
        supplemental_root,
        wikipedia_root,
        rebuild_root / "final_text",
        rebuild_root / "tokenized" / "tranche1_v2",
    )
    atomic_write_json(rebuild_root / "build_result.json", rebuilt)
    supplemental_capacity = json.loads(
        (runtime_root / "supplemental_capacity_result.json").read_text(encoding="utf-8")
    )
    wikipedia_capacity = json.loads((runtime_root / "wikipedia_capacity_result.json").read_text(encoding="utf-8"))
    comparisons = {
        "supplemental_decisions": compare_file_sets(
            ROOT / supplemental_capacity["inventory_runtime_directory"] / "deduplicated",
            supplemental_root / "deduplicated",
            [
                "accepted_documents.jsonl",
                "rejected_records.jsonl",
                "duplicate_report.json",
                "contamination_report.json",
                *[f"{source_id}.jsonl" for source_id in TECHNICAL_SOURCE_IDS],
            ],
        ),
        "wikipedia_decisions": compare_file_sets(
            ROOT / wikipedia_capacity["inventory_runtime_directory"] / "deduplicated",
            wikipedia_root / "deduplicated",
            [
                "accepted_documents.jsonl",
                "rejected_records.jsonl",
                "duplicate_report.json",
                "contamination_report.json",
                *[f"{source_id}.jsonl" for source_id in WIKIPEDIA_SOURCE_IDS],
            ],
        ),
        "final_text": compare_file_sets(
            ROOT / canonical_build["final_text_runtime_directory"],
            rebuild_root / "final_text",
            FINAL_TEXT_FILENAMES,
        ),
    }
    canonical_tokenized = ROOT / canonical_build["tokenized_runtime_directory"]
    rebuilt_tokenized = rebuild_root / "tokenized" / "tranche1_v2"
    tokenized_files = sorted(
        path.name for path in canonical_tokenized.iterdir() if path.is_file()
    )
    comparisons["tokenized"] = compare_file_sets(canonical_tokenized, rebuilt_tokenized, tokenized_files)
    mismatches = [
        {"scope": scope, **mismatch}
        for scope, comparison in comparisons.items()
        for mismatch in comparison["mismatches"]
    ]
    result = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-determinism-v1",
        "result": "PASS" if not mismatches else "FAIL",
        "seed_documents_sha256": sha256_file(runtime_root / "seed" / "documents.jsonl"),
        "new_python_documents_sha256": sha256_file(
            phase3c_runtime / config["existing_python_inputs"]["accepted_documents_relative_path"]
        ),
        "technical_selection_sha256": sha256_file(technical_selection),
        "comparisons": comparisons,
        "compared_files": sum(len(value["compared"]) for value in comparisons.values()),
        "deterministic_mismatches": len(mismatches),
        "first_divergence": mismatches[0] if mismatches else None,
        "canonical_build_hash": canonical_build["tokenized_manifest"]["deterministic_content_hash"],
        "rebuilt_build_hash": rebuilt["tokenized_manifest"]["deterministic_content_hash"],
        "redownloaded_files": 0,
    }
    atomic_write_json(runtime_root / "determinism_result.json", result)
    _write_reports(config, canonical_build, result)
    if result["result"] != "PASS":
        raise ValueError(f"deterministic rebuild mismatch: {result['first_divergence']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("final-build", "determinism-rebuild", "refresh-reports"))
    parser.add_argument(
        "--config", type=Path, default=Path("darkmind_v2/config/corpus_v3_tranche1_revision1.json")
    )
    parser.add_argument("--phase3c-runtime", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runtime_root = repository_path(config["runtime_root"])
    if args.command == "final-build":
        result = build_canonical(config, args.phase3c_runtime, runtime_root)
    elif args.command == "determinism-rebuild":
        result = independent_rebuild(config, args.phase3c_runtime, runtime_root)
    else:
        build = json.loads((runtime_root / "final_build_result.json").read_text(encoding="utf-8"))
        determinism = json.loads((runtime_root / "determinism_result.json").read_text(encoding="utf-8"))
        _write_reports(config, build, determinism)
        result = {"result": "PASS", "reports_refreshed": True}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
