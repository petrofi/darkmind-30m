"""Relocate immutable inputs and diagnose Corpus V3 sequence ordering for Phase 4B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, SPECIAL_TOKENS
from darkmind_v2.training.validate_phase4a_preflight import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_BOUNDARIES_HASH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_CORPUS_HASH,
    EXPECTED_SHARD_CHECKSUMS_HASH,
    EXPECTED_TOKENIZED_HASH,
    ROOT,
)


RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase4b")
INPUT_ROOT = RUNTIME_ROOT / "inputs"
TOKENIZED_INPUT = INPUT_ROOT / "corpus_v3_tokenized"
TOKENIZER_INPUT = INPUT_ROOT / "tokenizer" / "darkmind_v2_sp_bpe24k_v1"
MODEL_INPUT = INPUT_ROOT / "model" / "model_base_v1.json"
SOURCE_TOKENIZED = ROOT / "darkmind_v2" / "data" / "phase3c1" / "tokenized" / "tranche1_v2"
SOURCE_TOKENIZER = ROOT / "darkmind_v2" / "tokenizer" / "frozen" / "darkmind_v2_sp_bpe24k_v1"
SOURCE_MODEL = ROOT / "darkmind_v2" / "config" / "model_base_v1.json"
RELOCATION_MANIFEST = INPUT_ROOT / "runtime_relocation_manifest.json"
ORDER_ROOT = INPUT_ROOT / "sequence_orders"
AUDIT_JSON = ROOT / "darkmind_v2" / "reports" / "phase4b_sequence_order_audit.json"
AUDIT_MARKDOWN = ROOT / "darkmind_v2" / "reports" / "phase4b_sequence_order_audit.md"

# Predeclared before inspecting Phase 4A window results.
MATERIAL_PERCENTAGE_POINT_THRESHOLD = 10.0
MATERIAL_SOURCE_JSD_THRESHOLD = 0.10
MATERIAL_CONTIGUOUS_SEQUENCE_RUN = 64
SEVERE_PERCENTAGE_POINT_THRESHOLD = 25.0
SEVERE_SOURCE_JSD_THRESHOLD = 0.25
SEVERE_CONTIGUOUS_SEQUENCE_RUN = 512
SEQUENCE_LENGTH = 512
DATA_ORDER_SEED = 20260712

COPIED_METADATA = (
    "attribution_manifest.jsonl",
    "contamination_report.json",
    "corpus_manifest.json",
    "document_boundaries.jsonl",
    "duplicate_report.json",
    "language_category_allocation.json",
    "rejection_report.json",
    "seed_import_manifest.json",
    "shard_checksums.json",
    "source_allocation.json",
    "split_manifest.jsonl",
    "tokenization_statistics.json",
    "tokenized_corpus_manifest.json",
    "validation_report.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_runtime_path(path: Path) -> Path:
    resolved = path.resolve()
    root = RUNTIME_ROOT.resolve()
    if "\\onedrive\\" in str(resolved).lower():
        raise ValueError(f"Phase 4B mutable runtime path is under OneDrive: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Phase 4B runtime path must stay under {root}: {resolved}") from exc
    return resolved


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    retries: int = 20,
    replace: Callable[[str, str], None] = os.replace,
) -> None:
    path = ensure_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        for attempt in range(retries):
            try:
                replace(str(temporary), str(path))
                return
            except PermissionError:
                if attempt + 1 == retries:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_atomic(source: Path, destination: Path) -> bool:
    destination = ensure_runtime_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != source_hash:
            raise ValueError(f"existing relocated input differs from source: {destination}")
        return True
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.copying")
    if temporary.exists():
        raise FileExistsError(f"incomplete relocation requires inspection: {temporary}")
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size or sha256_file(temporary) != source_hash:
        raise ValueError(f"copied file failed verification: {destination}")
    for attempt in range(20):
        try:
            os.replace(temporary, destination)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    return False


def relocation_plan() -> list[tuple[str, Path, Path]]:
    manifest = json.loads((SOURCE_TOKENIZED / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    plan = [
        ("corpus", SOURCE_TOKENIZED / shard["filename"], TOKENIZED_INPUT / shard["filename"])
        for shard in manifest["shards"]
    ]
    plan.extend(("corpus_metadata", SOURCE_TOKENIZED / name, TOKENIZED_INPUT / name) for name in COPIED_METADATA)
    plan.extend(
        ("tokenizer", source, TOKENIZER_INPUT / source.name)
        for source in sorted(SOURCE_TOKENIZER.iterdir())
        if source.is_file()
    )
    plan.append(("model", SOURCE_MODEL, MODEL_INPUT))
    return plan


def relocate_inputs() -> dict[str, Any]:
    for name in ("inputs", "runs", "exports", "temporary", "logs"):
        ensure_runtime_path(RUNTIME_ROOT / name).mkdir(parents=True, exist_ok=True)
    records = []
    for role, source, destination in relocation_plan():
        reused = _copy_atomic(source, destination)
        digest = sha256_file(destination)
        records.append(
            {
                "role": role,
                "source": str(source),
                "destination": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "reused_existing_copy": reused,
            }
        )
        print(f"relocated role={role} file={destination.name} bytes={destination.stat().st_size} reused={reused}", flush=True)
    payload = {
        "schema_version": "darkmind-v2-phase4b-runtime-relocation-v1",
        "result": "PASS",
        "runtime_root": str(RUNTIME_ROOT),
        "source_copies_preserved": True,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "mutable_runtime_outside_onedrive": True,
    }
    atomic_write_json(RELOCATION_MANIFEST, payload)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_copied_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {Path(item["destination"]).name: item["sha256"] for item in manifest["files"]}


def validate_runtime_inputs(pass_index: int, compare: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    relocation = _load_json(RELOCATION_MANIFEST)
    expected_files = _expected_copied_hashes(relocation)
    snapshots: dict[str, Any] = {}
    for item in relocation["files"]:
        path = ensure_runtime_path(Path(item["destination"]))
        actual = sha256_file(path)
        if path.stat().st_size != item["bytes"] or actual != item["sha256"]:
            raise ValueError(f"relocated immutable input changed: {path}")
        snapshots[str(path.relative_to(RUNTIME_ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }

    manifest = _load_json(TOKENIZED_INPUT / "tokenized_corpus_manifest.json")
    if manifest["deterministic_content_hash"] != EXPECTED_TOKENIZED_HASH:
        raise ValueError("tokenized Corpus V3 content hash changed")
    if manifest["source"]["corpus_manifest_deterministic_hash"] != EXPECTED_CORPUS_HASH:
        raise ValueError("Corpus V3 deterministic hash changed")
    if manifest["document_boundaries"]["sha256"] != EXPECTED_BOUNDARIES_HASH:
        raise ValueError("document boundary hash changed")
    if sha256_file(TOKENIZED_INPUT / "document_boundaries.jsonl") != EXPECTED_BOUNDARIES_HASH:
        raise ValueError("document boundary file changed")
    if sha256_file(TOKENIZED_INPUT / "shard_checksums.json") != EXPECTED_SHARD_CHECKSUMS_HASH:
        raise ValueError("shard checksum manifest changed")

    checksums = _load_json(TOKENIZED_INPUT / "shard_checksums.json")
    shards: dict[str, np.memmap] = {}
    split_tokens: Counter[str] = Counter()
    minimum_token = 24000
    maximum_token = 0
    for record in manifest["shards"]:
        path = TOKENIZED_INPUT / record["filename"]
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"] or checksums.get(record["filename"]) != actual_hash:
            raise ValueError(f"shard hash mismatch: {record['filename']}")
        values = np.memmap(path, mode="r", dtype="<u2")
        if values.mode != "r" or len(values) != record["tokens"]:
            raise ValueError(f"shard read-only/count validation failed: {record['filename']}")
        minimum_token = min(minimum_token, int(values.min()))
        maximum_token = max(maximum_token, int(values.max()))
        split_tokens[record["split"]] += len(values)
        shards[record["filename"]] = values
    if minimum_token < 0 or maximum_token >= 24000:
        raise ValueError("token ID outside frozen vocabulary")

    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    split_documents: Counter[str] = Counter()
    eos_boundaries = 0
    records = 0
    boundary_path = TOKENIZED_INPUT / "document_boundaries.jsonl"
    attribution_path = TOKENIZED_INPUT / "attribution_manifest.jsonl"
    split_path = TOKENIZED_INPUT / "split_manifest.jsonl"
    with boundary_path.open("r", encoding="utf-8") as boundaries, attribution_path.open(
        "r", encoding="utf-8"
    ) as attributions, split_path.open("r", encoding="utf-8") as splits:
        for boundary_line, attribution_line, split_line in zip_longest(boundaries, attributions, splits):
            if boundary_line is None or attribution_line is None or split_line is None:
                raise ValueError("boundary, attribution, and split metadata counts differ")
            boundary = json.loads(boundary_line)
            attribution = json.loads(attribution_line)
            split = json.loads(split_line)
            document_id = boundary["id"]
            if attribution["id"] != document_id or split["id"] != document_id:
                raise ValueError(f"metadata ID mismatch: {document_id}")
            for key in ("split", "language", "category"):
                if attribution[key] != boundary[key] or split[key] != boundary[key]:
                    raise ValueError(f"metadata {key} mismatch: {document_id}")
            if attribution["source_id"] != split["source_id"]:
                raise ValueError(f"source metadata mismatch: {document_id}")
            if not attribution.get("license") or not attribution.get("source_url") or not attribution.get("attribution"):
                raise ValueError(f"incomplete attribution: {document_id}")
            prior_split = seen_ids.setdefault(document_id, boundary["split"])
            prior_hash_split = seen_hashes.setdefault(boundary["text_sha256"], boundary["split"])
            if prior_split != boundary["split"] or prior_hash_split != boundary["split"]:
                raise ValueError(f"cross-split leakage: {document_id}")
            shard = shards[boundary["shard"]]
            if boundary["end_offset"] <= boundary["start_offset"] or int(shard[boundary["end_offset"] - 1]) != 3:
                raise ValueError(f"missing EOS boundary: {document_id}")
            eos_boundaries += 1
            split_documents[boundary["split"]] += 1
            records += 1
    if records != manifest["document_boundaries"]["records"]:
        raise ValueError("document count differs from tokenized manifest")
    if dict(split_tokens) != manifest["statistics"]["split_tokens"]:
        raise ValueError("split token totals differ from tokenized manifest")

    for filename, expected in EXPECTED_HASHES.items():
        if sha256_file(TOKENIZER_INPUT / filename) != expected:
            raise ValueError(f"frozen tokenizer hash mismatch: {filename}")
    tokenizer_manifest = _load_json(TOKENIZER_INPUT / "tokenizer_freeze_manifest.json")
    if tokenizer_manifest["vocab_size"] != 24000 or tokenizer_manifest["special_token_ids"] != SPECIAL_TOKENS:
        raise ValueError("frozen tokenizer contract changed")
    if sha256_file(MODEL_INPUT) != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen Base V1 config file changed")
    model_config = DarkMindV2Config.from_json_file(MODEL_INPUT)
    if model_config_hash(model_config) != EXPECTED_ARCHITECTURE_HASH:
        raise ValueError("frozen Base V1 architecture changed")

    cross_pass = None
    if compare is not None:
        previous = _load_json(compare)
        if previous.get("result") != "PASS" or previous["asset_snapshot"] != snapshots:
            raise ValueError("runtime input identity differs across validator passes")
        cross_pass = True
    payload = {
        "schema_version": "darkmind-v2-phase4b-runtime-input-validation-v1",
        "result": "PASS",
        "pass_index": pass_index,
        "cross_pass_asset_identity": cross_pass,
        "asset_snapshot": snapshots,
        "copied_bytes": relocation["total_bytes"],
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        "boundary_hash": EXPECTED_BOUNDARIES_HASH,
        "shard_checksums_hash": EXPECTED_SHARD_CHECKSUMS_HASH,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "tokenizer_hashes": EXPECTED_HASHES,
        "shards": len(shards),
        "split_tokens": dict(split_tokens),
        "split_documents": dict(split_documents),
        "total_tokens": sum(split_tokens.values()),
        "complete_train_sequences": split_tokens["train"] // SEQUENCE_LENGTH,
        "complete_train_sequence_tokens": split_tokens["train"] // SEQUENCE_LENGTH * SEQUENCE_LENGTH,
        "unused_train_tail_tokens": split_tokens["train"] % SEQUENCE_LENGTH,
        "minimum_token_id": minimum_token,
        "maximum_token_id": maximum_token,
        "eos_boundaries": eos_boundaries,
        "mutable_runtime_outside_onedrive": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return payload


def _shard_global_starts(manifest: dict[str, Any], split: str) -> dict[str, int]:
    starts: dict[str, int] = {}
    total = 0
    for shard in manifest["shards"]:
        if shard["split"] == split:
            starts[shard["filename"]] = total
            total += int(shard["tokens"])
    return starts


def load_document_spans() -> dict[str, list[dict[str, Any]]]:
    manifest = _load_json(TOKENIZED_INPUT / "tokenized_corpus_manifest.json")
    starts = {split: _shard_global_starts(manifest, split) for split in ("train", "validation", "eval")}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (TOKENIZED_INPUT / "document_boundaries.jsonl").open("r", encoding="utf-8") as boundaries, (
        TOKENIZED_INPUT / "attribution_manifest.jsonl"
    ).open("r", encoding="utf-8") as attributions:
        for boundary_line, attribution_line in zip(boundaries, attributions):
            boundary = json.loads(boundary_line)
            attribution = json.loads(attribution_line)
            split = boundary["split"]
            start = starts[split][boundary["shard"]] + int(boundary["start_offset"])
            end = starts[split][boundary["shard"]] + int(boundary["end_offset"])
            result[split].append(
                {
                    "id": boundary["id"],
                    "start": start,
                    "end": end,
                    "tokens": end - start,
                    "language": boundary["language"],
                    "category": "technical" if boundary["category"] == "technical_educational" else "prose",
                    "source": attribution["source_id"],
                }
            )
    return dict(result)


def _percentages(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: value * 100.0 / total for key, value in sorted(counter.items())} if total else {}


def _distribution(spans: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    language: Counter[str] = Counter()
    category: Counter[str] = Counter()
    source: Counter[str] = Counter()
    touched: list[dict[str, Any]] = []
    eos = 0
    for doc in spans:
        if doc["end"] <= start:
            continue
        if doc["start"] >= end:
            break
        overlap = min(end, doc["end"]) - max(start, doc["start"])
        if overlap <= 0:
            continue
        language[doc["language"]] += overlap
        category[doc["category"]] += overlap
        source[doc["source"]] += overlap
        touched.append(doc)
        if start < doc["end"] <= end:
            eos += 1
    token_count = end - start
    lengths = [int(doc["tokens"]) for doc in touched]
    return {
        "token_count": token_count,
        "language_percent": _percentages(language, token_count),
        "category_percent": _percentages(category, token_count),
        "source_percent": _percentages(source, token_count),
        "document_count": len(touched),
        "unique_document_count": len({doc["id"] for doc in touched}),
        "average_document_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "p50_document_tokens": statistics.median(lengths) if lengths else 0.0,
        "p95_document_tokens": percentile(lengths, 0.95),
        "eos_density_per_1000_tokens": eos * 1000.0 / token_count if token_count else 0.0,
    }


def percentile(values: list[int] | list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values()) or 1.0
    right_total = sum(right.values()) or 1.0
    p = {key: left.get(key, 0.0) / left_total for key in keys}
    q = {key: right.get(key, 0.0) / right_total for key in keys}
    midpoint = {key: (p[key] + q[key]) / 2.0 for key in keys}

    def divergence(values: dict[str, float]) -> float:
        return sum(value * math.log2(value / midpoint[key]) for key, value in values.items() if value > 0)

    return (divergence(p) + divergence(q)) / 2.0


def sequence_labels(spans: list[dict[str, Any]], complete_sequences: int) -> list[tuple[str, str, str]]:
    labels: list[tuple[str, str, str]] = []
    document_index = 0
    for sequence_index in range(complete_sequences):
        start = sequence_index * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH
        while document_index < len(spans) and spans[document_index]["end"] <= start:
            document_index += 1
        contributions: Counter[tuple[str, str, str]] = Counter()
        cursor = document_index
        while cursor < len(spans) and spans[cursor]["start"] < end:
            doc = spans[cursor]
            overlap = min(end, doc["end"]) - max(start, doc["start"])
            if overlap > 0:
                contributions[(doc["language"], doc["category"], doc["source"])] += overlap
            cursor += 1
        if not contributions:
            raise ValueError(f"sequence {sequence_index} has no document provenance")
        labels.append(sorted(contributions.items(), key=lambda item: (-item[1], item[0]))[0][0])
    return labels


def longest_run(values: Iterable[str]) -> int:
    best = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        previous = value
        best = max(best, current)
    return best


def _compare_distribution(item: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    language_pp = {
        key: item["language_percent"].get(key, 0.0) - reference["language_percent"].get(key, 0.0)
        for key in sorted(set(item["language_percent"]) | set(reference["language_percent"]))
    }
    category_pp = {
        key: item["category_percent"].get(key, 0.0) - reference["category_percent"].get(key, 0.0)
        for key in sorted(set(item["category_percent"]) | set(reference["category_percent"]))
    }
    return {
        "language_percentage_point_difference": language_pp,
        "category_percentage_point_difference": category_pp,
        "maximum_absolute_language_or_category_pp": max(
            [abs(value) for value in language_pp.values()] + [abs(value) for value in category_pp.values()] + [0.0]
        ),
        "source_jensen_shannon_divergence_bits": jensen_shannon(
            item["source_percent"], reference["source_percent"]
        ),
    }


def audit_legacy_order() -> dict[str, Any]:
    manifest = _load_json(TOKENIZED_INPUT / "tokenized_corpus_manifest.json")
    spans = load_document_spans()
    split_totals = manifest["statistics"]["split_tokens"]
    references = {
        split: _distribution(spans[split], 0, int(split_totals[split]))
        for split in ("train", "validation", "eval")
    }
    complete_sequences = int(split_totals["train"]) // SEQUENCE_LENGTH
    labels = sequence_labels(spans["train"], complete_sequences)
    windows = {
        "steps_1_128": (0, 128 * 16),
        "steps_129_305": (128 * 16, 305 * 16),
        "steps_306_458": (305 * 16, 458 * 16),
        "steps_459_610": (458 * 16, 610 * 16),
    }
    results: dict[str, Any] = {}
    material = False
    severe = False
    mild = False
    for name, (start_sequence, end_sequence) in windows.items():
        item = _distribution(
            spans["train"], start_sequence * SEQUENCE_LENGTH, end_sequence * SEQUENCE_LENGTH
        )
        window_labels = labels[start_sequence:end_sequence]
        item["sequence_range"] = [start_sequence, end_sequence]
        item["longest_contiguous_source_family_run_sequences"] = longest_run(
            label[2] for label in window_labels
        )
        item["longest_contiguous_language_run_sequences"] = longest_run(label[0] for label in window_labels)
        item["comparisons"] = {
            split: _compare_distribution(item, references[split])
            for split in ("train", "validation", "eval")
        }
        train_compare = item["comparisons"]["train"]
        pp = train_compare["maximum_absolute_language_or_category_pp"]
        jsd = train_compare["source_jensen_shannon_divergence_bits"]
        run = item["longest_contiguous_source_family_run_sequences"]
        severe = severe or pp > SEVERE_PERCENTAGE_POINT_THRESHOLD or jsd > SEVERE_SOURCE_JSD_THRESHOLD or run >= SEVERE_CONTIGUOUS_SEQUENCE_RUN
        material = material or pp > MATERIAL_PERCENTAGE_POINT_THRESHOLD or jsd > MATERIAL_SOURCE_JSD_THRESHOLD or run >= MATERIAL_CONTIGUOUS_SEQUENCE_RUN
        mild = mild or pp > 5.0 or jsd > 0.05 or run >= 16
        results[name] = item
    classification = (
        "severely clustered"
        if severe
        else "materially clustered"
        if material
        else "mildly clustered"
        if mild
        else "adequately mixed"
    )
    payload = {
        "schema_version": "darkmind-v2-phase4b-sequence-order-audit-v1",
        "result": "PASS",
        "legacy_order": "legacy_order_v1",
        "classification": classification,
        "predeclared_thresholds": {
            "material_language_or_category_percentage_points": MATERIAL_PERCENTAGE_POINT_THRESHOLD,
            "material_source_jsd_bits": MATERIAL_SOURCE_JSD_THRESHOLD,
            "material_contiguous_source_run_sequences": MATERIAL_CONTIGUOUS_SEQUENCE_RUN,
            "severe_language_or_category_percentage_points": SEVERE_PERCENTAGE_POINT_THRESHOLD,
            "severe_source_jsd_bits": SEVERE_SOURCE_JSD_THRESHOLD,
            "severe_contiguous_source_run_sequences": SEVERE_CONTIGUOUS_SEQUENCE_RUN,
        },
        "references": references,
        "windows": results,
        "sequence_label_rule": "dominant token overlap; deterministic lexical tie break",
        "run_metric": "consecutive dominant sequence labels",
    }
    AUDIT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# DarkMind v2 Phase 4B Sequence-Order Audit",
        "",
        f"Legacy order classification: **{classification}**",
        "",
        "Material clustering was predeclared as any major window exceeding 10 percentage points in language/category, 0.10 bits source-family Jensen-Shannon divergence, or a 64-sequence contiguous source-family run.",
        "",
        "| Window | Tokens | TR % | EN % | Prose % | Technical % | Source JSD | Longest source run | Longest language run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        comparison = item["comparisons"]["train"]
        lines.append(
            f"| {name} | {item['token_count']:,} | {item['language_percent'].get('tr', 0):.3f} | "
            f"{item['language_percent'].get('en', 0):.3f} | {item['category_percent'].get('prose', 0):.3f} | "
            f"{item['category_percent'].get('technical', 0):.3f} | "
            f"{comparison['source_jensen_shannon_divergence_bits']:.6f} | "
            f"{item['longest_contiguous_source_family_run_sequences']} | "
            f"{item['longest_contiguous_language_run_sequences']} |"
        )
    lines.extend(["", "Detailed source-family percentages, document statistics, EOS density, validation/eval comparisons, and percentage-point differences are retained in the JSON report.", ""])
    AUDIT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return payload


def _order_manifest(name: str, order: list[int], labels: list[tuple[str, str, str]]) -> dict[str, Any]:
    core = {
        "schema_version": "darkmind-v2-phase4b-sequence-order-v1",
        "name": name,
        "seed": DATA_ORDER_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "available_complete_sequences": len(order),
        "available_complete_sequence_tokens": len(order) * SEQUENCE_LENGTH,
        "no_replacement": len(order) == len(set(order)),
        "no_wrap": True,
        "covers_all_complete_train_sequences": sorted(order) == list(range(len(order))),
        "excludes_validation_and_eval": True,
        "indices": order,
        "label_rule": "dominant token overlap; deterministic lexical tie break",
        "label_counts": dict(sorted(Counter("|".join(label) for label in labels).items())),
    }
    core["deterministic_content_hash"] = canonical_json_hash(core)
    return core


def deterministic_stratified_order(
    labels: list[tuple[str, str, str]], seed: int = DATA_ORDER_SEED
) -> list[int]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    for label, values in groups.items():
        label_seed = int(hashlib.sha256((str(seed) + "|" + "|".join(label)).encode()).hexdigest()[:16], 16)
        random.Random(label_seed).shuffle(values)
    group_sizes = {key: len(values) for key, values in groups.items()}
    used = Counter()
    positions = Counter()
    result: list[int] = []
    last_group: tuple[str, str, str] | None = None
    total = len(labels)
    tie_order = {
        key: hashlib.sha256(("tie|" + str(seed) + "|" + "|".join(key)).encode()).hexdigest()
        for key in groups
    }
    for position in range(total):
        candidates = [key for key, values in groups.items() if positions[key] < len(values)]
        selected = sorted(
            candidates,
            key=lambda key: (
                -(((position + 1) * group_sizes[key] / total) - used[key]),
                key == last_group and len(candidates) > 1,
                tie_order[key],
            ),
        )[0]
        result.append(groups[selected][positions[selected]])
        positions[selected] += 1
        used[selected] += 1
        last_group = selected
    return result


def build_orders() -> dict[str, Any]:
    manifest = _load_json(TOKENIZED_INPUT / "tokenized_corpus_manifest.json")
    spans = load_document_spans()["train"]
    complete_sequences = int(manifest["statistics"]["split_tokens"]["train"]) // SEQUENCE_LENGTH
    labels = sequence_labels(spans, complete_sequences)
    legacy = list(range(complete_sequences))
    stratified = deterministic_stratified_order(labels)
    ORDER_ROOT.mkdir(parents=True, exist_ok=True)
    legacy_manifest = _order_manifest("legacy_order_v1", legacy, labels)
    stratified_labels = [labels[index] for index in stratified]
    stratified_manifest = _order_manifest("deterministic_stratified_v1", stratified, stratified_labels)
    atomic_write_json(ORDER_ROOT / "legacy_order_v1.json", legacy_manifest)
    atomic_write_json(ORDER_ROOT / "deterministic_stratified_v1.json", stratified_manifest)
    return {
        "legacy_hash": legacy_manifest["deterministic_content_hash"],
        "stratified_hash": stratified_manifest["deterministic_content_hash"],
        "complete_sequences": complete_sequences,
        "legacy_first_9760": legacy[:9760],
        "stratified_first_9760": stratified[:9760],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("relocate")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--pass-index", type=int, required=True)
    validate_parser.add_argument("--compare", type=Path)
    validate_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("build-orders")
    args = parser.parse_args()
    if args.command == "relocate":
        payload = relocate_inputs()
    elif args.command == "validate":
        payload = validate_runtime_inputs(args.pass_index, args.compare)
        atomic_write_json(args.output, payload)
    elif args.command == "audit":
        payload = audit_legacy_order()
    else:
        payload = build_orders()
    printable = {key: value for key, value in payload.items() if key not in {"files", "asset_snapshot", "windows", "references", "legacy_first_9760", "stratified_first_9760"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
