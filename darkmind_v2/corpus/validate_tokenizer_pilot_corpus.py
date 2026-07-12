"""Validate the finalized DarkMind v2 Phase 1B tokenizer pilot corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_tokenizer_pilot_corpus import MOJIBAKE_MARKERS, WORD_RE, load_json, sha256_text
    from .detect_mojibake import detect_text
except ImportError:  # pragma: no cover - CLI fallback
    from build_tokenizer_pilot_corpus import MOJIBAKE_MARKERS, WORD_RE, load_json, sha256_text
    from detect_mojibake import detect_text


DEFAULT_PROCESSED_DIR = Path("darkmind_v2/data/phase1b/processed")
DEFAULT_PLAN = Path("darkmind_v2/config/tokenizer_pilot_corpus.json")


def read_split(path: Path) -> tuple[str, list[str]]:
    content = path.read_bytes().decode("utf-8")
    documents = [document for document in content.rstrip("\n").split("\n\n") if document]
    return content, documents


def within_tolerance(actual: int, target: int) -> bool:
    tolerance = target * 0.01
    return target - tolerance <= actual <= target + tolerance


def validate_processed_corpus(processed_dir: Path, plan_path: Path) -> tuple[dict[str, Any], list[str]]:
    expected_files = {
        "train": "tokenizer_train.txt",
        "validation": "tokenizer_validation.txt",
        "eval": "tokenizer_eval.txt",
        "corpus_manifest": "corpus_manifest.json",
        "attribution_manifest": "attribution_manifest.jsonl",
        "rejected_documents": "rejected_documents.jsonl",
        "source_allocation": "source_allocation.json",
        "split_manifest": "split_manifest.json",
        "determinism": "determinism_verification.json",
    }
    missing_files = [name for name, filename in expected_files.items() if not (processed_dir / filename).exists()]
    if missing_files:
        return {"result": "FAIL", "missing_files": missing_files}, [f"missing required output files: {', '.join(missing_files)}"]

    plan = load_json(plan_path)
    split_contents: dict[str, str] = {}
    split_documents: dict[str, list[str]] = {}
    utf8_failures = 0
    for split in ("train", "validation", "eval"):
        try:
            split_contents[split], split_documents[split] = read_split(processed_dir / expected_files[split])
        except UnicodeDecodeError:
            utf8_failures += 1
            split_contents[split], split_documents[split] = "", []

    attribution_records = [
        json.loads(line)
        for line in (processed_dir / expected_files["attribution_manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_manifest = load_json(processed_dir / expected_files["split_manifest"])
    corpus_manifest = load_json(processed_dir / expected_files["corpus_manifest"])
    source_allocation = load_json(processed_dir / expected_files["source_allocation"])
    determinism = load_json(processed_dir / expected_files["determinism"])

    all_documents = [document for split in ("train", "validation", "eval") for document in split_documents[split]]
    expected_attribution = sum(len(split_documents[split]) for split in ("train", "validation", "eval"))
    attribution_by_split = Counter(record.get("selected_split") for record in attribution_records)
    missing_license = sum(int(not record.get("license") or not record.get("license_url")) for record in attribution_records)
    required_attribution = (
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
    )
    missing_attribution = sum(
        int(any(not record.get(field) for field in required_attribution)) for record in attribution_records
    )
    language_characters = Counter()
    for record in attribution_records:
        language_characters[record.get("language", "unknown")] += int(record.get("selected_character_count", 0))

    mojibake = sum(
        int(any(marker in document for marker in MOJIBAKE_MARKERS) and bool(detect_text(document)))
        for document in all_documents
    )
    replacements = sum(document.count("\ufffd") for document in all_documents)
    exact_seen: set[str] = set()
    near_seen: set[tuple[str, ...]] = set()
    exact_duplicates = 0
    near_duplicates = 0
    for document in all_documents:
        exact = sha256_text(" ".join(document.casefold().split()))
        near = tuple(WORD_RE.findall(document.casefold())[:24])
        exact_duplicates += int(exact in exact_seen)
        near_duplicates += int(near in near_seen)
        exact_seen.add(exact)
        near_seen.add(near)

    actual_split_hashes = {split: sha256_text(split_contents[split]) for split in ("train", "validation", "eval")}
    split_hash_match = all(
        split_manifest["splits"][split]["sha256"] == actual_split_hashes[split]
        for split in ("train", "validation", "eval")
    )
    manifest_for_hash = dict(corpus_manifest)
    declared_manifest_hash = manifest_for_hash.pop("deterministic_content_sha256", "")
    computed_manifest_hash = sha256_text(
        json.dumps(manifest_for_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    source_cap_violations = [
        item["source_id"]
        for item in source_allocation["source_allocations"]
        if item["selected_total"] > item["source_cap"] or item["source_cap_status"] != "PASS"
    ]
    determinism_pass = (
        determinism.get("result") == "PASS"
        and determinism.get("first_pass_hashes") == determinism.get("second_pass_hashes")
    )
    total_characters = sum(len(document) for document in all_documents)
    target_total = int(plan["target_normalized_characters"])
    target_tr = int(plan["language_mix"]["tr"]["target_characters"])
    target_en = int(plan["language_mix"]["en"]["target_characters"])

    failures: list[str] = []
    if utf8_failures:
        failures.append(f"invalid UTF-8 split files: {utf8_failures}")
    if mojibake:
        failures.append(f"mojibake detections: {mojibake}")
    if replacements:
        failures.append(f"replacement characters: {replacements}")
    if exact_duplicates or near_duplicates:
        failures.append(f"unresolved duplicates exact={exact_duplicates} near={near_duplicates}")
    if missing_license or missing_attribution:
        failures.append(f"missing metadata license={missing_license} attribution={missing_attribution}")
    if expected_attribution != len(attribution_records) or any(
        attribution_by_split[split] != len(split_documents[split]) for split in ("train", "validation", "eval")
    ):
        failures.append("attribution records do not match selected split documents")
    if not split_hash_match:
        failures.append("split manifest hashes do not match split files")
    if declared_manifest_hash != computed_manifest_hash:
        failures.append("corpus manifest deterministic content hash does not match")
    if source_cap_violations:
        failures.append(f"source cap violations: {', '.join(source_cap_violations)}")
    if not determinism_pass:
        failures.append("deterministic verification artifact failed")
    if not within_tolerance(total_characters, target_total):
        failures.append(f"total characters {total_characters} outside +/-1% target")
    if not within_tolerance(language_characters["tr"], target_tr):
        failures.append(f"Turkish characters {language_characters['tr']} outside +/-1% target")
    if not within_tolerance(language_characters["en"], target_en):
        failures.append(f"English characters {language_characters['en']} outside +/-1% target")

    report = {
        "result": "FAIL" if failures else "PASS",
        "total_documents": len(all_documents),
        "total_characters": total_characters,
        "language_characters": dict(language_characters),
        "split_documents": {split: len(split_documents[split]) for split in split_documents},
        "split_characters": {split: sum(map(len, split_documents[split])) for split in split_documents},
        "utf8_failures": utf8_failures,
        "mojibake_detections": mojibake,
        "replacement_characters": replacements,
        "unresolved_exact_duplicates": exact_duplicates,
        "unresolved_near_duplicate_clusters": near_duplicates,
        "missing_license_metadata": missing_license,
        "missing_attribution_metadata": missing_attribution,
        "split_hashes_match": split_hash_match,
        "corpus_manifest_hash_match": declared_manifest_hash == computed_manifest_hash,
        "source_cap_violations": source_cap_violations,
        "deterministic_verification": determinism_pass,
        "failures": failures,
    }
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a finalized DarkMind v2 Phase 1B tokenizer pilot corpus.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    try:
        report, failures = validate_processed_corpus(args.processed_dir, args.plan)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        report, failures = {"result": "FAIL", "error": str(exc)}, [str(exc)]
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
