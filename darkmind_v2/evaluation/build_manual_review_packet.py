"""Build a blinded Phase 5A manual-review packet outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(r"C:\DarkMindRuntime\phase5a\evaluations\base_quality_suite_v1_v2")
DEFAULT_OUTPUT = Path(r"C:\DarkMindRuntime\phase5a\manual_review\phase5a_manual_review_packet_v2.md")
SCORE_FIELDS = {
    "grammatical_structure": "0-4",
    "topical_consistency": "0-4",
    "language_consistency": "0-4",
    "completion_quality": "0-4",
    "repetition_severity_reversed": "0-4",
    "factual_reliability_if_applicable": "0-4 or N/A",
    "technical_usefulness_if_applicable": "0-4 or N/A",
    "overall_usability": "0-4",
    "reviewer_note": "free text",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_stratified_records(manifests: dict[str, dict[str, Any]], target: int = 150) -> list[dict[str, Any]]:
    mode_buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for mode in sorted(manifests):
        for record in manifests[mode]["results"]:
            mode_buckets[record["category"]][mode].append({**record, "review_decoding_mode": mode})
    buckets: dict[str, list[dict[str, Any]]] = {}
    for category, by_mode in mode_buckets.items():
        interleaved = []
        for position in range(max(len(records) for records in by_mode.values())):
            for mode in sorted(by_mode):
                if position < len(by_mode[mode]):
                    interleaved.append(by_mode[mode][position])
        buckets[category] = interleaved
    categories = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < target:
        category = categories[cursor % len(categories)]
        position = cursor // len(categories)
        if position < len(buckets[category]):
            selected.append(buckets[category][position])
        cursor += 1
        if cursor > target * len(categories) * 2:
            raise ValueError("insufficient records for manual review target")
    return selected


def review_item(record: dict[str, Any]) -> dict[str, Any]:
    identity = f"{record['review_decoding_mode']}|{record['prompt_id']}".encode("utf-8")
    return {
        "review_id": "DM5A-" + hashlib.sha256(identity).hexdigest()[:12].upper(),
        "category": record["category"],
        "language": record["language"],
        "prompt": record["prompt"],
        "raw_continuation": record["output"],
        "decoding_mode": record["review_decoding_mode"],
        "repetition_warning": "repetition" in record["policy"]["warnings"],
        "loop_warning": bool(record["exact_repeated_ngram_loops"]),
        "eos_completed": bool(record["eos_completed"]),
        "scores": {name: None for name in SCORE_FIELDS},
    }


def validate_packet_schema(items: list[dict[str, Any]]) -> None:
    required = {
        "review_id", "category", "language", "prompt", "raw_continuation", "decoding_mode",
        "repetition_warning", "loop_warning", "eos_completed", "scores",
    }
    if len(items) < 150 or len({item["review_id"] for item in items}) != len(items):
        raise ValueError("manual review packet must contain at least 150 unique items")
    for item in items:
        if set(item) != required or set(item["scores"]) != set(SCORE_FIELDS):
            raise ValueError("manual review packet schema mismatch")
        if any(value is not None for value in item["scores"].values()):
            raise ValueError("human scores must remain blank")


def render_packet(items: list[dict[str, Any]], checkpoint_hash: str, prompt_hash: str) -> str:
    lines = [
        "# DarkMind v2 Phase 5A manual base-continuation review",
        "",
        f"Checkpoint model SHA-256: `{checkpoint_hash}`  ",
        f"Prompt manifest SHA-256: `{prompt_hash}`  ",
        f"Stratified outputs: {len(items)}",
        "",
        "Automatic semantic scores are intentionally absent. Review the raw continuation before entering any score.",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines.extend([
            f"## {index}. {item['review_id']}",
            "",
            f"- Category: `{item['category']}`",
            f"- Language: `{item['language']}`",
            f"- Decoding: `{item['decoding_mode']}`",
            f"- Repetition warning: `{str(item['repetition_warning']).lower()}`",
            f"- Loop warning: `{str(item['loop_warning']).lower()}`",
            f"- EOS completed: `{str(item['eos_completed']).lower()}`",
            "",
            "**Prompt**",
            "",
            item["prompt"],
            "",
            "**Raw continuation**",
            "",
            item["raw_continuation"] if item["raw_continuation"] else "[EMPTY OUTPUT]",
            "",
            "**Human scores**",
            "",
        ])
        for field, scale in SCORE_FIELDS.items():
            lines.append(f"- {field} ({scale}): ____")
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def build_packet(input_dir: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT, target: int = 150) -> dict[str, Any]:
    manifests = {
        "greedy": load_json(input_dir / "greedy_manifest.json"),
        "seeded_sampling": load_json(input_dir / "seeded_sampling_manifest.json"),
    }
    summary = load_json(input_dir / "automatic_summary.json")
    items = [review_item(record) for record in select_stratified_records(manifests, target)]
    validate_packet_schema(items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_packet(items, summary["checkpoint_model_sha256"], summary["prompt_manifest_sha256"]),
        encoding="utf-8",
    )
    schema_path = output_path.with_name("phase5a_manual_review_schema.json")
    schema_path.write_text(
        json.dumps({"schema_version": "darkmind-v2-phase5a-manual-review-v1", "score_fields": SCORE_FIELDS, "scores_prepopulated": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "darkmind-v2-phase5a-manual-packet-v1",
        "result": "PASS",
        "packet": str(output_path),
        "schema": str(schema_path),
        "review_items": len(items),
        "category_counts": dict(sorted(__import__("collections").Counter(item["category"] for item in items).items())),
        "decoding_counts": dict(sorted(__import__("collections").Counter(item["decoding_mode"] for item in items).items())),
        "human_scores_prepopulated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.input_dir, args.output, args.target), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
