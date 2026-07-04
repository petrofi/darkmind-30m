"""Validate the immutable DarkMind v2 fixed base-prompt suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "language",
    "category",
    "prompt",
    "expected_script",
    "forbidden_patterns",
    "maximum_repetition_ratio",
}
REQUIRED_CATEGORIES = {
    "ordinary prose",
    "factual continuation",
    "technical prose",
    "simple code continuation",
    "sentence completion",
    "repetition detection",
    "mixed-script detection",
    "encoding corruption detection",
}
FORBIDDEN_ASSISTANT_PREFIXES = (
    "explain",
    "how do i",
    "how can i",
    "açıkla",
    "nasıl",
)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_line_number"] = line_number
        prompts.append(record)
    return prompts


def validate_prompts(path: Path) -> tuple[dict[str, Any], list[str]]:
    prompts = load_prompts(path)
    failures: list[str] = []
    ids: set[str] = set()
    categories = Counter()
    languages = Counter()

    for record in prompts:
        line_number = record["_line_number"]
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            failures.append(f"line {line_number}: missing fields {sorted(missing)}")

        prompt_id = str(record.get("id", ""))
        if prompt_id in ids:
            failures.append(f"line {line_number}: duplicate id {prompt_id}")
        ids.add(prompt_id)

        language = record.get("language")
        category = record.get("category")
        categories[str(category)] += 1
        languages[str(language)] += 1
        if language not in {"tr", "en"}:
            failures.append(f"line {line_number}: invalid language {language!r}")
        if category not in REQUIRED_CATEGORIES:
            failures.append(f"line {line_number}: invalid category {category!r}")

        prompt = str(record.get("prompt", ""))
        if not prompt.strip():
            failures.append(f"line {line_number}: empty prompt")
        if prompt.strip().casefold().startswith(FORBIDDEN_ASSISTANT_PREFIXES):
            failures.append(f"line {line_number}: assistant-format prompt is not allowed")

        forbidden_patterns = record.get("forbidden_patterns")
        if not isinstance(forbidden_patterns, list) or not all(isinstance(item, str) for item in forbidden_patterns):
            failures.append(f"line {line_number}: forbidden_patterns must be a list of strings")

        ratio = record.get("maximum_repetition_ratio")
        if not isinstance(ratio, (int, float)) or ratio <= 0 or ratio > 1:
            failures.append(f"line {line_number}: maximum_repetition_ratio must be in (0, 1]")

    missing_categories = REQUIRED_CATEGORIES - set(categories)
    if missing_categories:
        failures.append(f"missing categories: {sorted(missing_categories)}")
    if len(prompts) < 40:
        failures.append(f"prompt count {len(prompts)} is below 40")

    report = {
        "prompt_count": len(prompts),
        "language_distribution": dict(sorted(languages.items())),
        "category_distribution": dict(sorted(categories.items())),
    }
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DarkMind v2 fixed base prompts.")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("fixed_base_prompts.jsonl"))
    args = parser.parse_args()

    report, failures = validate_prompts(args.prompts)
    payload = {"result": "FAIL" if failures else "PASS", "report": report, "failures": failures}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
