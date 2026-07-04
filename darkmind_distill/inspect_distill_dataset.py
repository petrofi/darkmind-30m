from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "darkmind_distill" / "data" / "darkmind_qwen_distill_v0_1.jsonl"

LANGUAGE_TARGETS = {"tr": 900, "en": 500, "de": 200, "ja": 200, "es": 100, "fr": 100}
CATEGORY_TARGETS = {
    "programming": 500,
    "debugging": 350,
    "backend": 300,
    "ai_llm": 300,
    "emotional_support": 200,
    "identity": 100,
    "general_reasoning": 150,
    "data_json_sql": 100,
}

CHATGPT_OPENAI_RE = re.compile(
    r"\bben\s+chatgpt\b|\bi\s+am\s+chatgpt\b|openai\s+taraf[ıi]ndan\s+geli[şs]tirildim|"
    r"i\s+was\s+developed\s+by\s+openai|developed\s+by\s+openai|as an ai language model",
    re.IGNORECASE,
)
QWEN_IDENTITY_RE = re.compile(r"\bben\s+qwen\b|\bi\s+am\s+qwen\b|\bqwen['’]?im\b", re.IGNORECASE)
DARKMIND_IDENTITY_RE = re.compile(
    r"^\s*ben\s+darkmind\b|darkmind.*tar[ıi]k\s+yasin|darkmind.*yaz[ıi]l[ıi]m\s+asistan",
    re.IGNORECASE,
)
UNSAFE_CYBER_RE = re.compile(
    r"phishing|keylogger|ransomware|credential theft|steal password|şifre çal|sifre cal|"
    r"exploit yaz|zararlı yazılım|zararli yazilim|sql injection ile saldır|ddos",
    re.IGNORECASE,
)
MEDICAL_THERAPY_RE = re.compile(
    r"tanı koy|teşhis|tedavi planı|terapi yerine|psikolog yerine|medical diagnosis|diagnose you|therapy replacement",
    re.IGNORECASE,
)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def normalized_key(row: dict[str, Any]) -> tuple[str, str]:
    prompt = re.sub(r"\s+", " ", str(row.get("prompt", "")).casefold()).strip()
    response = re.sub(r"\s+", " ", str(row.get("response", "")).casefold()).strip()
    return prompt, response


def duplicate_count(rows: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for row in rows:
        key = normalized_key(row)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def count_pattern(rows: list[dict[str, Any]], pattern: re.Pattern[str], *, response_only: bool = False) -> int:
    count = 0
    for row in rows:
        text = str(row.get("response", "")) if response_only else f"{row.get('prompt', '')}\n{row.get('response', '')}"
        if pattern.search(text):
            count += 1
    return count


def darkmind_identity_leak_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if row.get("category") == "identity":
            continue
        if DARKMIND_IDENTITY_RE.search(str(row.get("response", ""))):
            count += 1
    return count


def scaled_min(target: int, tolerance: float) -> int:
    return int(target * (1.0 - tolerance))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DarkMind Qwen distillation dataset.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--min_total", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--short_threshold", type=int, default=20)
    parser.add_argument("--long_threshold", type=int, default=900)
    parser.add_argument("--skip_target_checks", action="store_true")
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    rows = load_rows(data_path)
    categories = Counter(str(row.get("category", "unknown")) for row in rows)
    languages = Counter(str(row.get("language", "unknown")) for row in rows)
    difficulties = Counter(str(row.get("difficulty", "unknown")) for row in rows)
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    duplicates = duplicate_count(rows)
    chatgpt_openai = count_pattern(rows, CHATGPT_OPENAI_RE)
    qwen_identity = count_pattern(rows, QWEN_IDENTITY_RE)
    darkmind_leak = darkmind_identity_leak_count(rows)
    too_short = sum(1 for row in rows if len(str(row.get("response", "")).strip()) < args.short_threshold)
    too_long = sum(1 for row in rows if len(str(row.get("response", "")).strip()) > args.long_threshold)
    unsafe_cyber = count_pattern(rows, UNSAFE_CYBER_RE)
    medical_therapy = count_pattern(rows, MEDICAL_THERAPY_RE)
    identity_ratio = categories.get("identity", 0) / max(len(rows), 1)

    failures: list[str] = []
    if len(rows) < args.min_total:
        failures.append(f"total rows {len(rows)} is below {args.min_total}")
    if duplicates:
        failures.append(f"duplicate count is {duplicates}")
    if chatgpt_openai:
        failures.append(f"ChatGPT/OpenAI contamination count is {chatgpt_openai}")
    if qwen_identity:
        failures.append(f"Qwen identity contamination count is {qwen_identity}")
    if darkmind_leak:
        failures.append(f"DarkMind identity leakage count is {darkmind_leak}")
    if unsafe_cyber:
        failures.append(f"unsafe cyber/offensive count is {unsafe_cyber}")
    if medical_therapy:
        failures.append(f"medical/therapy claim count is {medical_therapy}")
    if identity_ratio > 0.07:
        failures.append(f"identity category ratio {identity_ratio:.2%} exceeds 7%")

    if not args.skip_target_checks:
        for language, target in LANGUAGE_TARGETS.items():
            minimum = scaled_min(target, args.tolerance)
            if languages.get(language, 0) < minimum:
                failures.append(f"language {language} count {languages.get(language, 0)} is below {minimum}")
        for category, target in CATEGORY_TARGETS.items():
            minimum = scaled_min(target, args.tolerance)
            if categories.get(category, 0) < minimum:
                failures.append(f"category {category} count {categories.get(category, 0)} is below {minimum}")

    print("=" * 70)
    print("DarkMind Qwen distillation dataset inspection")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Duplicate count: {duplicates}")
    print(f"ChatGPT/OpenAI contamination count: {chatgpt_openai}")
    print(f"Qwen identity contamination count: {qwen_identity}")
    print(f"DarkMind identity leakage in non-identity category: {darkmind_leak}")
    print(f"Too short responses: {too_short}")
    print(f"Too long responses: {too_long}")
    print(f"Unsafe cyber/offensive count: {unsafe_cyber}")
    print(f"Medical/therapy claim count: {medical_therapy}")
    print(f"Identity ratio: {identity_ratio:.2%}")
    print("-" * 70)
    print(f"Category distribution: {dict(categories)}")
    print(f"Language distribution: {dict(languages)}")
    print(f"Difficulty distribution: {dict(difficulties)}")
    print(f"Source distribution: {dict(sources)}")
    print("-" * 70)
    print("First 20 examples:")
    for index, row in enumerate(rows[:20], start=1):
        print(f"\n[{index}] {row.get('language')} {row.get('category')} {row.get('difficulty')} source={row.get('source')}")
        print(f"Prompt: {row.get('prompt', '')}")
        print(f"Response: {row.get('response', '')}")
    print("=" * 70)

    if failures:
        print("Inspection failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
