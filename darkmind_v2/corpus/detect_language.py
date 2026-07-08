"""Small deterministic Turkish/English language heuristic for Phase 0 validation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TURKISH_CHARS = set("çğıİöşüÇĞIıÖŞÜ")
TURKISH_WORDS = {
    "bir",
    "ve",
    "ile",
    "için",
    "kullanıcı",
    "veritabanı",
    "bugün",
    "türkiye",
    "yapay",
    "zeka",
    "sunucu",
}
ENGLISH_WORDS = {
    "the",
    "and",
    "with",
    "for",
    "user",
    "database",
    "today",
    "turkey",
    "artificial",
    "intelligence",
    "server",
}
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", re.UNICODE)


def detect_language(text: str) -> str:
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if not words:
        return "unknown"

    turkish_score = sum(1 for char in text if char in TURKISH_CHARS) * 2
    turkish_score += sum(1 for word in words if word in TURKISH_WORDS)
    english_score = sum(1 for word in words if word in ENGLISH_WORDS)

    ascii_letters = sum(1 for char in text if "a" <= char.lower() <= "z")
    if ascii_letters and not any(char in TURKISH_CHARS for char in text):
        english_score += 1

    if turkish_score > english_score:
        return "tr"
    if english_score > turkish_score:
        return "en"
    return "unknown"


def language_distribution(texts: list[str]) -> Counter[str]:
    return Counter(detect_language(text) for text in texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Turkish/English language labels.")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    labels = [detect_language(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(dict(Counter(labels)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
