from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import random
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_v0_3.jsonl"
BLOCKED_IDENTITY_PHRASES = [
    "ben chatgpt",
    "chatgpt olarak",
    "openai tarafından geliştirildim",
    "openai tarafindan gelistirildim",
    "openai geliştirdi",
    "openai gelistirdi",
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_rows(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            prompt = row.get("prompt", "")
            response = row.get("response", "")

            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(f"prompt and response must be strings at {path}:{line_number}")

            rows.append(row)

    return rows


def row_hash(row: dict) -> str:
    payload = f"{row.get('prompt', '').strip().lower()}\n{row.get('response', '').strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_duplicates(rows: list[dict]) -> int:
    seen = set()
    duplicates = 0

    for row in rows:
        key = row_hash(row)

        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    return duplicates


def count_blocked_identity(rows: list[dict]) -> int:
    count = 0

    for row in rows:
        text = f"{row.get('prompt', '')}\n{row.get('response', '')}".lower()

        if any(phrase in text for phrase in BLOCKED_IDENTITY_PHRASES):
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DarkMind instruction JSONL data.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA.relative_to(ROOT_DIR)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = resolve_path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    rows = load_rows(data_path)
    source_counts = Counter(row.get("source", "unknown") for row in rows)
    category_counts = Counter(row.get("category", "unknown") for row in rows)
    avg_prompt_len = sum(len(row.get("prompt", "")) for row in rows) / max(len(rows), 1)
    avg_response_len = sum(len(row.get("response", "")) for row in rows) / max(len(rows), 1)
    duplicates = count_duplicates(rows)
    blocked_identity = count_blocked_identity(rows)
    rng = random.Random(args.seed)
    samples = rng.sample(rows, min(10, len(rows))) if rows else []

    print("=" * 70)
    print("DarkMind instruction dataset inspection")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Total examples: {len(rows):,}")
    print(f"Average prompt length: {avg_prompt_len:.1f}")
    print(f"Average response length: {avg_response_len:.1f}")
    print(f"Duplicate count: {duplicates:,}")
    print(f"Blocked identity phrase count: {blocked_identity:,}")
    print("-" * 70)
    print("Examples per source:")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count:,}")
    print("-" * 70)
    print("Examples per category:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count:,}")
    print("-" * 70)
    print("Random samples:")
    for index, row in enumerate(samples, start=1):
        print(f"\n[{index}] source={row.get('source', 'unknown')} category={row.get('category', 'unknown')}")
        print(f"Prompt: {row.get('prompt', '')}")
        print(f"Response: {row.get('response', '')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
