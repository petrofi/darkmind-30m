from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import random
import re
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_v0_3.jsonl"
BLOCKED_IDENTITY_PHRASES = [
    "ben chatgpt",
    "chatgpt olarak",
    "ben openai",
    "openai tarafından geliştirildim",
    "openai tarafindan gelistirildim",
    "openai geliştirdi",
    "openai gelistirdi",
    "as an ai language model",
    "i am chatgpt",
]
FORBIDDEN_IDENTITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bben\s+chatgpt\b",
        r"\bchatgpt\b",
        r"openai\s+taraf.ndan",
        r"\bi\s+am\s+chatgpt\b",
        r"\bi['’]m\s+chatgpt\b",
        r"\bas\s+an\s+ai\s+language\s+model\b",
        r"\bas\s+a\s+language\s+model\b",
        r"\bdeveloped\s+by\s+openai\b",
        r"\btrained\s+by\s+openai\b",
        r"\bopenai\b",
        r"\bgpt-?3\b",
        r"\bgpt-?4\b",
        r"\bgpt-?5\b",
    ]
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "evet"}:
        return True

    if normalized in {"0", "false", "no", "n", "hayır", "hayir"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def metadata_path_for(data_path: Path) -> Path:
    return data_path.with_suffix(".meta.json")


def load_metadata(data_path: Path) -> dict:
    meta_path = metadata_path_for(data_path)

    if not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


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
        if has_forbidden_identity_claim(row):
            count += 1

    return count


def contains_forbidden_identity_claim(text: str) -> bool:
    if not text:
        return False

    return any(pattern.search(text) for pattern in FORBIDDEN_IDENTITY_PATTERNS)


def is_allowed_local_identity_exception(row: dict) -> bool:
    if row.get("source") not in {"local_identity", "local_identity_anchor"}:
        return False

    prompt = str(row.get("prompt", "")).lower()
    response = str(row.get("response", "")).lower()

    asks_about_other_model = "chatgpt" in prompt or "openai" in prompt or "gpt" in prompt
    denies_other_model = (
        "hayır" in response
        or "hayir" in response
        or "değilim" in response
        or "degilim" in response
        or "geliştirilmedim" in response
        or "gelistirilmedim" in response
    )
    says_darkmind = "darkmind" in response
    return asks_about_other_model and denies_other_model and says_darkmind


def has_forbidden_identity_claim(row: dict) -> bool:
    if is_allowed_local_identity_exception(row):
        response = str(row.get("response", "")).lower()
        claims_openai_developed = (
            "openai tarafından geliştirildim" in response
            or "openai tarafindan gelistirildim" in response
            or "developed by openai" in response
            or "trained by openai" in response
        )
        claims_chatgpt = "ben chatgpt" in response or "i am chatgpt" in response
        return claims_openai_developed or claims_chatgpt

    text = f"{row.get('prompt', '')}\n{row.get('response', '')}"
    return contains_forbidden_identity_claim(text)


def rows_with_forbidden_identity(rows: list[dict]) -> list[dict]:
    return [row for row in rows if has_forbidden_identity_claim(row)]


def is_external(row: dict) -> bool:
    return not str(row.get("source", "")).startswith("local_")


def is_local_identity(row: dict) -> bool:
    return str(row.get("source", "")) in {"local_identity", "local_identity_anchor"}


def is_local_fallback(row: dict) -> bool:
    source = str(row.get("source", ""))
    allowed_local_sources = {"local_identity", "local_identity_anchor", "local_support_seed"}
    return source.startswith("local_") and source not in allowed_local_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DarkMind instruction JSONL data.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA.relative_to(ROOT_DIR)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_external", type=parse_bool, default=False)
    parser.add_argument("--min_external_ratio", type=float, default=0.8)
    parser.add_argument("--max_identity_ratio", type=float, default=0.12)
    parser.add_argument("--fail_on_quality_issues", type=parse_bool, default=False)
    args = parser.parse_args()

    data_path = resolve_path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    rows = load_rows(data_path)
    metadata = load_metadata(data_path)
    require_external = bool(metadata.get("require_external", args.require_external))
    min_external_ratio = float(metadata.get("min_external_ratio", args.min_external_ratio))
    max_identity_ratio = float(metadata.get("max_identity_ratio", args.max_identity_ratio))
    source_counts = Counter(row.get("source", "unknown") for row in rows)
    category_counts = Counter(row.get("category", "unknown") for row in rows)
    avg_prompt_len = sum(len(row.get("prompt", "")) for row in rows) / max(len(rows), 1)
    avg_response_len = sum(len(row.get("response", "")) for row in rows) / max(len(rows), 1)
    duplicates = count_duplicates(rows)
    blocked_identity = count_blocked_identity(rows)
    forbidden_rows = rows_with_forbidden_identity(rows)
    external_count = sum(1 for row in rows if is_external(row))
    local_count = len(rows) - external_count
    local_identity_count = sum(1 for row in rows if is_local_identity(row))
    local_fallback_count = sum(1 for row in rows if is_local_fallback(row))
    identity_count = category_counts.get("identity", 0)
    external_ratio = external_count / max(len(rows), 1)
    local_ratio = local_count / max(len(rows), 1)
    local_identity_ratio = local_identity_count / max(len(rows), 1)
    local_fallback_ratio = local_fallback_count / max(len(rows), 1)
    identity_ratio = identity_count / max(len(rows), 1)
    rng = random.Random(args.seed)
    samples = rng.sample(rows, min(10, len(rows))) if rows else []
    failures = []

    if duplicates > 0:
        failures.append(f"duplicate count is {duplicates}")
    if blocked_identity > 0:
        failures.append(f"blocked identity phrase count is {blocked_identity}")
    if local_fallback_count > 0:
        failures.append(f"local fallback count is {local_fallback_count}")
    if identity_ratio > max_identity_ratio:
        failures.append(f"identity ratio {identity_ratio:.2%} exceeds {max_identity_ratio:.2%}")
    if require_external and external_ratio < min_external_ratio:
        failures.append(f"external ratio {external_ratio:.2%} is below {min_external_ratio:.2%}")

    print("=" * 70)
    print("DarkMind instruction dataset inspection")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Metadata: {metadata_path_for(data_path)}")
    print(f"Total examples: {len(rows):,}")
    print(f"Average prompt length: {avg_prompt_len:.1f}")
    print(f"Average response length: {avg_response_len:.1f}")
    print(f"Duplicate count: {duplicates:,}")
    print(f"Blocked identity phrase count: {blocked_identity:,}")
    print(f"External examples: {external_count:,} ({external_ratio:.1%})")
    print(f"Local examples: {local_count:,} ({local_ratio:.1%})")
    print(f"Local identity examples: {local_identity_count:,} ({local_identity_ratio:.1%})")
    print(f"Local fallback examples: {local_fallback_count:,} ({local_fallback_ratio:.1%})")
    print(f"Identity category ratio: {identity_ratio:.1%}")
    print(f"Require external: {require_external}")
    print(f"Min external ratio: {min_external_ratio:.1%}")
    print("-" * 70)
    print("Examples per source:")
    for source, count in source_counts.most_common():
        ratio = count / max(len(rows), 1)
        print(f"  {source}: {count:,} ({ratio:.1%})")
    print("-" * 70)
    print("Examples per category:")
    for category, count in category_counts.most_common():
        ratio = count / max(len(rows), 1)
        print(f"  {category}: {count:,} ({ratio:.1%})")
    print("-" * 70)
    print("Random samples:")
    for index, row in enumerate(samples, start=1):
        print(f"\n[{index}] source={row.get('source', 'unknown')} category={row.get('category', 'unknown')}")
        print(f"Prompt: {row.get('prompt', '')}")
        print(f"Response: {row.get('response', '')}")
    print("=" * 70)

    if forbidden_rows:
        print("Examples containing forbidden ChatGPT/OpenAI identity text:")
        for index, row in enumerate(forbidden_rows[:10], start=1):
            print(f"\n[{index}] source={row.get('source', 'unknown')} category={row.get('category', 'unknown')}")
            print(f"Prompt: {row.get('prompt', '')}")
            print(f"Response: {row.get('response', '')}")
        print("=" * 70)

    if failures:
        print("Quality failures:")
        for failure in failures:
            print(f"  {failure}")

    if args.fail_on_quality_issues and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
