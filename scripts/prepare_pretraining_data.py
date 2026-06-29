from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from pathlib import Path
import argparse
import hashlib
import json
import random
import re
import sys
from typing import Iterable

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "pretrain_corpus.jsonl"

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
SECRET_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|credential|private[_-]?key)",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t]+")
BOILERPLATE_LINE_RE = re.compile(
    r"(cookie|privacy policy|gizlilik|login|sign in|subscribe|newsletter|"
    r"advertisement|reklam|terms of service|kullanım şartları)",
    re.IGNORECASE,
)
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_HINTS = {
    "bir",
    "ve",
    "için",
    "nasıl",
    "nedir",
    "değil",
    "olarak",
    "olan",
    "çok",
}
ENGLISH_HINTS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "are",
    "language",
}


@dataclass(frozen=True)
class SourcePlan:
    name: str
    dataset: str
    category: str
    language: str
    config_language: str | None = None
    preferred_configs: tuple[str, ...] = ()


@dataclass
class LoadedPlan:
    plan: SourcePlan
    dataset: Iterable
    config: str | None


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def import_datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install data requirements first: "
            "pip install -r requirements-data.txt"
        ) from exc

    return get_dataset_config_names, load_dataset


def optional_fix_text(text: str) -> str:
    try:
        import ftfy
    except ImportError:
        return text

    return ftfy.fix_text(text)


def strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return HTML_TAG_RE.sub(" ", text)

    return BeautifulSoup(text, "html.parser").get_text(" ")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    seen_lines = set()

    for raw_line in text.split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()

        if not line:
            continue

        if BOILERPLATE_LINE_RE.search(line):
            continue

        line_key = " ".join(line.casefold().split())

        if line_key in seen_lines:
            continue

        seen_lines.add(line_key)
        lines.append(line)

    return "\n".join(lines).strip()


def clean_text(text: str) -> str:
    text = unescape(str(text))
    text = optional_fix_text(text)
    text = strip_html(text)
    return normalize_whitespace(text)


def too_many_urls(text: str) -> bool:
    return len(URL_RE.findall(text)) > 3


def contains_private_or_secret_data(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or SECRET_RE.search(text)
    )


def non_text_ratio(text: str) -> float:
    if not text:
        return 1.0

    allowed_punctuation = set(".,;:!?()[]{}'\"`-_/\\+*=<>#%&|@$€₺\n ")
    non_text = 0

    for char in text:
        if char.isalnum() or char.isspace() or char in allowed_punctuation:
            continue

        non_text += 1

    return non_text / max(len(text), 1)


def simple_language_guess(text: str) -> str:
    lowered = text.casefold()
    words = set(re.findall(r"\b\w+\b", lowered))

    if any(char in text for char in TURKISH_CHARS) or len(words & TURKISH_HINTS) >= 2:
        return "tr"

    if len(words & ENGLISH_HINTS) >= 2:
        return "en"

    return "unknown"


def detect_language(text: str, fallback_language: str) -> str:
    if fallback_language == "code":
        return "code"

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42
        detected = detect(text[:2000])

        if detected in {"tr", "en"}:
            return detected
    except Exception:
        pass

    guessed = simple_language_guess(text)
    return guessed if guessed != "unknown" else fallback_language


def text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_text(row: dict) -> str:
    for field in [
        "text",
        "content",
        "raw_content",
        "markdown",
        "code",
        "body",
        "article",
    ]:
        value = row.get(field)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def choose_config(
    dataset_name: str,
    plan: SourcePlan,
    get_dataset_config_names,
) -> str | None:
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as exc:
        print(f"WARNING: could not list configs for {dataset_name}: {exc}")
        return None

    if not configs:
        return None

    for preferred in plan.preferred_configs:
        for config in configs:
            if preferred.casefold() == config.casefold():
                return config

    language = plan.config_language or plan.language

    if language in {"tr", "en"}:
        language_patterns = [
            f".{language}",
            f"_{language}",
            f"-{language}",
            language,
        ]

        for config in configs:
            lowered = config.casefold()

            if any(lowered.endswith(pattern) for pattern in language_patterns):
                return config

        for config in configs:
            lowered = config.casefold()

            if any(pattern in lowered for pattern in language_patterns):
                return config

    for keyword in ["sample", "default", "python", "train"]:
        for config in configs:
            if keyword in config.casefold():
                return config

    return configs[0]


def load_stream(plan: SourcePlan, load_dataset, get_dataset_config_names):
    config = choose_config(plan.dataset, plan, get_dataset_config_names)

    try:
        if config:
            dataset = load_dataset(
                plan.dataset,
                config,
                split="train",
                streaming=True,
            )
        else:
            dataset = load_dataset(
                plan.dataset,
                split="train",
                streaming=True,
            )
    except Exception as exc:
        print(
            f"WARNING: skipped {plan.name} ({plan.dataset}) "
            f"because it could not be loaded: {exc}"
        )
        return None, config

    return dataset, config


def build_source_plans(sources: list[str], languages: list[str]) -> list[SourcePlan]:
    plans: list[SourcePlan] = []

    if "wikipedia" in sources:
        if "tr" in languages:
            plans.append(
                SourcePlan(
                    name="wikipedia_tr",
                    dataset="wikimedia/wikipedia",
                    category="tr_text",
                    language="tr",
                    config_language="tr",
                    preferred_configs=("20231101.tr", "20220301.tr"),
                )
            )

        if "en" in languages:
            plans.append(
                SourcePlan(
                    name="wikipedia_en",
                    dataset="wikimedia/wikipedia",
                    category="en_text",
                    language="en",
                    config_language="en",
                    preferred_configs=("20231101.en", "20220301.en"),
                )
            )

    if "fineweb" in sources and "en" in languages:
        plans.append(
            SourcePlan(
                name="fineweb_edu_en",
                dataset="HuggingFaceFW/fineweb-edu",
                category="en_text",
                language="en",
                preferred_configs=("sample-10BT", "sample-100BT"),
            )
        )

    if "redpajama" in sources and "en" in languages:
        plans.append(
            SourcePlan(
                name="redpajama_v2_en",
                dataset="togethercomputer/RedPajama-Data-V2",
                category="en_text",
                language="en",
                preferred_configs=("sample", "default"),
            )
        )

    if "stack" in sources:
        plans.append(
            SourcePlan(
                name="the_stack_code",
                dataset="bigcode/the-stack-v2",
                category="code",
                language="code",
                preferred_configs=("Python", "python"),
            )
        )

    return plans


def compute_plan_quotas(plans: list[SourcePlan], max_docs: int) -> dict[str, int]:
    if not plans:
        return {}

    base_weights = {
        "tr_text": 0.60,
        "en_text": 0.25,
        "code": 0.15,
    }
    present_categories = {plan.category for plan in plans}
    total_weight = sum(base_weights[category] for category in present_categories)
    category_targets = {
        category: int(max_docs * (base_weights[category] / total_weight))
        for category in present_categories
    }
    leftover = max_docs - sum(category_targets.values())

    for category in sorted(present_categories):
        if leftover <= 0:
            break

        category_targets[category] += 1
        leftover -= 1

    plans_by_category: dict[str, list[SourcePlan]] = defaultdict(list)

    for plan in plans:
        plans_by_category[plan.category].append(plan)

    quotas = {}

    for category, category_plans in plans_by_category.items():
        target = category_targets[category]
        per_plan = target // len(category_plans)
        remainder = target % len(category_plans)

        for index, plan in enumerate(category_plans):
            quotas[plan.name] = per_plan + (1 if index < remainder else 0)

    return quotas


def is_accepted_text(
    text: str,
    min_chars: int,
    max_chars: int,
    allowed_languages: set[str],
    source_language: str,
    seen_hashes: set[str],
) -> tuple[bool, str, str]:
    if len(text) < min_chars:
        return False, "too_short", source_language

    if len(text) > max_chars:
        text = text[:max_chars].strip()

    if too_many_urls(text):
        return False, "too_many_urls", source_language

    if contains_private_or_secret_data(text):
        return False, "private_or_secret_pattern", source_language

    if non_text_ratio(text) > 0.20:
        return False, "too_many_non_text_chars", source_language

    language = detect_language(text, source_language)

    if language not in allowed_languages and language != "code":
        return False, f"language_{language}", language

    digest = text_hash(text)

    if digest in seen_hashes:
        return False, "duplicate_exact_text", language

    seen_hashes.add(digest)
    return True, "accepted", language


def iter_plan_records(
    plan: SourcePlan,
    dataset,
    config: str | None,
    quota: int,
    args,
    seen_hashes: set[str],
) -> Iterable[dict]:
    accepted = 0
    skipped = Counter()
    progress = tqdm(total=quota, desc=plan.name, unit="doc", leave=False)

    for row in dataset:
        if accepted >= quota:
            break

        raw_text = extract_text(row)

        if not raw_text:
            skipped["missing_text"] += 1
            continue

        text = clean_text(raw_text)
        ok, reason, language = is_accepted_text(
            text=text,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            allowed_languages=set(args.languages),
            source_language=plan.language,
            seen_hashes=seen_hashes,
        )

        if not ok:
            skipped[reason] += 1
            continue

        accepted += 1
        progress.update(1)
        yield {
            "text": text,
            "source": plan.name,
            "language": language,
        }

    progress.close()
    print(
        f"{plan.name}: accepted={accepted} quota={quota} "
        f"config={config or 'default'} skipped={dict(skipped)}"
    )


def load_existing_output(path: Path) -> tuple[set[str], Counter]:
    hashes: set[str] = set()
    source_counts: Counter = Counter()

    if not path.exists():
        return hashes, source_counts

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = row.get("text", "")
            source = row.get("source", "unknown")

            if isinstance(text, str) and text.strip():
                hashes.add(text_hash(text))
                source_counts[source] += 1

    return hashes, source_counts


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "a" if append else "w"

    with path.open(mode, encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a small legal streaming pretraining JSONL corpus."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--max_docs", type=int, default=50000)
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=6000)
    parser.add_argument(
        "--sources",
        type=str,
        default="wikipedia,fineweb,redpajama,stack",
    )
    parser.add_argument("--languages", type=str, default="tr,en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip exact duplicate texts.",
    )
    args = parser.parse_args()

    args.sources = parse_csv(args.sources)
    args.languages = parse_csv(args.languages)

    if args.max_docs < 1:
        raise ValueError("--max_docs must be at least 1")

    random.seed(args.seed)
    get_dataset_config_names, load_dataset = import_datasets()
    plans = build_source_plans(args.sources, args.languages)

    if not plans:
        raise ValueError("No compatible source plans were selected.")

    output_path = resolve_path(args.out)
    loaded_plans: list[LoadedPlan] = []

    for plan in plans:
        dataset, config = load_stream(plan, load_dataset, get_dataset_config_names)

        if dataset is not None:
            loaded_plans.append(LoadedPlan(plan=plan, dataset=dataset, config=config))

    if not loaded_plans:
        raise RuntimeError("No selected datasets were accessible.")

    quotas = compute_plan_quotas(
        [loaded_plan.plan for loaded_plan in loaded_plans],
        args.max_docs,
    )
    seen_hashes, existing_source_counts = (
        load_existing_output(output_path)
        if args.resume
        else (set(), Counter())
    )

    if args.resume and existing_source_counts:
        for source_name, existing_count in existing_source_counts.items():
            if source_name in quotas:
                quotas[source_name] = max(0, quotas[source_name] - existing_count)

    def rows():
        for loaded_plan in loaded_plans:
            plan = loaded_plan.plan
            quota = quotas.get(plan.name, 0)

            if quota <= 0:
                continue

            yield from iter_plan_records(
                plan,
                loaded_plan.dataset,
                loaded_plan.config,
                quota,
                args,
                seen_hashes,
            )

    written = write_jsonl(output_path, rows(), append=args.resume)
    existing_total = sum(existing_source_counts.values())

    print("=" * 70)
    print("DarkMind pretraining data preparation")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Requested max docs: {args.max_docs:,}")
    print(f"Existing docs reused: {existing_total:,}")
    print(f"New docs written: {written:,}")
    print(f"Total docs now: {existing_total + written:,}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Languages: {', '.join(args.languages)}")
    print("Quotas:")

    for name, quota in sorted(quotas.items()):
        print(f"- {name}: {quota:,}")

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
