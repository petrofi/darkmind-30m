from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATHS = [
    ROOT_DIR / "data" / "instruct" / "openorca_tr_1k.jsonl",
    ROOT_DIR / "data" / "instruct" / "software_instruct_1k.jsonl",
    ROOT_DIR / "data" / "instruct" / "support_instruct_1k.jsonl",
    ROOT_DIR / "data" / "instruct" / "identity_anchor_300.jsonl",
    ROOT_DIR / "data" / "instruct" / "darkmind_instruct_seed.jsonl",
]
OUTPUT_PATH = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_mix_v0_4.jsonl"
META_PATH = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_mix_v0_4.meta.json"
TARGET_RATIOS = {
    "programming": 0.40,
    "ai": 0.15,
    "emotional_support": 0.15,
    "general": 0.20,
    "identity": 0.10,
}
MAX_IDENTITY_RATIO = 0.12
MAX_TARGET_ROWS = 2500
SEED = 42

IDENTITY_PATTERNS = [
    re.compile(r"\bben\s+chatgpt\b", re.IGNORECASE),
    re.compile(r"\bchatgpt\b", re.IGNORECASE),
    re.compile(r"\bopenai\b", re.IGNORECASE),
    re.compile(r"as an ai language model", re.IGNORECASE),
    re.compile(r"developed by openai|trained by openai", re.IGNORECASE),
]
SECRET_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}(?!\w)"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}"),
    re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
]
DISALLOWED_PATTERNS = [
    re.compile(r"\b(porno|pornografi|erotik|çıplak|cinsel içerik)\b", re.IGNORECASE),
    re.compile(r"\b(bomba yap|silah yap|uyuşturucu üret|uyusturucu uret|çalınmış şifre|calinmis sifre)\b", re.IGNORECASE),
    re.compile(r"\b(intihar yöntemi|kendime zarar yöntemi|kendimi öldürme)\b", re.IGNORECASE),
]
BROKEN_TURKISH_RE = re.compile(r"Ã|Ä|Å|Ð|ð|�")
AI_RE = re.compile(
    r"\b(yapay zeka|yapay zekâ|dil modeli|llm|tokenizer|transformer|attention|"
    r"checkpoint|validation loss|model eğitimi|model egitimi|pytorch|cuda|"
    r"fine-?tuning|embedding)\b",
    re.IGNORECASE,
)
SOFTWARE_STRONG_RE = re.compile(
    r"\b(python|docker|github|api|http|json|sql|backend|frontend|javascript|"
    r"typescript|terminal|error|exception|debug|debugging|linux|pip|npm|"
    r"controller|dto|authentication|jwt|endpoint|pytorch|cuda|tokenizer|"
    r"checkpoint|programlama|yazılım|yazilim)\b|\.net|c#|entity framework",
    re.IGNORECASE,
)
SOFTWARE_FALSE_POSITIVE_RE = re.compile(
    r"monty python|python ve kutsal kase|\btrna\b|\bmrna\b|amino asit|protein sentezi|ribozom|genetik kod",
    re.IGNORECASE,
)
SUPPORT_PROMPT_RE = re.compile(
    r"kendimi .{0,30}(kötü|kotu|yetersiz|yalnız|yalniz|mutsuz|üzgün|uzgun)|"
    r"moralim .{0,30}(bozuk|düşük|dusuk)|motivasyon|çok yoruldum|cok yoruldum|"
    r"tükendim|tukendim|devam edemiyorum|odaklanamıyorum|odaklanamiyorum|"
    r"iş bulamıyorum|is bulamiyorum|projem .{0,40}(ilerlemiyor|takıldı|takildi)|"
    r"kaygılıyım|kaygiliyim|endişeliyim|endiseliyim|stresliyim|"
    r"hata aldıkça|hata aldikca|kendime güvenim|kendime guvenim",
    re.IGNORECASE,
)
SUPPORT_RESPONSE_RE = re.compile(
    r"küçük|kucuk|mola|nefes|not al|plan yap|parçalara böl|parcalara bol|"
    r"yalnız değilsin|yalniz degilsin|zor olabilir|anlaşılır|anlasilir|destek",
    re.IGNORECASE,
)
SUPPORT_TASK_NOISE_RE = re.compile(
    r"çevir|cevir|türkçeye|turkceye|ingilizceye|seçenekler|secenekler|"
    r"çoktan seçmeli|coktan secmeli|soru:|bağlam:|baglam:|inceleme|özet|ozet|"
    r"aynı anlama|ayni anlama|diyalog|makale|doğru mu|dogru mu",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["prompt"].casefold(), row["response"].casefold()


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            prompt = normalize_text(raw.get("prompt"))
            response = normalize_text(raw.get("response"))
            source = normalize_text(raw.get("source")) or path.stem
            category = normalize_text(raw.get("category")) or "general"
            rows.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "source": source,
                    "category": category,
                }
            )
    return rows


def allowed_identity_exception(row: dict[str, str]) -> bool:
    source = row["source"]
    if source not in {"local_identity", "local_identity_anchor"}:
        return False

    prompt = row["prompt"].lower()
    response = row["response"].lower()
    asks_other_model = "chatgpt" in prompt or "openai" in prompt or "gpt" in prompt
    denies_other_model = (
        "hayır" in response
        or "hayir" in response
        or "değilim" in response
        or "degilim" in response
    )
    return asks_other_model and denies_other_model and "darkmind" in response


def has_identity_contamination(row: dict[str, str]) -> bool:
    if allowed_identity_exception(row):
        response = row["response"].lower()
        return "ben chatgpt" in response or "i am chatgpt" in response

    text = f"{row['prompt']}\n{row['response']}"
    return any(pattern.search(text) for pattern in IDENTITY_PATTERNS)


def infer_category(row: dict[str, str]) -> str:
    text = f"{row['prompt']}\n{row['response']}"
    category = row["category"]

    if category == "identity" or row["source"] in {"local_identity", "local_identity_anchor"}:
        return "identity"
    if category == "emotional_support" or row["source"] == "local_support_seed":
        return "emotional_support"
    if AI_RE.search(text):
        return "ai"
    if category == "programming" or SOFTWARE_STRONG_RE.search(text):
        return "programming"
    return "general"


def clean_enough(row: dict[str, str], category: str) -> str | None:
    prompt = row["prompt"]
    response = row["response"]
    text = f"{prompt}\n{response}"

    if not prompt:
        return "empty_prompt"
    if not response:
        return "empty_response"
    if len(prompt) > 700:
        return "prompt_too_long"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 1200:
        return "response_too_long"
    if BROKEN_TURKISH_RE.search(text):
        return "broken_turkish"
    if has_identity_contamination(row):
        return "blocked_identity_phrase"
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return "secret_or_private_data"
    if any(pattern.search(text) for pattern in DISALLOWED_PATTERNS):
        return "unsafe_content"
    if category == "programming" and SOFTWARE_FALSE_POSITIVE_RE.search(text):
        return "software_false_positive"
    if category == "programming" and not SOFTWARE_STRONG_RE.search(text):
        return "not_programming_related"
    if category == "emotional_support":
        if SUPPORT_TASK_NOISE_RE.search(prompt):
            return "support_task_noise"
        if not (SUPPORT_PROMPT_RE.search(prompt) and SUPPORT_RESPONSE_RE.search(response)):
            return "not_support_related"
    return None


def pick_rows(pool: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = random.Random(SEED)
    raw_rows: list[dict[str, str]] = []
    input_counts: Counter[str] = Counter()
    missing_inputs: list[str] = []

    for path in INPUT_PATHS:
        rows = load_rows(path)
        if not rows:
            missing_inputs.append(str(path.relative_to(ROOT_DIR)))
            continue
        input_counts[str(path.relative_to(ROOT_DIR))] = len(rows)
        raw_rows.extend(rows)

    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    rejected_rows_per_reason: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for row in raw_rows:
        category = infer_category(row)
        cleaned = {
            "prompt": row["prompt"],
            "response": row["response"],
            "source": row["source"],
            "category": category,
        }
        reason = clean_enough(cleaned, category)
        if reason:
            rejected_rows_per_reason[reason] += 1
            continue
        key = row_key(cleaned)
        if key in seen:
            rejected_rows_per_reason["duplicate"] += 1
            continue
        seen.add(key)
        pools[category].append(cleaned)

    available_counts = {category: len(rows) for category, rows in pools.items()}
    feasible_totals = [MAX_TARGET_ROWS]
    for category, ratio in TARGET_RATIOS.items():
        available = len(pools.get(category, []))
        if available > 0:
            feasible_totals.append(int(available / ratio))
    target_total = max(1, min(feasible_totals))

    desired_counts = {
        category: int(target_total * ratio)
        for category, ratio in TARGET_RATIOS.items()
    }
    desired_counts["identity"] = min(
        desired_counts["identity"],
        int(target_total * MAX_IDENTITY_RATIO),
    )

    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str]] = set()
    shortages: dict[str, int] = {}

    for category in ["programming", "ai", "emotional_support", "general", "identity"]:
        wanted = desired_counts.get(category, 0)
        picked = pick_rows(pools.get(category, []), wanted, rng)
        if len(picked) < wanted:
            shortages[category] = wanted - len(picked)
        for row in picked:
            key = row_key(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(row)

    fill_order = ["programming", "ai", "general", "emotional_support"]
    target_without_identity_pressure = min(MAX_TARGET_ROWS, sum(len(rows) for rows in pools.values()))
    for category in fill_order:
        if len(selected) >= target_without_identity_pressure:
            break
        for row in pick_rows(pools.get(category, []), len(pools.get(category, [])), rng):
            key = row_key(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(row)
            if len(selected) >= target_without_identity_pressure:
                break

    rng.shuffle(selected)
    category_counts = Counter(row["category"] for row in selected)
    source_counts = Counter(row["source"] for row in selected)
    identity_ratio = category_counts.get("identity", 0) / max(len(selected), 1)
    if identity_ratio > MAX_IDENTITY_RATIO:
        raise ValueError(f"identity ratio {identity_ratio:.2%} exceeds {MAX_IDENTITY_RATIO:.2%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in selected:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "inputs": [str(path.relative_to(ROOT_DIR)) for path in INPUT_PATHS],
        "missing_inputs": missing_inputs,
        "input_counts": dict(input_counts),
        "target_ratios": TARGET_RATIOS,
        "max_identity_ratio": MAX_IDENTITY_RATIO,
        "selected_rows": len(selected),
        "available_rows_by_category_after_filter": available_counts,
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "identity_ratio": identity_ratio,
        "rejected_rows_per_reason": dict(sorted(rejected_rows_per_reason.items())),
        "duplicate_count": rejected_rows_per_reason.get("duplicate", 0),
        "blocked_identity_phrase_count": rejected_rows_per_reason.get("blocked_identity_phrase", 0),
        "shortages": shortages,
        "first_10_samples": selected[:10],
        "completed": (
            len(selected) > 0
            and category_counts.get("programming", 0) > 0
            and category_counts.get("emotional_support", 0) > 0
            and identity_ratio <= MAX_IDENTITY_RATIO
        ),
        "require_external": False,
        "min_external_ratio": 0.0,
        "local_sources_allowed": ["local_identity_anchor", "local_support_seed"],
    }
    with META_PATH.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
