from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


SOURCES = [
    "ucekmez/OpenOrca-tr",
    "cgulse/alpaca-cleaned-tr",
    "TFLai/Turkish-Alpaca",
    "malhajar/alpaca-gpt4-tr",
]
TARGET_EXAMPLES = 1000
MAX_SCAN_ROWS_PER_SOURCE = 300000
OUTPUT_PATH = Path("data/instruct/software_instruct_1k.jsonl")
META_PATH = Path("data/instruct/software_instruct_1k.meta.json")

IDENTITY_PATTERNS = [
    re.compile(r"chatgpt", re.IGNORECASE),
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"as an ai language model", re.IGNORECASE),
    re.compile(r"i am chatgpt", re.IGNORECASE),
    re.compile(r"ben chatgpt", re.IGNORECASE),
]

STRONG_SOFTWARE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?<!\w)python['’]?d[ae](?!\w)",
        r"(?<!\w)python ile(?!\w)",
        r"(?<!\w)python kullanarak(?!\w)",
        r"(?<!\w)git\s+(?:commit|branch|push|pull|clone|merge|rebase|status|log)(?!\w)",
        r"(?<!\w)(?:commit|branch|push|pull|clone|merge|rebase)\s+git(?!\w)",
        r"(?<!\w)rest\s+api(?!\w)",
        r"(?<!\w)api\s+rest(?!\w)",
        r"(?<!\w)docker(?!\w)",
        r"(?<!\w)github(?!\w)",
        r"(?<!\w)api(?!\w)",
        r"(?<!\w)http(?!\w)",
        r"(?<!\w)json(?!\w)",
        r"(?<!\w)sql(?!\w)",
        r"(?<!\w)backend(?!\w)",
        r"(?<!\w)frontend(?!\w)",
        r"(?<!\w)c#(?!\w)",
        r"(?<!\w)\.net(?!\w)",
        r"(?<!\w)javascript(?!\w)",
        r"(?<!\w)typescript(?!\w)",
        r"(?<!\w)terminal(?!\w)",
        r"(?<!\w)error(?!\w)",
        r"(?<!\w)exception(?!\w)",
        r"(?<!\w)debugging(?!\w)",
        r"(?<!\w)debug(?!\w)",
        r"(?<!\w)linux(?!\w)",
        r"(?<!\w)virtual environment(?!\w)",
        r"(?<!\w)pip(?!\w)",
        r"(?<!\w)npm(?!\w)",
        r"(?<!\w)controller(?!\w)",
        r"(?<!\w)dto(?!\w)",
        r"(?<!\w)entity framework(?!\w)",
        r"(?<!\w)authentication(?!\w)",
        r"(?<!\w)jwt(?!\w)",
        r"(?<!\w)endpoint(?!\w)",
        r"(?<!\w)pytorch(?!\w)",
        r"(?<!\w)cuda(?!\w)",
        r"(?<!\w)tokenizer(?!\w)",
        r"(?<!\w)checkpoint(?!\w)",
        r"(?<!\w)validation loss(?!\w)",
        r"(?<!\w)model eğitimi(?!\w)",
        r"(?<!\w)model egitimi(?!\w)",
        r"(?<!\w)llm training(?!\w)",
        r"(?<!\w)yazılım\w*(?!\w)",
        r"(?<!\w)programlama\w*(?!\w)",
        r"(?<!\w)programc[ıi]\w*(?!\w)",
    ]
]

WEAK_SOFTWARE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?<!\w)python(?!\w)",
        r"(?<!\w)git(?!\w)",
        r"(?<!\w)rest(?!\w)",
        r"(?<!\w)database(?!\w)",
        r"veritaban[ıi]",
        r"hata\w*",
        r"(?<!\w)package(?!\w)",
        r"(?<!\w)dependency(?!\w)",
        r"(?<!\w)service(?!\w)",
        r"(?<!\w)server(?!\w)",
        r"(?<!\w)client(?!\w)",
        r"(?<!\w)request(?!\w)",
        r"(?<!\w)response(?!\w)",
    ]
]

SOFTWARE_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?<!\w)kod\w*(?!\w)",
        r"(?<!\w)programlama\w*(?!\w)",
        r"(?<!\w)yazılım\w*(?!\w)",
        r"(?<!\w)uygulama\w*(?!\w)",
        r"(?<!\w)script(?!\w)",
        r"(?<!\w)komut\w*(?!\w)",
        r"(?<!\w)terminal(?!\w)",
        r"(?<!\w)fonksiyon\w*(?!\w)",
        r"(?<!\w)sınıf\w*(?!\w)",
        r"(?<!\w)sinif\w*(?!\w)",
        r"(?<!\w)metot\w*(?!\w)",
        r"(?<!\w)değişken\w*(?!\w)",
        r"(?<!\w)degisken\w*(?!\w)",
        r"(?<!\w)algoritma\w*(?!\w)",
        r"(?<!\w)repo\w*(?!\w)",
        r"(?<!\w)repository(?!\w)",
        r"(?<!\w)derle\w*(?!\w)",
        r"(?<!\w)compile(?!\w)",
        r"(?<!\w)runtime(?!\w)",
        r"(?<!\w)framework(?!\w)",
    ]
]

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}(?!\w)"
)
API_KEY_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}"
)
KEY_LIKE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"
)
MOJIBAKE_RE = re.compile(r"Ã|Ä|Å|Ð|ð|�")
FALSE_POSITIVE_CONTEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"monty python",
        r"python ve kutsal kase",
        r"\btrna\b",
        r"\bmrna\b",
        r"amino asit",
        r"protein sentezi",
        r"ribozom",
        r"genetik kod",
    ]
]
DISALLOWED_CONTENT_PATTERNS = {
    "adult_content": re.compile(r"\b(porno|pornografi|erotik|çıplak|cinsel içerik)\b", re.IGNORECASE),
    "hate_or_harassment": re.compile(
        r"\b(ırkçı|irkci|nefret söylemi|aşağılayıcı|asagilayici hakaret)\b",
        re.IGNORECASE,
    ),
    "illegal_or_unsafe": re.compile(
        r"\b(bomba yap|silah yap|uyuşturucu üret|uyusturucu uret|hackle|çalınmış şifre|calinmis sifre)\b",
        re.IGNORECASE,
    ),
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def first_present(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = normalize_text(row.get(name))
        if value:
            return value
    return ""


def role_name(message: dict[str, Any]) -> str:
    for key in ("role", "from", "speaker", "author"):
        value = normalize_text(message.get(key)).lower()
        if value:
            return value
    return ""


def message_content(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        value = normalize_text(message.get(key))
        if value:
            return value
    return ""


def extract_from_messages(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""

    pending_user = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = role_name(item)
        content = message_content(item)
        if not content:
            continue
        if role in {"system", "context"}:
            continue
        if role in {"user", "human", "instruction", "prompt"}:
            pending_user = content
            continue
        if role in {"assistant", "gpt", "bot", "model", "response"} and pending_user:
            return pending_user, content

    return "", ""


def extract_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    for field_name in ("messages", "conversations"):
        prompt, response = extract_from_messages(row.get(field_name))
        if prompt and response:
            return prompt, response

    instruction = first_present(row, ["instruction", "question", "prompt", "soru", "input_text"])
    input_text = first_present(row, ["input", "context", "başlık", "baslik"])
    if instruction and input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction or input_text

    response = first_present(
        row,
        ["answer", "response", "output", "completion", "cevap", "yanıt", "yanit", "target"],
    )
    return prompt, response


def compact_row(row: Any) -> str:
    try:
        return json.dumps(row, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(row)


def has_identity_phrase(text: str) -> bool:
    return any(pattern.search(text) for pattern in IDENTITY_PATTERNS)


def sensitive_reason(text: str) -> str | None:
    if URL_RE.search(text):
        return "url"
    if EMAIL_RE.search(text):
        return "email"
    if PHONE_RE.search(text):
        return "phone"
    if API_KEY_RE.search(text) or KEY_LIKE_RE.search(text):
        return "secret_or_api_key"
    return None


def has_software_keyword(prompt: str, response: str) -> bool:
    combined_text = f"{prompt}\n{response}"
    if any(pattern.search(combined_text) for pattern in FALSE_POSITIVE_CONTEXT_PATTERNS):
        return False
    if any(pattern.search(combined_text) for pattern in STRONG_SOFTWARE_PATTERNS):
        return True
    has_weak_term = any(pattern.search(combined_text) for pattern in WEAK_SOFTWARE_PATTERNS)
    has_context = any(pattern.search(combined_text) for pattern in SOFTWARE_CONTEXT_PATTERNS)
    return has_weak_term and has_context


def looks_like_name_list(text: str) -> bool:
    stripped = text.strip(" -•*0123456789.,;:\n\t")
    if not stripped:
        return False

    separator_count = sum(text.count(separator) for separator in [",", ";", "\n", "•"])
    if separator_count < 3:
        return False
    if re.search(r"[.!?]", text):
        return False

    parts = [part.strip(" -•*0123456789.)(") for part in re.split(r"[,;\n•]+", text)]
    parts = [part for part in parts if part]
    if len(parts) < 4:
        return False

    short_parts = sum(1 for part in parts if len(part.split()) <= 4)
    return short_parts / len(parts) >= 0.85


def looks_broken_turkish(text: str) -> bool:
    if MOJIBAKE_RE.search(text):
        return True
    replacement_count = text.count("\ufffd")
    return replacement_count >= 2


def disallowed_content_reason(text: str) -> str | None:
    for reason, pattern in DISALLOWED_CONTENT_PATTERNS.items():
        if pattern.search(text):
            return reason
    return None


def reject_reason(row: dict[str, Any], prompt: str, response: str) -> str | None:
    row_text = compact_row(row)
    combined_text = f"{prompt}\n{response}"
    identity_text = f"{row_text}\n{combined_text}"

    if not prompt:
        return "empty_prompt"
    if not response:
        return "empty_response"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 1000:
        return "response_too_long"
    if len(prompt) > 600:
        return "prompt_too_long"
    if has_identity_phrase(identity_text):
        return "blocked_identity_phrase"

    reason = sensitive_reason(identity_text)
    if reason:
        return reason
    reason = disallowed_content_reason(combined_text)
    if reason:
        return reason

    if looks_like_name_list(prompt) or looks_like_name_list(response):
        return "name_list"
    if looks_broken_turkish(combined_text):
        return "broken_turkish"
    if not has_software_keyword(prompt, response):
        return "not_software_related"

    return None


def load_streaming_dataset(source: str) -> tuple[Any, str]:
    try:
        return load_dataset(source, split="train", streaming=True), "train"
    except Exception as split_error:
        try:
            dataset = load_dataset(source, streaming=True)
        except Exception as dict_error:
            raise RuntimeError(
                f"streaming load failed with split='train': {split_error}; "
                f"streaming load without split also failed: {dict_error}"
            ) from dict_error

        if not hasattr(dataset, "keys"):
            raise RuntimeError(f"streaming load returned unsupported object: {type(dataset)!r}")

        split_names = list(dataset.keys())
        if not split_names:
            raise RuntimeError("streaming load returned no splits")
        split_name = "train" if "train" in split_names else split_names[0]
        return dataset[split_name], split_name


def scan_source(
    source: str,
    accepted: list[dict[str, str]],
    seen: set[tuple[str, str]],
    scanned_rows_per_dataset: Counter[str],
    accepted_rows_per_dataset: Counter[str],
    rejection_reasons: Counter[str],
) -> tuple[int, int, int]:
    duplicate_count = 0
    blocked_identity_phrase_count = 0

    dataset, split_name = load_streaming_dataset(source)
    print(f"Scanning {source} [{split_name}] with streaming=True")

    for row in dataset:
        scanned_rows_per_dataset[source] += 1
        if scanned_rows_per_dataset[source] > MAX_SCAN_ROWS_PER_SOURCE:
            rejection_reasons["source_scan_limit_reached"] += 1
            break

        if not isinstance(row, dict):
            rejection_reasons["invalid_row"] += 1
            continue

        prompt, response = extract_prompt_response(row)
        prompt = normalize_text(prompt)
        response = normalize_text(response)

        reason = reject_reason(row, prompt, response)
        if reason:
            rejection_reasons[reason] += 1
            if reason == "blocked_identity_phrase":
                blocked_identity_phrase_count += 1
            continue

        duplicate_key = (prompt.casefold(), response.casefold())
        if duplicate_key in seen:
            duplicate_count += 1
            rejection_reasons["duplicate"] += 1
            continue
        seen.add(duplicate_key)

        accepted.append(
            {
                "prompt": prompt,
                "response": response,
                "source": source,
                "category": "programming",
            }
        )
        accepted_rows_per_dataset[source] += 1

        if len(accepted) >= TARGET_EXAMPLES:
            break

    return scanned_rows_per_dataset[source], duplicate_count, blocked_identity_phrase_count


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned_rows_per_dataset: Counter[str] = Counter()
    accepted_rows_per_dataset: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    source_errors: dict[str, str] = {}
    duplicate_count = 0
    blocked_identity_phrase_count = 0

    for source in SOURCES:
        if len(accepted) >= TARGET_EXAMPLES:
            break
        try:
            _, source_duplicates, source_identity_blocks = scan_source(
                source,
                accepted,
                seen,
                scanned_rows_per_dataset,
                accepted_rows_per_dataset,
                rejection_reasons,
            )
            duplicate_count += source_duplicates
            blocked_identity_phrase_count += source_identity_blocks
        except Exception as exc:
            source_errors[source] = str(exc)
            rejection_reasons["source_load_or_scan_error"] += 1
            print(f"Source failed: {source}: {exc}")

    scanned_rows = sum(scanned_rows_per_dataset.values())
    metadata = {
        "sources": SOURCES,
        "target_rows": TARGET_EXAMPLES,
        "scanned_rows": scanned_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": scanned_rows - len(accepted),
        "scanned_rows_per_dataset": dict(scanned_rows_per_dataset),
        "accepted_rows_per_dataset": dict(accepted_rows_per_dataset),
        "rejected_rows_per_reason": dict(sorted(rejection_reasons.items())),
        "duplicate_count": rejection_reasons.get("duplicate", duplicate_count),
        "blocked_identity_phrase_count": rejection_reasons.get(
            "blocked_identity_phrase",
            blocked_identity_phrase_count,
        ),
        "completed": len(accepted) >= TARGET_EXAMPLES,
        "first_10_samples": accepted[:10],
        "source_errors": source_errors,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in accepted:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    with META_PATH.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
