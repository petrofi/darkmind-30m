from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ucekmez/OpenOrca-tr"
SPLIT = "train"
TARGET_EXAMPLES = 1000
OUTPUT_PATH = Path("data/instruct/openorca_tr_1k.jsonl")
META_PATH = Path("data/instruct/openorca_tr_1k.meta.json")

IDENTITY_PATTERNS = [
    re.compile(r"chatgpt", re.IGNORECASE),
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"as an ai language model", re.IGNORECASE),
    re.compile(r"i am chatgpt", re.IGNORECASE),
    re.compile(r"ben chatgpt", re.IGNORECASE),
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

    prompt = first_present(row, ["question", "prompt", "instruction"])
    response = first_present(row, ["answer", "response", "output"])
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
    if len(response) > 800:
        return "response_too_long"
    if len(prompt) > 500:
        return "prompt_too_long"
    if has_identity_phrase(identity_text):
        return "blocked_identity_phrase"

    reason = sensitive_reason(combined_text)
    if reason:
        return reason

    return None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    scanned_rows = 0
    accepted: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    rejection_reasons: Counter[str] = Counter()
    duplicate_count = 0
    blocked_identity_phrase_count = 0

    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
        for row in dataset:
            scanned_rows += 1
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
                    "source": DATASET_NAME,
                    "category": "other",
                }
            )

            if len(accepted) >= TARGET_EXAMPLES:
                break
    except Exception as exc:
        raise RuntimeError(f"Streaming load failed for {DATASET_NAME}: {exc}") from exc

    rejected_rows = scanned_rows - len(accepted)
    metadata = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "target_rows": TARGET_EXAMPLES,
        "scanned_rows": scanned_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": rejected_rows,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_count": duplicate_count,
        "blocked_identity_phrase_count": blocked_identity_phrase_count,
        "completed": len(accepted) >= TARGET_EXAMPLES,
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
