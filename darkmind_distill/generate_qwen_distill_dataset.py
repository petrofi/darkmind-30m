from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DISTILL_DIR = ROOT_DIR / "darkmind_distill"
CONFIG_PATH = DISTILL_DIR / "config.json"
CONFIG_EXAMPLE_PATH = DISTILL_DIR / "config.example.json"
TEACHER_SYSTEM_PROMPT_PATH = DISTILL_DIR / "teacher_system_prompt.md"
SOURCE = "qwen3_vl_30b_teacher"

LANGUAGE_TARGETS = {
    "tr": 900,
    "en": 500,
    "de": 200,
    "ja": 200,
    "es": 100,
    "fr": 100,
}
LANGUAGE_NAMES = {
    "tr": "Turkish",
    "en": "English",
    "de": "German",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
}
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
DIFFICULTY_TARGETS = {
    "beginner": 1000,
    "intermediate": 800,
    "advanced": 200,
}

CATEGORY_TOPICS = {
    "programming": "Python, C#, JavaScript, TypeScript, Java basics, algorithms, data structures, short code.",
    "debugging": "ModuleNotFoundError, CUDA out of memory, git push rejected, port already in use, SyntaxError, Docker exits, .NET build errors.",
    "backend": "C# .NET Web API, REST API, controller/service/repository, DTO, Entity Framework, SQL, auth basics, HTTP status codes, Docker.",
    "ai_llm": "token, tokenizer, pretraining, instruction tuning, validation loss, overfitting, checkpoint, transformer, attention, embedding, temperature, top_p, quantization, LoRA, QLoRA, RAG, distillation.",
    "emotional_support": "safe short support for learning stress, low motivation, code errors, job search frustration, project fatigue.",
    "identity": "DarkMind identity, ChatGPT/OpenAI/Qwen denial, creator identity.",
    "general_reasoning": "simple reasoning, summarization, planning, practical tradeoffs, no obscure trivia.",
    "data_json_sql": "JSON, JSONL, CSV, SQL, schema, API payloads, validation, data cleaning.",
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
SECRET_RE = re.compile(
    r"\b(?:api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}|"
    r"\b(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b",
    re.IGNORECASE,
)
PRIVATE_DATA_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}(?!\w)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
UNSAFE_CYBER_RE = re.compile(
    r"phishing|keylogger|ransomware|credential theft|steal password|şifre çal|sifre cal|"
    r"exploit yaz|zararlı yazılım|zararli yazilim|sql injection ile saldır|ddos",
    re.IGNORECASE,
)
ILLEGAL_RE = re.compile(r"sahte kimlik|uyuşturucu üret|uyusturucu uret|bomba yap|silah yap", re.IGNORECASE)
MEDICAL_THERAPY_RE = re.compile(
    r"tanı koy|teşhis|tedavi planı|terapi yerine|psikolog yerine|medical diagnosis|diagnose you|therapy replacement",
    re.IGNORECASE,
)
SELF_HARM_RE = re.compile(r"intihar yöntemi|kendine zarar verme yöntemi|self-harm method|how to harm yourself", re.IGNORECASE)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_config() -> dict[str, Any]:
    path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["_loaded_from"] = str(path)
    return config


def load_teacher_prompt() -> str:
    with TEACHER_SYSTEM_PROMPT_PATH.open("r", encoding="utf-8") as file:
        return file.read().strip()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_key(row: dict[str, str]) -> tuple[str, str]:
    return row["prompt"].casefold(), row["response"].casefold()


def scale_targets(base: dict[str, int], limit: int | None) -> dict[str, int]:
    total = sum(base.values())
    if limit is None or limit >= total:
        return dict(base)
    if limit < 1:
        raise ValueError("--limit must be at least 1")

    raw = {key: value * limit / total for key, value in base.items()}
    scaled = {key: int(value) for key, value in raw.items()}
    remaining = limit - sum(scaled.values())
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if remaining <= 0:
            break
        scaled[key] += 1
        remaining -= 1
    return {key: value for key, value in scaled.items() if value > 0}


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "prompt": normalize_text(raw.get("prompt")),
                    "response": normalize_text(raw.get("response")),
                    "source": normalize_text(raw.get("source")) or SOURCE,
                    "category": normalize_text(raw.get("category")),
                    "language": normalize_text(raw.get("language")),
                    "difficulty": normalize_text(raw.get("difficulty")),
                }
            )
    return rows


def write_jsonl_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_teacher_prompt(category: str, language: str, difficulty: str, count: int, strict: bool) -> str:
    language_name = LANGUAGE_NAMES[language]
    stricter = (
        "Previous output was invalid. Return only a valid JSON array. No markdown, no comments, no code fences."
        if strict
        else "Return only a valid JSON array. No markdown, no comments, no code fences."
    )
    return f"""
{stricter}

Generate {count} original {language_name} instruction-response examples for category={category}, difficulty={difficulty}.
Each item must include prompt, response, source, category, language, difficulty.

Required fixed fields:
- source: "{SOURCE}"
- category: "{category}"
- language: "{language}"
- difficulty: "{difficulty}"

Category focus: {CATEGORY_TOPICS[category]}

Rules:
- Use {language_name} exactly.
- Responses must be 1-4 sentences.
- Keep answers practical and accurate.
- Include code only if short.
- Do not make fake benchmark claims.
- Do not include private data, passwords, tokens, secrets, or API keys.
- Do not include URL-heavy content.
- Do not claim DarkMind is ChatGPT, OpenAI, or Qwen.
- DarkMind identity only appears in identity examples.
- For emotional_support: validate the feeling briefly and suggest one small next step; no diagnosis or therapy claim.
- For debugging: explain likely cause and give exact next steps; prefer Windows PowerShell when relevant.
- For identity: only identity questions; include DarkMind identity and denials for ChatGPT/OpenAI/Qwen.

Example shape:
[
  {{"prompt":"...","response":"...","source":"{SOURCE}","category":"{category}","language":"{language}","difficulty":"{difficulty}"}}
]
""".strip()


def call_teacher(config: dict[str, Any], teacher_system_prompt: str, user_prompt: str) -> str:
    if config.get("teacher_backend") != "openai_compatible":
        raise ValueError("Only teacher_backend=openai_compatible is supported for v0.1")

    base_url = str(config.get("teacher_base_url", "http://localhost:1234/v1")).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": config.get("teacher_model", "local-model"),
        "messages": [
            {"role": "system", "content": teacher_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.get("temperature", 0.4),
        "top_p": config.get("top_p", 0.9),
        "max_tokens": config.get("max_tokens", 1024),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('teacher_api_key', 'lm-studio')}",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def parse_json_array(text: str) -> list[Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Teacher output is not a JSON array")
    return parsed


def normalize_candidate(raw: Any, category: str, language: str, difficulty: str) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "prompt": normalize_text(raw.get("prompt")),
        "response": normalize_text(raw.get("response")),
        "source": SOURCE,
        "category": category,
        "language": language,
        "difficulty": difficulty,
    }


def rejection_reason(row: dict[str, str]) -> str | None:
    prompt = row["prompt"]
    response = row["response"]
    category = row["category"]
    text = f"{prompt}\n{response}"

    if not prompt:
        return "empty_prompt"
    if not response:
        return "empty_response"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 900:
        return "response_too_long"
    if CHATGPT_OPENAI_RE.search(text):
        return "chatgpt_openai_contamination"
    if QWEN_IDENTITY_RE.search(text):
        return "qwen_identity_contamination"
    if SECRET_RE.search(text):
        return "secret_or_token"
    if PRIVATE_DATA_RE.search(text):
        return "private_data"
    if len(URL_RE.findall(text)) >= 1:
        return "url_content"
    if UNSAFE_CYBER_RE.search(text):
        return "unsafe_cyber"
    if ILLEGAL_RE.search(text):
        return "illegal_instruction"
    if MEDICAL_THERAPY_RE.search(text) or SELF_HARM_RE.search(text):
        return "medical_or_therapy_claim"
    if category != "identity" and DARKMIND_IDENTITY_RE.search(response):
        return "darkmind_identity_leak"
    if category != "identity" and response.casefold().startswith("ben darkmind"):
        return "darkmind_identity_leak"
    return None


def current_counts(rows: list[dict[str, str]]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    return (
        Counter(row["category"] for row in rows),
        Counter(row["language"] for row in rows),
        Counter(row["difficulty"] for row in rows),
    )


def choose_next_batch(
    rows: list[dict[str, str]],
    category_targets: dict[str, int],
    language_targets: dict[str, int],
    difficulty_targets: dict[str, int],
    max_batch_size: int,
) -> tuple[str, str, str, int] | None:
    category_counts, language_counts, difficulty_counts = current_counts(rows)
    category_deficits = {key: target - category_counts[key] for key, target in category_targets.items() if category_counts[key] < target}
    language_deficits = {key: target - language_counts[key] for key, target in language_targets.items() if language_counts[key] < target}
    difficulty_deficits = {key: target - difficulty_counts[key] for key, target in difficulty_targets.items() if difficulty_counts[key] < target}
    if not category_deficits or not language_deficits or not difficulty_deficits:
        return None

    category = max(category_deficits, key=category_deficits.get)
    language = max(language_deficits, key=language_deficits.get)
    difficulty = max(difficulty_deficits, key=difficulty_deficits.get)
    count = min(max_batch_size, category_deficits[category], language_deficits[language], difficulty_deficits[difficulty])
    count = max(1, min(10, count))
    return category, language, difficulty, count


def can_accept(
    row: dict[str, str],
    rows: list[dict[str, str]],
    category_targets: dict[str, int],
    language_targets: dict[str, int],
    difficulty_targets: dict[str, int],
) -> bool:
    category_counts, language_counts, difficulty_counts = current_counts(rows)
    return (
        category_counts[row["category"]] < category_targets.get(row["category"], 0)
        and language_counts[row["language"]] < language_targets.get(row["language"], 0)
        and difficulty_counts[row["difficulty"]] < difficulty_targets.get(row["difficulty"], 0)
    )


def write_meta(
    meta_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    rejected: Counter[str],
    errors: list[str],
    targets: dict[str, dict[str, int]],
    config: dict[str, Any],
) -> None:
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "teacher_backend": config.get("teacher_backend"),
        "teacher_base_url": config.get("teacher_base_url"),
        "teacher_model": config.get("teacher_model"),
        "target_counts": targets,
        "selected_rows": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "rejected_rows_per_reason": dict(sorted(rejected.items())),
        "errors": errors[-20:],
        "completed": all(
            Counter(row[key] for row in rows)[name] >= target
            for key, target_map in [
                ("category", targets["categories"]),
                ("language", targets["languages"]),
                ("difficulty", targets["difficulties"]),
            ]
            for name, target in target_map.items()
        ),
        "first_20_samples": rows[:20],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen teacher distillation data for DarkMind.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total example limit for smoke runs.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--meta_out", type=str, default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("--batch_size must be between 1 and 10")

    random.seed(args.seed)
    config = load_config()
    teacher_system_prompt = load_teacher_prompt()
    output_path = resolve_path(args.out or config["output_path"])
    meta_path = resolve_path(args.meta_out or config["meta_path"])
    category_targets = scale_targets(CATEGORY_TARGETS, args.limit)
    language_targets = scale_targets(LANGUAGE_TARGETS, args.limit)
    difficulty_targets = scale_targets(DIFFICULTY_TARGETS, args.limit)
    targets = {
        "categories": category_targets,
        "languages": language_targets,
        "difficulties": difficulty_targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(output_path)
    seen = {normalized_key(row) for row in rows}
    rejected: Counter[str] = Counter()
    errors: list[str] = []

    print("=" * 70)
    print("DarkMind Qwen distillation generator")
    print("=" * 70)
    print(f"Config: {config['_loaded_from']}")
    print(f"Teacher: {config.get('teacher_base_url')} model={config.get('teacher_model')}")
    print(f"Output: {output_path}")
    print(f"Existing rows: {len(rows)}")
    print(f"Targets: {sum(category_targets.values())} rows")
    print("=" * 70)

    batches = 0
    while batches < args.max_batches:
        next_batch = choose_next_batch(rows, category_targets, language_targets, difficulty_targets, args.batch_size)
        if next_batch is None:
            break
        category, language, difficulty, count = next_batch
        batches += 1

        for attempt in range(2):
            prompt = build_teacher_prompt(category, language, difficulty, count, strict=attempt > 0)
            try:
                raw_text = call_teacher(config, teacher_system_prompt, prompt)
                parsed = parse_json_array(raw_text)
                break
            except Exception as exc:  # keep generation moving; metadata records details
                if attempt == 1:
                    message = f"batch={batches} category={category} language={language} difficulty={difficulty}: {exc}"
                    errors.append(message)
                    print(f"SKIP {message}")
                    parsed = []
                continue

        accepted_this_batch = 0
        for raw in parsed:
            row = normalize_candidate(raw, category, language, difficulty)
            if row is None:
                rejected["invalid_item"] += 1
                continue
            key = normalized_key(row)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            reason = rejection_reason(row)
            if reason:
                rejected[reason] += 1
                continue
            if not can_accept(row, rows, category_targets, language_targets, difficulty_targets):
                rejected["target_already_full"] += 1
                continue
            seen.add(key)
            rows.append(row)
            write_jsonl_row(output_path, row)
            accepted_this_batch += 1

        if batches % 5 == 0 or accepted_this_batch:
            category_counts, language_counts, difficulty_counts = current_counts(rows)
            print(
                f"batch={batches} accepted={accepted_this_batch} total={len(rows)} "
                f"category={dict(category_counts)} language={dict(language_counts)} difficulty={dict(difficulty_counts)}"
            )

    write_meta(meta_path, output_path, rows, rejected, errors, targets, config)
    print("=" * 70)
    print(f"Rows: {len(rows)}")
    print(f"Metadata: {meta_path}")
    print(f"Rejected: {dict(rejected)}")
    print(f"Errors: {len(errors)}")
    print("=" * 70)

    if choose_next_batch(rows, category_targets, language_targets, difficulty_targets, args.batch_size) is not None:
        sys.exit(2)


if __name__ == "__main__":
    main()
