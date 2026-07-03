from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT_DIR / "scripts" / "generate_from_checkpoint.py"
DEFAULT_BASE = ROOT_DIR / "models" / "darkmind-30m-10k-step15000.pt"
DEFAULT_STUDENT = ROOT_DIR / "models" / "darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "darkmind_30m_1000step.json"
DEFAULT_TOKENIZER = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_REPORT = ROOT_DIR / "darkmind_distill" / "reports" / "base_vs_student_greedy_diagnostic.md"
DEFAULT_JSON = ROOT_DIR / "darkmind_distill" / "reports" / "base_vs_student_greedy_diagnostic.json"

PROMPTS = [
    {"id": "tr_hello", "language": "tr", "category": "general", "prompt": "Merhaba."},
    {"id": "tr_turkey", "language": "tr", "category": "general", "prompt": "Türkiye hakkında iki cümle yaz."},
    {"id": "tr_python", "language": "tr", "category": "programming", "prompt": "Python nedir?"},
    {"id": "tr_docker", "language": "tr", "category": "backend", "prompt": "Docker nedir?"},
    {"id": "tr_identity", "language": "tr", "category": "identity", "prompt": "Sen kimsin?"},
    {"id": "en_hello", "language": "en", "category": "general", "prompt": "Hello."},
    {"id": "en_python", "language": "en", "category": "programming", "prompt": "What is Python?"},
    {"id": "en_sentence", "language": "en", "category": "general", "prompt": "Write one English sentence."},
]

FOREIGN_SCRIPT_RE = re.compile(r"[\u0590-\u05ff\u0370-\u03ff\u0600-\u06ff\u0400-\u04ff\u0530-\u058f\u0900-\u097f\u3040-\u30ff\u3400-\u9fff]")
REPLACEMENT_RE = re.compile(r"�|\\ufffd")
LATIN_WORD_RE = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]{2,}")
TURKISH_HINT_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]|\b(bir|ve|değil|nedir|hakkında|merhaba|python|docker)\b", re.IGNORECASE)
ENGLISH_HINT_RE = re.compile(r"\b(the|is|a|an|hello|python|sentence|write|what|one|english)\b", re.IGNORECASE)
DARKMIND_RE = re.compile(r"\bdarkmind\b", re.IGNORECASE)
IDENTITY_DENIAL_RE = re.compile(r"\b(chatgpt|openai|qwen)\b.{0,40}\b(not|değil|degil|değilim|degilim)\b|\b(not|değil|degil|değilim|degilim)\b.{0,40}\b(chatgpt|openai|qwen)\b", re.IGNORECASE)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("darkmind_generate_from_checkpoint", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator module: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chat_prompt(prompt: str) -> str:
    # Use the same formatter family as the SFT code path.
    return f"Kullanıcı: {prompt.strip()}\nAsistan:"


def greedy_decode(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    eos_ids: set[int],
) -> tuple[str, list[int]]:
    encoded = tokenizer.encode(prompt_text)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id in eos_ids:
                break
    return tokenizer.decode(idx[0].tolist()), generated


def extract_answer(generator: Any, raw_output: str, prompt_text: str) -> str:
    return generator.extract_answer(raw_output, prompt_text)


def repeated_or_degenerate(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 6:
        return False
    grams = Counter(tuple(words[index : index + 3]) for index in range(len(words) - 2))
    return any(count >= 3 for count in grams.values())


def diagnostic_flags(row: dict[str, str], answer: str) -> dict[str, bool]:
    stripped = answer.strip()
    foreign_count = len(FOREIGN_SCRIPT_RE.findall(stripped))
    replacement_count = len(REPLACEMENT_RE.findall(stripped))
    latin_word_count = len(LATIN_WORD_RE.findall(stripped))
    nonspace_count = len(re.findall(r"\S", stripped))
    foreign_ratio = foreign_count / max(nonspace_count, 1)
    malformed = replacement_count > 0 or foreign_ratio > 0.02 or (foreign_count >= 3)
    gibberish = malformed or repeated_or_degenerate(stripped) or (len(stripped) >= 20 and latin_word_count == 0)
    language = row["language"]
    language_match = (
        bool(TURKISH_HINT_RE.search(stripped)) if language == "tr" else bool(ENGLISH_HINT_RE.search(stripped))
    ) and not malformed
    identity_prompt = row["category"] == "identity"
    identity_correct = identity_prompt and bool(DARKMIND_RE.search(stripped))
    if identity_prompt and any(term in row["prompt"].casefold() for term in ["chatgpt", "qwen", "openai"]):
        identity_correct = identity_correct and bool(IDENTITY_DENIAL_RE.search(stripped))
    identity_leakage = not identity_prompt and bool(DARKMIND_RE.search(stripped))
    prompt_terms = {
        token
        for token in re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ0-9]{4,}", row["prompt"].casefold())
        if token not in {"nedir", "what", "write", "hakkında", "about"}
    }
    output_terms = set(re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ0-9]{4,}", stripped.casefold()))
    prompt_relevant = bool(prompt_terms & output_terms) and not gibberish and language_match
    return {
        "empty": len(stripped) < 3,
        "mixed_or_foreign_script": malformed,
        "gibberish": gibberish,
        "language_match": language_match,
        "prompt_relevant": prompt_relevant,
        "identity_correct": identity_correct,
        "identity_leakage": identity_leakage,
    }


def run_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    generator: Any,
    tokenizer: Any,
    config: Path,
    device: str,
    vocab_size: int,
    eos_ids: set[int],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model, checkpoint_meta = generator.load_checkpoint_model(
        checkpoint_path=checkpoint,
        device=device,
        config_path=config,
        vocab_size=vocab_size,
    )
    rows: list[dict[str, Any]] = []
    for prompt_row in PROMPTS:
        model_input = chat_prompt(prompt_row["prompt"])
        raw_output, generated_ids = greedy_decode(
            model=model,
            tokenizer=tokenizer,
            prompt_text=model_input,
            device=device,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
        )
        answer = extract_answer(generator, raw_output, model_input)
        flags = diagnostic_flags(prompt_row, answer)
        rows.append(
            {
                **prompt_row,
                "checkpoint_label": label,
                "checkpoint": str(checkpoint),
                "run_name": checkpoint_meta.get("run_name", "unknown") if isinstance(checkpoint_meta, dict) else "unknown",
                "model_input": model_input,
                "generated_token_ids": generated_ids,
                "raw_output": raw_output,
                "answer": answer,
                "flags": flags,
            }
        )
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def write_report(path: Path, rows: list[dict[str, Any]], device: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["checkpoint_label"], []).append(row)
    lines = [
        "# Base vs Student Greedy Diagnostic",
        "",
        "Decoding is deterministic greedy argmax. No sampling, no top-p, no multinomial sampling.",
        f"Device: `{device}`",
        "Settings: `max_new_tokens=80`, `argmax=True`",
        "",
        "## Summary",
        "",
    ]
    for label, items in grouped.items():
        flag_counts = Counter(flag for item in items for flag, value in item["flags"].items() if value is True)
        failures = sum(
            1
            for item in items
            if item["flags"]["empty"]
            or item["flags"]["gibberish"]
            or item["flags"]["mixed_or_foreign_script"]
            or not item["flags"]["language_match"]
            or item["flags"]["identity_leakage"]
        )
        lines.append(f"- {label}: failures `{failures}/{len(items)}`, true-flag counts `{dict(flag_counts)}`")
    lines.append("")

    for prompt in PROMPTS:
        lines.extend([f"## `{prompt['id']}`", "", f"Prompt: `{prompt['prompt']}`", ""])
        for label in grouped:
            item = next(row for row in grouped[label] if row["id"] == prompt["id"])
            lines.extend(
                [
                    f"### {label}",
                    f"- Model input: `{item['model_input']}`",
                    f"- Generated token IDs: `{item['generated_token_ids']}`",
                    f"- Flags: `{item['flags']}`",
                    "",
                    "```text",
                    item["answer"],
                    "```",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic greedy base/student diagnostic eval.")
    parser.add_argument("--base", default=str(DEFAULT_BASE.relative_to(ROOT_DIR)))
    parser.add_argument("--student", default=str(DEFAULT_STUDENT.relative_to(ROOT_DIR)))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER.relative_to(ROOT_DIR)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT_DIR)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON.relative_to(ROOT_DIR)))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    generator = load_generator_module()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    tokenizer_dir = resolve_path(args.tokenizer)
    tokenizer = generator.load_tokenizer(tokenizer_dir)
    vocab = generator.load_json(tokenizer_dir / "vocab.json")
    eos_ids = {token_id for token in ["</s>", "<|end|>"] if (token_id := vocab.get(token)) is not None}
    rows = []
    rows.extend(
        run_checkpoint(
            label="base",
            checkpoint=resolve_path(args.base),
            generator=generator,
            tokenizer=tokenizer,
            config=resolve_path(args.config),
            device=device,
            vocab_size=len(vocab),
            eos_ids=eos_ids,
            max_new_tokens=args.max_new_tokens,
        )
    )
    rows.extend(
        run_checkpoint(
            label="student",
            checkpoint=resolve_path(args.student),
            generator=generator,
            tokenizer=tokenizer,
            config=resolve_path(args.config),
            device=device,
            vocab_size=len(vocab),
            eos_ids=eos_ids,
            max_new_tokens=args.max_new_tokens,
        )
    )
    write_json(resolve_path(args.json_out), rows)
    write_report(resolve_path(args.report), rows, device)
    print(f"Report written: {resolve_path(args.report)}")
    print(f"JSON written: {resolve_path(args.json_out)}")


if __name__ == "__main__":
    main()
