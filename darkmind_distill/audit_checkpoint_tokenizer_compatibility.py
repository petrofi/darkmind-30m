from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "darkmind-30m-10k-step15000.pt"
DEFAULT_STUDENT = ROOT_DIR / "models" / "darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt"
DEFAULT_DATA = ROOT_DIR / "darkmind_distill" / "data" / "darkmind_qwen_distill_pilot500_tr_en_v2.jsonl"
DEFAULT_REPORT = ROOT_DIR / "darkmind_distill" / "reports" / "checkpoint_tokenizer_compatibility.md"

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|end|>"]
ROUND_TRIP_TEXTS = [
    "Merhaba, sen kimsin?",
    "Python kullanarak küçük bir REST servisini nasıl başlatırım?",
    "Docker konteynerim hemen kapanıyor.",
    "Validation loss neden yükselir?",
    "Hello, how do I create a REST API?",
    "A short Python function returns a list.",
]

SCRIPT_RANGES = {
    "turkish_latin": re.compile(r"[çğıöşüÇĞİÖŞÜ]"),
    "english_ascii": re.compile(r"^[\x20-\x7e]+$"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "greek": re.compile(r"[\u0370-\u03ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "japanese_cjk": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "replacement_or_malformed": re.compile(r"�|\\ufffd|Ã|Ä|Å|Â|ð|ğŸ"),
}
SUSPICIOUS_RE = re.compile(r"�|\\ufffd|Ã|Ä|Å|Â|ð|ğŸ|[\u0590-\u05ff\u0370-\u03ff\u0600-\u06ff\u0400-\u04ff\u0900-\u097f]")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = metadata.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint state dict: {path}")
    return state_dict, metadata


def tensor_shape(state: dict[str, torch.Tensor], key: str) -> list[int] | None:
    tensor = state.get(key)
    if tensor is None:
        return None
    return list(tensor.shape)


def state_vocab_size(state: dict[str, torch.Tensor]) -> int | None:
    for key in ["token_embedding.weight", "lm_head.weight"]:
        tensor = state.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    return None


def weight_tying_status(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    token = state.get("token_embedding.weight")
    head = state.get("lm_head.weight")
    if token is None or head is None:
        return {"present": False, "allclose": False, "max_abs_diff": None}
    if token.shape != head.shape:
        return {"present": True, "allclose": False, "max_abs_diff": "shape_mismatch"}
    diff = (token.float() - head.float()).abs()
    return {
        "present": True,
        "allclose": bool(torch.allclose(token, head)),
        "max_abs_diff": float(diff.max().item()),
    }


def token_strings(tokenizer: ByteLevelBPETokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.id_to_token(token_id) or "" for token_id in ids]


def suspicious_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if SUSPICIOUS_RE.search(token)]


def vocabulary_script_counts(vocab: dict[str, int]) -> tuple[Counter[str], list[tuple[int, str]]]:
    counts: Counter[str] = Counter()
    suspicious: list[tuple[int, str]] = []
    for token, token_id in vocab.items():
        for label, pattern in SCRIPT_RANGES.items():
            if pattern.search(token):
                counts[label] += 1
        if SUSPICIOUS_RE.search(token):
            suspicious.append((int(token_id), token))
    suspicious.sort(key=lambda item: item[0])
    return counts, suspicious[:100]


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"git command failed: {exc}"


def round_trip_rows(tokenizer: ByteLevelBPETokenizer, texts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        tokens = token_strings(tokenizer, encoded.ids)
        rows.append(
            {
                "original": text,
                "ids": encoded.ids,
                "tokens": tokens,
                "decoded": decoded,
                "exact_match": decoded == text,
                "tokens_per_character": round(len(encoded.ids) / max(len(text), 1), 4),
                "suspicious_script_tokens": suspicious_tokens(tokens),
            }
        )
    return rows


def deterministic_training_texts(rows: list[dict[str, Any]], count: int = 20) -> list[str]:
    if not rows:
        return []
    selected: list[str] = []
    step = max(1, len(rows) // count)
    for row in rows[::step][:count]:
        selected.append(str(row.get("prompt", "")))
        selected.append(str(row.get("response", "")))
    return selected[: count * 2]


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tokenizer/checkpoint compatibility for DarkMind.")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER.relative_to(ROOT_DIR)))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT.relative_to(ROOT_DIR)))
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_STUDENT.relative_to(ROOT_DIR)))
    parser.add_argument("--data", default=str(DEFAULT_DATA.relative_to(ROOT_DIR)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT_DIR)))
    args = parser.parse_args()

    tokenizer_dir = resolve_path(args.tokenizer)
    checkpoint_path = resolve_path(args.checkpoint)
    student_path = resolve_path(args.student_checkpoint)
    data_path = resolve_path(args.data)
    report_path = resolve_path(args.report)

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    vocab = load_json(vocab_path)
    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    base_state, base_meta = load_checkpoint(checkpoint_path)
    student_state, student_meta = load_checkpoint(student_path)
    base_vocab = state_vocab_size(base_state)
    student_vocab = state_vocab_size(student_state)
    vocab_counts, suspicious_vocab = vocabulary_script_counts(vocab)
    rows = load_rows(data_path)

    max_token_id = max(int(token_id) for token_id in vocab.values())
    special_ids = {token: vocab.get(token) for token in SPECIAL_TOKENS}
    token_ids_within_base = base_vocab is not None and max_token_id < base_vocab
    token_ids_within_student = student_vocab is not None and max_token_id < student_vocab
    tokenizer_files = sorted(path for path in tokenizer_dir.iterdir() if path.is_file())
    history_paths = [
        "tokenizer/darkmind-tokenizer/vocab.json",
        "tokenizer/darkmind-tokenizer/merges.txt",
        "scripts/train_tokenizer.py",
        "configs",
        "docs",
        "README.md",
        "MODEL_CARD.md",
    ]

    lines = [
        "# Checkpoint and Tokenizer Compatibility Audit",
        "",
        f"Tokenizer: `{tokenizer_dir}`",
        f"Base checkpoint: `{checkpoint_path}`",
        f"Student checkpoint: `{student_path}`",
        "",
        "## Tokenizer File Hashes",
        "",
    ]
    for file_path in tokenizer_files:
        lines.append(f"- `{file_path.name}`: `{sha256(file_path)}`")

    lines.extend(
        [
            "",
            "## Compatibility",
            "",
            f"- Tokenizer vocabulary size: `{len(vocab)}`",
            f"- Maximum tokenizer ID: `{max_token_id}`",
            f"- Base checkpoint vocabulary size: `{base_vocab}`",
            f"- Student checkpoint vocabulary size: `{student_vocab}`",
            f"- Base embedding shape: `{tensor_shape(base_state, 'token_embedding.weight')}`",
            f"- Base LM head shape: `{tensor_shape(base_state, 'lm_head.weight')}`",
            f"- Student embedding shape: `{tensor_shape(student_state, 'token_embedding.weight')}`",
            f"- Student LM head shape: `{tensor_shape(student_state, 'lm_head.weight')}`",
            f"- Base input/output weights tied/equal in state dict: `{weight_tying_status(base_state)}`",
            f"- Student input/output weights tied/equal in state dict: `{weight_tying_status(student_state)}`",
            f"- Special token IDs: `{special_ids}`",
            f"- EOS token ID: `{vocab.get('</s>')}`",
            f"- PAD token ID: `{vocab.get('<pad>')}`",
            f"- UNK token ID: `{vocab.get('<unk>')}`",
            f"- Token IDs within base checkpoint bounds: `{token_ids_within_base}`",
            f"- Token IDs within student checkpoint bounds: `{token_ids_within_student}`",
            f"- Base checkpoint metadata keys: `{sorted(base_meta.keys()) if isinstance(base_meta, dict) else []}`",
            f"- Student checkpoint metadata keys: `{sorted(student_meta.keys()) if isinstance(student_meta, dict) else []}`",
            "",
            "## Vocabulary Script Counts",
            "",
        ]
    )
    for label in SCRIPT_RANGES:
        lines.append(f"- {label}: `{vocab_counts.get(label, 0)}`")

    lines.extend(["", "## Suspicious Vocabulary Entries", ""])
    if suspicious_vocab:
        for token_id, token in suspicious_vocab:
            lines.append(f"- `{token_id}`: `{token}`")
    else:
        lines.append("- None detected by script heuristics.")

    lines.extend(["", "## Exact Round-Trip Tests", ""])
    for item in round_trip_rows(tokenizer, ROUND_TRIP_TEXTS):
        lines.extend(
            [
                f"### `{item['original']}`",
                f"- Token IDs: `{item['ids']}`",
                f"- Token strings: `{item['tokens']}`",
                f"- Decoded: `{item['decoded']}`",
                f"- Exact round-trip match: `{item['exact_match']}`",
                f"- Tokens per character: `{item['tokens_per_character']}`",
                f"- Suspicious-script tokens: `{item['suspicious_script_tokens']}`",
                "",
            ]
        )

    lines.extend(["", "## Deterministic Training Round Trips", ""])
    for index, item in enumerate(round_trip_rows(tokenizer, deterministic_training_texts(rows)), start=1):
        lines.extend(
            [
                f"### Training text {index}",
                f"- Original: `{item['original'][:260]}`",
                f"- Token IDs: `{item['ids'][:120]}{'...' if len(item['ids']) > 120 else ''}`",
                f"- Token strings: `{item['tokens'][:120]}{'...' if len(item['tokens']) > 120 else ''}`",
                f"- Decoded: `{item['decoded'][:260]}`",
                f"- Exact round-trip match: `{item['exact_match']}`",
                f"- Tokens per character: `{item['tokens_per_character']}`",
                f"- Suspicious-script tokens: `{item['suspicious_script_tokens'][:30]}`",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Repository Evidence",
            "",
            "### Tokenizer history",
            "```text",
            run_git(["log", "--oneline", "--", "tokenizer/darkmind-tokenizer", "scripts/train_tokenizer.py"]),
            "```",
            "",
            "### Tokenizer mentions in history/config/docs",
            "```text",
            run_git(["grep", "-n", "-i", "tokenizer", "--", *history_paths]),
            "```",
            "",
            "### Recent commits touching tokenizer files",
            "```text",
            run_git(["log", "--stat", "--oneline", "--", "tokenizer/darkmind-tokenizer"]),
            "```",
        ]
    )

    write_report(report_path, lines)
    print("\n".join(lines[:80]))
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
