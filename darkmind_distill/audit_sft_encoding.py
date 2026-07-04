from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from scripts.train_instruct_jsonl import format_prompt


DEFAULT_DATA = ROOT_DIR / "darkmind_distill" / "data" / "darkmind_qwen_distill_pilot500_tr_en_v2.jsonl"
DEFAULT_TOKENIZER = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_REPORT = ROOT_DIR / "darkmind_distill" / "reports" / "sft_encoding_label_mask_audit.md"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line"] = line_number
            rows.append(row)
    return rows


def load_tokenizer(path: Path) -> tuple[ByteLevelBPETokenizer, dict[str, int]]:
    vocab_path = path / "vocab.json"
    merges_path = path / "merges.txt"
    with vocab_path.open("r", encoding="utf-8") as file:
        vocab = json.load(file)
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path)), vocab


def deterministic_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    step = max(1, len(rows) // count)
    return rows[::step][:count]


def build_training_encoding(
    row: dict[str, Any],
    tokenizer: ByteLevelBPETokenizer,
    block_size: int,
    eos_id: int,
) -> dict[str, Any]:
    record = {"prompt": str(row.get("prompt", "")).strip(), "response": str(row.get("response", "")).strip()}
    prompt_text = format_prompt(record)
    full_text = f"{prompt_text} {record['response']}\n"
    prompt_ids = tokenizer.encode(prompt_text).ids
    original_full_ids = tokenizer.encode(full_text).ids + [eos_id]
    truncated_ids = original_full_ids[: block_size + 1]
    x_ids = truncated_ids[:-1]
    y_ids = truncated_ids[1:]
    labels: list[int] = []

    for label_index, token_id in enumerate(y_ids):
        original_token_index = label_index + 1
        labels.append(-100 if original_token_index < len(prompt_ids) else token_id)

    pad_len = block_size - len(x_ids)
    if pad_len > 0:
        x_ids.extend([0] * pad_len)
        labels.extend([-100] * pad_len)

    supervised_ids = [token_id for token_id in labels if token_id != -100]
    response_ids = tokenizer.encode(f" {record['response']}\n").ids + [eos_id]
    response_retained = len([token_id for token_id in supervised_ids if token_id != 0])
    intended_response_tokens = len(response_ids)
    prompt_supervised = any(
        label != -100 and index + 1 < len(prompt_ids)
        for index, label in enumerate(labels[: len(y_ids)])
    )
    eos_positions = [index for index, token_id in enumerate(labels) if token_id == eos_id]
    decoded_supervised = tokenizer.decode([token_id for token_id in supervised_ids if token_id >= 0])
    decoded_input = tokenizer.decode([token_id for token_id in x_ids if token_id != 0])
    expected_prefix = tokenizer.decode(response_ids[: max(0, len(supervised_ids))])
    response_alignment = decoded_supervised.startswith(expected_prefix[: min(len(expected_prefix), 40)])

    return {
        "line": row["_line"],
        "raw_prompt": record["prompt"],
        "raw_response": record["response"],
        "formatted_training_text": full_text + f"<EOS:{eos_id}>",
        "prompt_ids": prompt_ids,
        "original_full_ids": original_full_ids,
        "input_token_ids": x_ids,
        "decoded_complete_input": decoded_input,
        "label_token_ids": labels,
        "decoded_supervised_label_region": decoded_supervised,
        "masked_prompt_tokens": sum(1 for label in labels[: max(0, len(prompt_ids) - 1)] if label == -100),
        "supervised_response_tokens": len(supervised_ids),
        "eos_token_position": eos_positions,
        "truncation_status": len(original_full_ids) > block_size + 1,
        "response_retained_percent": round(100 * response_retained / max(intended_response_tokens, 1), 2),
        "prompt_tokens_accidentally_supervised": prompt_supervised,
        "response_labels_align": response_alignment,
        "labels_inputs_equal_lengths": len(labels) == len(x_ids),
        "eos_supervised": bool(eos_positions),
        "response_not_empty_after_truncation": len(supervised_ids) > 0,
    }


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SFT tokenization, formatting, and label masking.")
    parser.add_argument("--data", default=str(DEFAULT_DATA.relative_to(ROOT_DIR)))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER.relative_to(ROOT_DIR)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT_DIR)))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    tokenizer_path = resolve_path(args.tokenizer)
    report_path = resolve_path(args.report)
    rows = load_rows(data_path)
    tokenizer, vocab = load_tokenizer(tokenizer_path)
    eos_id = int(vocab.get("</s>", 2))
    pad_id = int(vocab.get("<pad>", 0))
    audits = [
        build_training_encoding(row, tokenizer, args.block_size, eos_id)
        for row in deterministic_sample(rows, args.samples)
    ]

    failures: list[str] = []
    for audit in audits:
        line = audit["line"]
        if audit["prompt_tokens_accidentally_supervised"]:
            failures.append(f"line {line}: prompt tokens accidentally supervised")
        if not audit["response_not_empty_after_truncation"]:
            failures.append(f"line {line}: empty supervised response after truncation")
        if not audit["labels_inputs_equal_lengths"]:
            failures.append(f"line {line}: input/label length mismatch")
        if not audit["eos_supervised"]:
            failures.append(f"line {line}: EOS is not supervised")
        if audit["response_retained_percent"] <= 0:
            failures.append(f"line {line}: no response tokens retained")
        if not audit["response_labels_align"]:
            failures.append(f"line {line}: response labels do not align with intended response prefix")

    lines = [
        "# SFT Encoding and Label-Mask Audit",
        "",
        f"Data: `{data_path}`",
        f"Tokenizer: `{tokenizer_path}`",
        f"Block size: `{args.block_size}`",
        f"Samples: `{len(audits)}`",
        f"EOS token ID: `{eos_id}`",
        f"PAD token ID: `{pad_id}`",
        f"Result: `{'FAIL' if failures else 'PASS'}`",
        "",
        "## Strict Checks",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.extend(
            [
                "- Prompt tokens are masked in sampled examples.",
                "- Assistant response labels are supervised in sampled examples.",
                "- EOS is supervised in sampled examples.",
                "- Labels and inputs have equal lengths.",
                "- No off-by-one alignment failure detected in sampled examples.",
            ]
        )

    lines.extend(["", "## Samples", ""])
    for index, audit in enumerate(audits, start=1):
        lines.extend(
            [
                f"### Sample {index} - line {audit['line']}",
                f"- Raw prompt: `{audit['raw_prompt']}`",
                f"- Raw response: `{audit['raw_response']}`",
                f"- Exact formatted training text: `{audit['formatted_training_text']}`",
                f"- Input token IDs: `{audit['input_token_ids']}`",
                f"- Decoded complete input: `{audit['decoded_complete_input']}`",
                f"- Label token IDs: `{audit['label_token_ids']}`",
                f"- Decoded supervised label region: `{audit['decoded_supervised_label_region']}`",
                f"- Number of prompt tokens masked with -100: `{audit['masked_prompt_tokens']}`",
                f"- Number of supervised response tokens: `{audit['supervised_response_tokens']}`",
                f"- EOS token position: `{audit['eos_token_position']}`",
                f"- Truncation status: `{audit['truncation_status']}`",
                f"- Percentage of response retained: `{audit['response_retained_percent']}`",
                f"- Prompt tokens accidentally supervised: `{audit['prompt_tokens_accidentally_supervised']}`",
                f"- Response labels align with intended response: `{audit['response_labels_align']}`",
                "",
            ]
        )

    write_report(report_path, lines)
    print("\n".join(lines[:60]))
    print(f"Report written: {report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
