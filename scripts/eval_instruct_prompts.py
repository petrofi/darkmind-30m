from pathlib import Path
import argparse
import json
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR))

from generate_from_checkpoint import (  # noqa: E402
    DEFAULT_CONFIG,
    TOKENIZER_DIR,
    extract_answer,
    generate_with_controls,
    load_checkpoint_model,
    load_json,
    load_tokenizer,
    set_seed,
)
from model.gpt import count_parameters  # noqa: E402


DEFAULT_EVAL = ROOT_DIR / "data" / "eval" / "darkmind_eval_prompts.jsonl"
DEFAULT_OUT = ROOT_DIR / "reports" / "instruct_eval_outputs.md"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_eval_rows(path: Path) -> list[dict[str, str]]:
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
            expected_behavior = row.get("expected_behavior", "")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt at {path}:{line_number}")

            if not isinstance(expected_behavior, str) or not expected_behavior.strip():
                raise ValueError(f"Missing expected_behavior at {path}:{line_number}")

            rows.append({
                "prompt": prompt.strip(),
                "expected_behavior": expected_behavior.strip(),
            })

    if not rows:
        raise ValueError(f"No eval prompts found in {path}")

    return rows


def write_markdown_report(
    output_path: Path,
    checkpoint_path: Path,
    eval_path: Path,
    rows: list[dict[str, str]],
    metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DarkMind Instruct Eval Outputs",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Eval set: `{eval_path}`",
        f"- Device: `{metadata['device']}`",
        f"- Parameters: `{metadata['parameter_count']:,}`",
        f"- Temperature: `{metadata['temperature']}`",
        f"- Top-k: `{metadata['top_k']}`",
        f"- Top-p: `{metadata['top_p']}`",
        f"- Repetition penalty: `{metadata['repetition_penalty']}`",
        f"- No-repeat n-gram size: `{metadata['no_repeat_ngram_size']}`",
        f"- Max new tokens: `{metadata['max_new_tokens']}`",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.extend([
            f"## {index}. {row['expected_behavior']}",
            "",
            f"**Prompt:** {row['prompt']}",
            "",
            f"**Generated answer:** {row['answer']}",
            "",
            "<details>",
            "<summary>Raw output</summary>",
            "",
            "```text",
            row["raw_output"],
            "```",
            "",
            "</details>",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed instruction eval prompts against a DarkMind checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval", type=str, default=str(DEFAULT_EVAL.relative_to(ROOT_DIR)))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT.relative_to(ROOT_DIR)))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)))
    parser.add_argument("--tokenizer", type=str, default=str(TOKENIZER_DIR.relative_to(ROOT_DIR)))
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    eval_path = resolve_path(args.eval)
    output_path = resolve_path(args.out)
    config_path = resolve_path(args.config)
    tokenizer_dir = resolve_path(args.tokenizer)

    eval_rows = load_eval_rows(eval_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = len(load_json(tokenizer_dir / "vocab.json"))
    model, _ = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=device,
        config_path=config_path,
        vocab_size=vocab_size,
    )
    parameter_count = count_parameters(model)

    completed_rows = []

    for index, row in enumerate(eval_rows):
        set_seed(args.seed + index, device)
        prompt_text = f"Kullanıcı: {row['prompt']}\nAsistan:"
        raw_output = generate_with_controls(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_on_user_marker=True,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        answer = extract_answer(raw_output, prompt_text)
        completed_rows.append({
            **row,
            "raw_output": raw_output.strip(),
            "answer": answer or "(empty)",
        })

    metadata = {
        "device": device,
        "parameter_count": parameter_count,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
    }
    write_markdown_report(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        eval_path=eval_path,
        rows=completed_rows,
        metadata=metadata,
    )

    print("=" * 70)
    print("DarkMind instruct eval completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval prompts: {len(completed_rows)}")
    print(f"Report: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
