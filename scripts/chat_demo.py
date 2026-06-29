from pathlib import Path
import argparse
from datetime import datetime
import json
import re
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "scripts"))

from generate_from_checkpoint import (  # noqa: E402
    SAFE_FALLBACK,
    clean_generated_text,
    extract_answer,
    generate_with_controls,
    load_checkpoint_model,
    load_tokenizer,
    resolve_path,
    set_seed,
)
from model.gpt import count_parameters  # noqa: E402


EXIT_COMMANDS = {"q", "quit", "exit", "çık"}


def normalize_user_input(user_input: str) -> str:
    return re.sub(r"[ \t]+", " ", user_input.strip())


def format_prompt(question: str) -> str:
    normalized_question = normalize_user_input(question)
    return f"Kullanıcı: {normalized_question}\nAsistan:"


def append_chat_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_answer(
    model,
    tokenizer,
    question: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    stop_on_user_marker: bool,
    min_answer_chars: int,
    debug_prompt: bool,
) -> tuple[str, str]:
    prompt = format_prompt(question)

    if debug_prompt:
        print("-" * 70)
        print("DEBUG PROMPT:")
        print(prompt)
        print("-" * 70)

    output_text = generate_with_controls(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_on_user_marker=stop_on_user_marker,
    )
    answer = extract_answer(output_text, prompt)
    answer = clean_generated_text(answer)

    if len(answer) < min_answer_chars:
        answer = SAFE_FALLBACK

    return answer or SAFE_FALLBACK, prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an interactive DarkMind terminal chat demo."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--stop_on_user_marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation when a new Kullanıcı:/Sen: turn appears.",
    )
    parser.add_argument("--min_answer_chars", type=int, default=0)
    parser.add_argument("--save_chat_log", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        help="Print the exact prompt sent to the model.",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    if args.min_answer_chars < 0:
        raise ValueError("--min_answer_chars must be non-negative")

    checkpoint_path = resolve_path(args.checkpoint)
    chat_log_path = resolve_path(args.save_chat_log) if args.save_chat_log else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device)

    tokenizer = load_tokenizer()
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)

    print("=" * 70)
    print("DarkMind chat demo")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {checkpoint.get('run_name', 'unknown')}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Vocab size: {model.config.vocab_size}")
    print("=" * 70)
    print("Çıkmak için q, quit, exit veya çık yaz.")

    while True:
        try:
            user_input = normalize_user_input(input("Sen: "))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            break

        answer, prompt = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=user_input,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_on_user_marker=args.stop_on_user_marker,
            min_answer_chars=args.min_answer_chars,
            debug_prompt=args.debug_prompt,
        )

        print(f"DarkMind: {answer}")

        if chat_log_path:
            append_chat_log(
                chat_log_path,
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "prompt": prompt,
                    "user": user_input,
                    "answer": answer,
                    "checkpoint": str(checkpoint_path),
                    "settings": {
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "repetition_penalty": args.repetition_penalty,
                        "stop_on_user_marker": args.stop_on_user_marker,
                        "min_answer_chars": args.min_answer_chars,
                    },
                },
            )


if __name__ == "__main__":
    main()
