from pathlib import Path
import argparse
from datetime import datetime
import json
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


PRESETS_PATH = ROOT_DIR / "configs" / "inference_presets.json"
DEFAULT_PROMPTS_PATH = ROOT_DIR / "data" / "evals" / "inference_smoke_prompts.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "eval"
DEFAULT_PROMPTS = [
    "Merhaba",
    "sen kimsin",
    "DarkMind ChatGPT gibi mi?",
    "Tokenizer nedir?",
    "CUDA neden önemli?",
    "Python'da listeye eleman nasıl eklenir?",
    "Python'da try except ne işe yarar?",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_path() -> Path:
    return DEFAULT_OUTPUT_DIR / f"inference_suite_{timestamp_for_filename()}.jsonl"


def ensure_prompts_file(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(DEFAULT_PROMPTS) + "\n", encoding="utf-8")


def load_prompts(path: Path) -> list[str]:
    ensure_prompts_file(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def format_prompt(prompt: str) -> str:
    return f"Kullanıcı: {prompt.strip()}\nAsistan:"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small inference smoke suite against a checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preset", type=str, default="deterministic")
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=str(DEFAULT_PROMPTS_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    presets = load_json(PRESETS_PATH)

    if args.preset not in presets:
        raise ValueError(
            f"Unknown preset {args.preset!r}. Available: {', '.join(sorted(presets))}"
        )

    checkpoint_path = resolve_path(args.checkpoint)
    prompts_path = resolve_path(args.prompts_path)
    output_path = resolve_path(args.output_path) if args.output_path else default_output_path()
    settings = presets[args.preset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device)

    tokenizer = load_tokenizer()
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    prompts = load_prompts(prompts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, prompt in enumerate(prompts, start=1):
            set_seed(args.seed + index, device)
            prompt_text = format_prompt(prompt)
            output_text = generate_with_controls(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                device=device,
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                stop_on_user_marker=True,
            )
            answer = clean_generated_text(extract_answer(output_text, prompt_text))

            if not answer:
                answer = SAFE_FALLBACK

            file.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "answer": answer,
                        "checkpoint": str(checkpoint_path),
                        "run_name": checkpoint.get("run_name", "unknown"),
                        "preset": args.preset,
                        "settings": settings,
                        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 70)
    print(f"Inference suite output: {output_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Preset: {args.preset}")
    print("=" * 70)


if __name__ == "__main__":
    main()
