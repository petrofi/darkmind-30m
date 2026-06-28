from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

import torch
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel


DEFAULT_EVAL_PATH = ROOT_DIR / "data" / "evals" / "darkmind_eval_v01.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "self_improvement" / "runs"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
SEED = 42


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_jsonl(path: Path) -> list[dict]:
    items = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

    return items


def load_tokenizer() -> ByteLevelBPETokenizer:
    vocab_path = TOKENIZER_DIR / "vocab.json"
    merges_path = TOKENIZER_DIR / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {TOKENIZER_DIR}")

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def load_model(checkpoint_path: Path, device: str) -> GPTLanguageModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def set_generation_seed(device: str) -> None:
    torch.manual_seed(SEED)

    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)


def clean_special_tokens(text: str) -> str:
    for token in ["</s>", "<s>", "<pad>", "<unk>", "<mask>"]:
        if token in text:
            text = text.split(token)[0]

    return text.strip()


def extract_answer(output_text: str, prompt_text: str) -> str:
    output_text = clean_special_tokens(output_text)

    if output_text.startswith(prompt_text):
        answer = output_text[len(prompt_text):]
    else:
        answer = output_text

    stop_markers = [
        "\n\nKullanıcı:",
        "\nKullanıcı:",
        "\n\nSoru:",
        "\nSoru:",
        "\n\nSen:",
        "\nSen:",
    ]

    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0]
            break

    answer = answer.strip()

    if answer.startswith("Asistan:"):
        answer = answer[len("Asistan:"):].strip()

    return answer


def format_prompt(prompt: str) -> str:
    return f"Kullanıcı: {prompt.strip()}\nAsistan:"


def generate_answer(
    model: GPTLanguageModel,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: str,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    prompt_text = format_prompt(prompt)
    encoded = tokenizer.encode(prompt_text)

    idx = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    generation_top_k = None

    if top_k > 0:
        generation_top_k = min(top_k, model.config.vocab_size)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=generation_top_k,
        )

    output_text = tokenizer.decode(generated[0].tolist())
    return extract_answer(output_text, prompt_text)


def score_answer(answer: str, expected_keywords: list[str]) -> tuple[list[str], list[str]]:
    answer_folded = answer.casefold()
    matched_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.casefold() in answer_folded:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    return matched_keywords, missing_keywords


def build_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"eval_run_{timestamp}.jsonl"


def run_evaluation(
    checkpoint_path: Path,
    eval_path: Path,
    output_dir: Path,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
) -> Path:
    if temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_items = load_jsonl(eval_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_generation_seed(device)

    tokenizer = load_tokenizer()
    model = load_model(checkpoint_path, device)

    results = []

    for item in eval_items:
        expected_keywords = item.get("expected_keywords", [])
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            device=device,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        matched_keywords, missing_keywords = score_answer(
            answer,
            expected_keywords,
        )

        results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "answer": answer,
                "expected_keywords": expected_keywords,
                "matched_keywords": matched_keywords,
                "missing_keywords": missing_keywords,
                "passed": len(missing_keywords) == 0,
                "category": item["category"],
            }
        )

    output_path = build_output_path(output_dir)

    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0
    failures_by_category = Counter(
        result["category"]
        for result in results
        if not result["passed"]
    )

    print("=" * 70)
    print("DarkMind Eval Run")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Eval file: {eval_path}")
    print(f"Output file: {output_path}")
    print(f"Total evals: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print("-" * 70)
    print("Failures by category:")

    if failures_by_category:
        for category, count in sorted(failures_by_category.items()):
            print(f"- {category}: {count}")
    else:
        print("No failures.")

    print("=" * 70)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DarkMind eval prompts and save JSONL results."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=str(DEFAULT_EVAL_PATH.relative_to(ROOT_DIR)),
        help="Evaluation JSONL path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
        help="Directory where eval run JSONL files are saved.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k sampling value. Use 0 to disable top-k.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens per eval answer.",
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=resolve_path(args.checkpoint),
        eval_path=resolve_path(args.eval_path),
        output_dir=resolve_path(args.output_dir),
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
