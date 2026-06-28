from pathlib import Path
import argparse
import re
import sys

import torch
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from generate_from_checkpoint import clean_special_tokens, fix_dialogue_spacing
from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


EXIT_COMMANDS = {"q", "quit", "exit", "çık"}


def normalize_user_input(user_input: str) -> str:
    return re.sub(r"[ \t]+", " ", user_input.strip())


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_tokenizer() -> ByteLevelBPETokenizer:
    tokenizer_dir = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def load_model(checkpoint_path: Path, device: str) -> GPTLanguageModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 70)
    print("DarkMind chat demo")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {checkpoint.get('run_name', 'unknown')}")
    print(f"Model parameters: {count_parameters(model):,}")
    print("=" * 70)

    return model


def extract_first_answer(output_text: str, prompt_text: str) -> str:
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

    answer = fix_dialogue_spacing(answer).strip()

    if answer.startswith("Asistan:"):
        answer = answer[len("Asistan:"):].strip()

    return answer


def generate_answer(
    model: GPTLanguageModel,
    tokenizer: ByteLevelBPETokenizer,
    question: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    debug_prompt: bool,
) -> str:
    normalized_question = normalize_user_input(question)
    prompt = f"Kullanıcı: {normalized_question}\nAsistan:"

    if debug_prompt:
        print("-" * 70)
        print("DEBUG PROMPT:")
        print(prompt)
        print("-" * 70)

    encoded = tokenizer.encode(prompt)

    idx = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    if top_k <= 0:
        generation_top_k = None
    else:
        generation_top_k = min(top_k, model.config.vocab_size)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=generation_top_k,
        )

    output_text = tokenizer.decode(generated[0].tolist())
    answer = extract_first_answer(output_text, prompt)

    return answer or "(boş cevap)"


def main():
    parser = argparse.ArgumentParser(
        description="Run an interactive DarkMind terminal chat demo."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling value. Use 0 to disable top-k.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Maximum number of new tokens per answer.",
    )
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        help="Print the exact prompt sent to the model.",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    checkpoint_path = resolve_path(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()
    model = load_model(checkpoint_path, device)

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

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=user_input,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            debug_prompt=args.debug_prompt,
        )

        print(f"DarkMind: {answer}")


if __name__ == "__main__":
    main()
