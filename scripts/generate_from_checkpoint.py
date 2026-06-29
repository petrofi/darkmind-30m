from pathlib import Path
import argparse
from datetime import datetime
import json
import random
import re
import sys

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
SPECIAL_STOP_TOKENS = ["</s>", "<s>", "<pad>", "<unk>", "<mask>", "<|end|>"]
TURN_STOP_MARKERS = [
    "\n\nKullanıcı:",
    "\nKullanıcı:",
    "\n\nSen:",
    "\nSen:",
    "\n\nSoru:",
    "\nSoru:",
    "<|end|>",
]
SAFE_FALLBACK = "Bu konuda yeterli bilgiye sahip olmayabilirim."


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def clean_special_tokens(output_text: str) -> str:
    for stop_token in SPECIAL_STOP_TOKENS:
        if stop_token in output_text:
            output_text = output_text.split(stop_token)[0]

    return output_text.strip()


def trim_at_turn_marker(text: str) -> str:
    for marker in TURN_STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
            break

    return text


def clean_generated_text(text: str) -> str:
    text = clean_special_tokens(text)
    text = trim_at_turn_marker(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(Asistan:)(\S)", r"\1 \2", text)
    text = re.sub(r"(Kullanıcı:)(\S)", r"\1 \2", text)
    return text.strip()


def fix_dialogue_spacing(output_text: str) -> str:
    return clean_generated_text(output_text)


def extract_answer(output_text: str, prompt_text: str) -> str:
    output_text = clean_special_tokens(output_text)

    if output_text.startswith(prompt_text):
        answer = output_text[len(prompt_text):]
    else:
        answer = output_text

    answer = clean_generated_text(answer)

    if answer.startswith("Asistan:"):
        answer = answer[len("Asistan:"):].strip()

    return answer


def stop_after_first_answer(output_text: str, prompt_text: str) -> str:
    answer = extract_answer(output_text, prompt_text)
    return f"{prompt_text}{answer}".strip()


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> ByteLevelBPETokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def load_checkpoint_model(
    checkpoint_path: Path,
    device: str,
) -> tuple[GPTLanguageModel, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits

    for token_id in set(generated_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= repetition_penalty
        else:
            logits[0, token_id] /= repetition_penalty

    return logits


@torch.no_grad()
def generate_with_controls(
    model: GPTLanguageModel,
    tokenizer: ByteLevelBPETokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    stop_on_user_marker: bool = True,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    encoded = tokenizer.encode(prompt_text)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    top_k_value = None if top_k <= 0 else min(top_k, model.config.vocab_size)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)

        if top_k_value is not None:
            values, _ = torch.topk(logits, top_k_value)
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if stop_on_user_marker:
            decoded_text = tokenizer.decode(idx[0].tolist())
            generated_part = decoded_text[len(prompt_text):] if decoded_text.startswith(prompt_text) else decoded_text

            if any(marker in generated_part for marker in TURN_STOP_MARKERS):
                break

    return tokenizer.decode(idx[0].tolist())


def write_samples_jsonl(
    output_path: Path,
    prompt: str,
    settings: dict,
    samples: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, sample in enumerate(samples, start=1):
            row = {
                "sample_index": index,
                "prompt": prompt,
                "settings": settings,
                "output": sample,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from a DarkMind checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )
    parser.add_argument("--prompt", type=str, default="DarkMind")
    parser.add_argument(
        "--dialogue",
        type=str,
        default=None,
        help="Dialogue mode. Example: --dialogue 'DarkMind hazır bir model mi?'",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop_at_next_user",
        action="store_true",
        help="If enabled, stop generation when the next dialogue turn starts.",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")

    if args.dialogue is not None:
        args.prompt = f"Kullanıcı: {args.dialogue.strip()}\nAsistan:"
        args.stop_at_next_user = True

    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output_path) if args.output_path else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)

    print("=" * 70)
    print("DarkMind checkpoint generation")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {checkpoint.get('run_name', 'unknown')}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Block size: {model.config.block_size}")

    settings = {
        "checkpoint": str(checkpoint_path),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "stop_at_next_user": args.stop_at_next_user,
    }

    samples = []

    for sample_index in range(args.num_samples):
        set_seed(args.seed + sample_index, device)
        output_text = generate_with_controls(
            model=model,
            tokenizer=tokenizer,
            prompt_text=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_on_user_marker=args.stop_at_next_user,
        )
        output_text = clean_generated_text(output_text)
        samples.append(output_text or SAFE_FALLBACK)

    if output_path:
        write_samples_jsonl(output_path, args.prompt, settings, samples)

    print("=" * 70)
    print("PROMPT:")
    print(args.prompt)

    for index, sample in enumerate(samples, start=1):
        print("=" * 70)
        print(f"OUTPUT {index}:")
        print(sample)

    if output_path:
        print("=" * 70)
        print(f"Samples saved: {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
