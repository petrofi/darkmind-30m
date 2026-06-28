from pathlib import Path
import argparse
import sys

import torch
from tokenizers import ByteLevelBPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Generate text from a DarkMind checkpoint.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="DarkMind",
        help="Prompt text.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Maximum number of new tokens to generate.",
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
        help="Top-k sampling value.",
    )

    args = parser.parse_args()

    checkpoint_path = ROOT_DIR / args.checkpoint

    tokenizer_dir = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("DarkMind checkpoint generation")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = ByteLevelBPETokenizer(
        str(vocab_path),
        str(merges_path),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = GPTConfig(**checkpoint["config"])

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Run name: {checkpoint.get('run_name', 'unknown')}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Block size: {config.block_size}")

    encoded = tokenizer.encode(args.prompt)

    idx = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    top_k = min(args.top_k, config.vocab_size)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

    output_ids = generated[0].tolist()
    output_text = tokenizer.decode(output_ids)

    print("=" * 70)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 70)
    print("OUTPUT:")
    print(output_text)
    print("=" * 70)


if __name__ == "__main__":
    main()