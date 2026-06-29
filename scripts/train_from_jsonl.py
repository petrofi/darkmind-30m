from pathlib import Path
import argparse
import json
import sys

import torch
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


DEFAULT_DATA = ROOT_DIR / "data" / "processed" / "pretrain_corpus.jsonl"
DEFAULT_TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_SAVE_PATH = ROOT_DIR / "models" / "darkmind-30m-pretrain.pt"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_texts(path: Path) -> list[str]:
    texts = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            text = row.get("text", "")

            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    return texts


def encode_texts(texts: list[str], tokenizer: ByteLevelBPETokenizer) -> list[int]:
    ids: list[int] = []

    for text in tqdm(texts, desc="tokenizing", unit="doc"):
        ids.extend(tokenizer.encode(text).ids)
        ids.append(2)

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small DarkMind GPT model from pretraining JSONL text."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_SAVE_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=str(DEFAULT_TOKENIZER_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")

    data_path = resolve_path(args.data)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    save_path = resolve_path(args.save_path)
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    vocab_size = len(load_json(vocab_path))
    texts = load_texts(data_path)

    if not texts:
        raise ValueError(f"No usable text rows found in {data_path}")

    ids = encode_texts(texts, tokenizer)

    if len(ids) <= args.block_size + 1:
        raise ValueError("Dataset is too small for the selected block_size.")

    tokens = torch.tensor(ids, dtype=torch.long)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(tokens) // (args.batch_size * args.block_size))
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs)

    print("=" * 70)
    print("DarkMind JSONL pretraining")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Documents: {len(texts):,}")
    print(f"Tokens: {len(tokens):,}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Steps: {total_steps:,}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    def get_batch():
        ix = torch.randint(
            len(tokens) - args.block_size - 1,
            (args.batch_size,),
        )
        x = torch.stack([tokens[i:i + args.block_size] for i in ix])
        y = torch.stack([tokens[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model.train()
    final_loss = None

    for step in tqdm(range(1, total_steps + 1), desc="training", unit="step"):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": "darkmind_jsonl_pretrain",
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "training_config": {
                "data": str(data_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "lr": args.lr,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
            "vocab_size": vocab_size,
            "final_train_loss": final_loss,
        },
        save_path,
    )

    print("=" * 70)
    print("Training completed.")
    print(f"Checkpoint saved: {save_path}")
    print(f"Final train loss: {final_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
