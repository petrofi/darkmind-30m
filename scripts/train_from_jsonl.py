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
DEFAULT_CONFIG = ROOT_DIR / "configs" / "darkmind_30m_1000step.json"
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


def pick_value(cli_value, config_values: dict, key: str, fallback):
    if cli_value is not None:
        return cli_value

    if key in config_values:
        return config_values[key]

    return fallback


def encode_texts(texts: list[str], tokenizer: ByteLevelBPETokenizer) -> list[int]:
    ids: list[int] = []

    for text in tqdm(texts, desc="tokenizing", unit="doc"):
        ids.extend(tokenizer.encode(text).ids)
        ids.append(2)

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the real DarkMind GPT model from pretraining JSONL text."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)),
        help="DarkMind config JSON used for the real model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_SAVE_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer directory override. Defaults to tokenizer_dir in --config.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Legacy alias for --tokenizer.",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be greater than or equal to 0 and less than 1")

    if args.val_ratio > 0 and args.eval_interval < 1:
        raise ValueError("--eval_interval must be at least 1")

    if args.val_ratio > 0 and args.eval_batches < 1:
        raise ValueError("--eval_batches must be at least 1")

    config_path = resolve_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_json(config_path)
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    if not model_cfg:
        raise ValueError(f"Config has no model section: {config_path}")

    data_path = resolve_path(args.data)
    tokenizer_arg = args.tokenizer or args.tokenizer_dir
    tokenizer_value = tokenizer_arg or cfg.get(
        "tokenizer_dir",
        str(DEFAULT_TOKENIZER_DIR.relative_to(ROOT_DIR)),
    )
    tokenizer_dir = resolve_path(tokenizer_value)
    save_path = resolve_path(args.save_path)
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    seed = pick_value(args.seed, cfg, "seed", 42)

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    vocab_size = len(load_json(vocab_path))
    texts = load_texts(data_path)

    if not texts:
        raise ValueError(f"No usable text rows found in {data_path}")

    ids = encode_texts(texts, tokenizer)

    block_size = pick_value(args.block_size, model_cfg, "block_size", 256)
    n_layer = pick_value(args.n_layer, model_cfg, "n_layer", 8)
    n_head = pick_value(args.n_head, model_cfg, "n_head", 8)
    n_embd = pick_value(args.n_embd, model_cfg, "n_embd", 512)
    dropout = pick_value(args.dropout, model_cfg, "dropout", 0.1)
    batch_size = pick_value(args.batch_size, training_cfg, "batch_size", 4)
    learning_rate = pick_value(args.lr, training_cfg, "learning_rate", 3e-4)

    if len(ids) <= block_size + 1:
        raise ValueError("Dataset is too small for the selected block_size.")

    tokens = torch.tensor(ids, dtype=torch.long)
    split_idx = int(len(tokens) * (1.0 - args.val_ratio))

    if args.val_ratio > 0:
        train_data = tokens[:split_idx]
        val_data = tokens[split_idx:]
    else:
        train_data = tokens
        val_data = None

    if len(train_data) <= block_size + 1:
        raise ValueError("Train split is too small for the selected block_size.")

    if val_data is not None and len(val_data) <= block_size + 1:
        raise ValueError("Validation split is too small for the selected block_size.")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPTLanguageModel(config).to(device)
    parameter_count = count_parameters(model)

    if args.resume_from:
        resume_path = resolve_path(args.resume_from)

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from: {resume_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = max(1, len(train_data) // (batch_size * block_size))
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs)

    print("=" * 70)
    print("DarkMind JSONL pretraining")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Data: {data_path}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Documents: {len(texts):,}")
    print(f"Tokens: {len(tokens):,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data) if val_data is not None else 0:,}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Architecture: real DarkMind GPTLanguageModel from model/gpt.py")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding size: {config.n_embd}")
    print(f"Block size: {config.block_size}")
    print(f"Dropout: {config.dropout}")
    print(f"Vocab size: {config.vocab_size:,}")
    print(f"Parameter count: {parameter_count:,}")
    if parameter_count < 20_000_000:
        print(
            "Warning: this config defines fewer than 20M parameters. "
            "Use a larger DarkMind config if a 30M-family run is expected."
        )
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Eval batches: {args.eval_batches}")
    print(f"Steps: {total_steps:,}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    def get_batch(split: str = "train"):
        data = train_data if split == "train" else val_data

        if data is None:
            raise ValueError("Validation data is not available.")

        ix = torch.randint(
            len(data) - block_size - 1,
            (batch_size,),
        )
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_losses() -> dict[str, float]:
        model.eval()
        losses = {}
        splits = ["train"]

        if val_data is not None:
            splits.append("val")

        for split in splits:
            split_losses = []

            for _ in range(args.eval_batches):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                split_losses.append(loss.item())

            losses[split] = sum(split_losses) / len(split_losses)

        model.train()
        return losses

    model.train()
    final_step_loss = None
    final_train_loss = None
    final_val_loss = None

    for step in tqdm(range(1, total_steps + 1), desc="training", unit="step"):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_step_loss = loss.item()

        should_eval = (
            args.val_ratio > 0
            and (step % args.eval_interval == 0 or step == total_steps)
        )

        if should_eval:
            losses = estimate_losses()
            final_train_loss = losses["train"]
            final_val_loss = losses.get("val")
            tqdm.write(
                f"Step {step}: "
                f"train loss={final_train_loss:.4f}, "
                f"val loss={final_val_loss:.4f}"
            )

    if final_train_loss is None:
        final_train_loss = final_step_loss

    if args.val_ratio > 0 and final_val_loss is None:
        losses = estimate_losses()
        final_train_loss = losses["train"]
        final_val_loss = losses.get("val")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": "darkmind_jsonl_pretrain",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "training_config": {
                "config": str(config_path),
                "data": str(data_path),
                "tokenizer": str(tokenizer_dir),
                "epochs": args.epochs,
                "batch_size": batch_size,
                "block_size": block_size,
                "lr": learning_rate,
                "max_steps": args.max_steps,
                "val_ratio": args.val_ratio,
                "eval_interval": args.eval_interval,
                "eval_batches": args.eval_batches,
                "seed": seed,
            },
            "train_tokens": len(train_data),
            "val_tokens": len(val_data) if val_data is not None else 0,
            "vocab_size": vocab_size,
            "parameter_count": parameter_count,
            "final_step_loss": final_step_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        },
        save_path,
    )

    print("=" * 70)
    print("Training completed.")
    print(f"Checkpoint saved: {save_path}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss: {final_val_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
