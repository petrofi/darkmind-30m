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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DarkMind GPT from config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file. Example: configs/darkmind_tiny.json",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional corpus path override.",
    )
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    cfg = load_json(config_path)

    run_name = cfg["run_name"]
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = ROOT_DIR / (args.data_path or cfg["data_path"])
    tokenizer_dir = ROOT_DIR / cfg["tokenizer_dir"]
    checkpoint_path = ROOT_DIR / cfg["checkpoint_path"]

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    print("=" * 70)
    print(f"DarkMind training run: {run_name}")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = ByteLevelBPETokenizer(
        str(vocab_path),
        str(merges_path),
    )

    vocab = load_json(vocab_path)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")

    text = data_path.read_text(encoding="utf-8")
    encoded = tokenizer.encode(text)
    ids = encoded.ids

    print(f"Original token count: {len(ids)}")

    training_cfg = cfg["training"]
    min_tokens = training_cfg.get("min_tokens", 50000)

    while len(ids) < min_tokens:
        ids = ids + ids

    print(f"Training token count after repeat: {len(ids)}")

    tokens = torch.tensor(ids, dtype=torch.long)

    train_ratio = training_cfg.get("train_ratio", 0.9)
    split_idx = int(len(tokens) * train_ratio)

    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]

    model_cfg = cfg["model"]

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    )

    batch_size = training_cfg["batch_size"]
    max_steps = training_cfg["max_steps"]
    eval_interval = training_cfg["eval_interval"]
    eval_iters = training_cfg["eval_iters"]
    learning_rate = training_cfg["learning_rate"]

    model = GPTLanguageModel(gpt_config).to(device)

    print("-" * 70)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Block size: {gpt_config.block_size}")
    print(f"Layers: {gpt_config.n_layer}")
    print(f"Heads: {gpt_config.n_head}")
    print(f"Embedding size: {gpt_config.n_embd}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print("-" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch(split):
        data = train_data if split == "train" else val_data

        if len(data) <= gpt_config.block_size + 1:
            raise ValueError("Dataset is too small for the selected block_size.")

        ix = torch.randint(
            len(data) - gpt_config.block_size - 1,
            (batch_size,),
        )

        x = torch.stack([
            data[i:i + gpt_config.block_size]
            for i in ix
        ])

        y = torch.stack([
            data[i + 1:i + gpt_config.block_size + 1]
            for i in ix
        ])

        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}

        for split in ["train", "val"]:
            total_loss = 0.0

            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                total_loss += loss.item()

            losses[split] = total_loss / eval_iters

        model.train()
        return losses

    model.train()

    final_train_loss = None
    final_val_loss = None

    progress = tqdm(range(1, max_steps + 1), dynamic_ncols=True)

    for step in progress:
        xb, yb = get_batch("train")

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_description(f"step {step} loss {loss.item():.4f}")

        if step % eval_interval == 0:
            losses = estimate_loss()
            final_train_loss = losses["train"]
            final_val_loss = losses["val"]

            print(
                f"\nStep {step}: "
                f"train loss={final_train_loss:.4f}, "
                f"val loss={final_val_loss:.4f}"
            )

    torch.save(
        {
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "config": gpt_config.__dict__,
            "training_config": training_cfg,
            "vocab_size": vocab_size,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        },
        checkpoint_path,
    )

    print("=" * 70)
    print("Training completed.")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss: {final_val_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
