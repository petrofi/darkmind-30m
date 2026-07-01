from pathlib import Path
import argparse
from datetime import datetime
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


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def read_token_ids(path: Path, tokenizer: ByteLevelBPETokenizer) -> list[int]:
    text = path.read_text(encoding="utf-8")
    return tokenizer.encode(text).ids


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Optional explicit train split path.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Optional explicit validation split path.",
    )
    args = parser.parse_args()

    if bool(args.train_path) != bool(args.val_path):
        parser.error("--train_path and --val_path must be provided together.")

    config_path = resolve_path(args.config)
    cfg = load_json(config_path)

    run_name = cfg["run_name"]
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = resolve_path(args.data_path or cfg["data_path"])
    train_path = resolve_path(args.train_path) if args.train_path else None
    val_path = resolve_path(args.val_path) if args.val_path else None
    tokenizer_dir = resolve_path(cfg["tokenizer_dir"])
    checkpoint_path = resolve_path(cfg.get(
        "checkpoint_path",
        f"checkpoints/{run_name}.pt",
    ))
    best_checkpoint_path = ROOT_DIR / "checkpoints" / f"{run_name}_best.pt"
    last_checkpoint_path = ROOT_DIR / "checkpoints" / f"{run_name}_last.pt"
    metrics_path = ROOT_DIR / "reports" / "training" / f"{run_name}_metrics.jsonl"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")

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

    training_cfg = cfg["training"]
    min_tokens = training_cfg.get("min_tokens", 50000)

    if train_path and val_path:
        train_ids = read_token_ids(train_path, tokenizer)
        val_ids = read_token_ids(val_path, tokenizer)

        print(f"Train path: {train_path}")
        print(f"Val path: {val_path}")
        print(f"Original train token count: {len(train_ids)}")
        print(f"Original val token count: {len(val_ids)}")

        while len(train_ids) < min_tokens:
            train_ids = train_ids + train_ids

        print(f"Training token count after repeat: {len(train_ids)}")

        train_data = torch.tensor(train_ids, dtype=torch.long)
        val_data = torch.tensor(val_ids, dtype=torch.long)
    else:
        ids = read_token_ids(data_path, tokenizer)

        print(f"Data path: {data_path}")
        print(f"Original token count: {len(ids)}")

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
    gradient_accumulation_steps = training_cfg.get(
        "gradient_accumulation_steps",
        1,
    )

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    model = GPTLanguageModel(gpt_config).to(device)

    print("-" * 70)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Block size: {gpt_config.block_size}")
    print(f"Layers: {gpt_config.n_layer}")
    print(f"Heads: {gpt_config.n_head}")
    print(f"Embedding size: {gpt_config.n_embd}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Max steps: {max_steps}")
    print(f"Metrics path: {metrics_path}")
    print(f"Best checkpoint path: {best_checkpoint_path}")
    print(f"Last checkpoint path: {last_checkpoint_path}")
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
    best_val_loss = None
    best_step = None

    def build_checkpoint(step: int | None) -> dict:
        return {
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "config": gpt_config.__dict__,
            "training_config": training_cfg,
            "data_path": str(data_path),
            "train_path": str(train_path) if train_path else None,
            "val_path": str(val_path) if val_path else None,
            "vocab_size": vocab_size,
            "step": step,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        }

    def save_checkpoint(path: Path, step: int | None) -> None:
        torch.save(build_checkpoint(step), path)

    progress = tqdm(range(1, max_steps + 1), dynamic_ncols=True)

    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            xb, yb = get_batch("train")

            _, loss = model(xb, yb)
            step_loss += loss.item()
            (loss / gradient_accumulation_steps).backward()

        optimizer.step()

        average_step_loss = step_loss / gradient_accumulation_steps
        progress.set_description(f"step {step} loss {average_step_loss:.4f}")

        if step % eval_interval == 0:
            losses = estimate_loss()
            final_train_loss = losses["train"]
            final_val_loss = losses["val"]
            metric = {
                "step": step,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "timestamp": current_timestamp(),
            }
            append_jsonl(metrics_path, metric)

            print(
                f"\nStep {step}: "
                f"train loss={final_train_loss:.4f}, "
                f"val loss={final_val_loss:.4f}"
            )

            if best_val_loss is None or final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_step = step
                save_checkpoint(best_checkpoint_path, step)
                print(f"New best checkpoint saved: {best_checkpoint_path}")

    save_checkpoint(last_checkpoint_path, max_steps)

    if checkpoint_path not in {best_checkpoint_path, last_checkpoint_path}:
        save_checkpoint(checkpoint_path, max_steps)

    print("=" * 70)
    print("Training completed.")
    print(f"Configured checkpoint saved: {checkpoint_path}")
    print(f"Best checkpoint saved: {best_checkpoint_path}")
    print(f"Last checkpoint saved: {last_checkpoint_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Best step: {best_step}")
    print(f"Best val loss: {best_val_loss}")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss: {final_val_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
