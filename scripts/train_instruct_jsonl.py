from pathlib import Path
import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


DEFAULT_DATA = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_seed.jsonl"
DEFAULT_BASE_CHECKPOINT = ROOT_DIR / "models" / "darkmind-30m-10k-step15000.pt"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "darkmind_30m_1000step.json"
DEFAULT_TOKENIZER = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_SAVE_PATH = ROOT_DIR / "models" / "darkmind-30m-instruct-v0.1.pt"
DEFAULT_RUN_NAME = "darkmind_instruct_v0_3"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_records(path: Path) -> list[dict[str, str]]:
    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            prompt = row.get("prompt", "")
            response = row.get("response", "")

            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(f"prompt and response must be strings at {path}:{line_number}")

            prompt = prompt.strip()
            response = response.strip()

            if not prompt or not response:
                raise ValueError(f"Empty prompt or response at {path}:{line_number}")

            records.append({"prompt": prompt, "response": response})

    if not records:
        raise ValueError(f"No instruction records found in {path}")

    return records


def format_example(record: dict[str, str]) -> str:
    return f"Kullanıcı: {record['prompt']}\nAsistan: {record['response']}\n"


def format_prompt(record: dict[str, str]) -> str:
    return f"Kullanıcı: {record['prompt']}\nAsistan:"


def split_records(
    records: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    if val_ratio <= 0:
        return shuffled, []

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)

    return shuffled[val_count:], shuffled[:val_count]


def load_tokenizer(tokenizer_dir: Path) -> ByteLevelBPETokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def build_config(config_path: Path, vocab_size: int, block_size: int) -> GPTConfig:
    cfg = load_json(config_path)
    model_cfg = cfg["model"]

    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    )


def build_supervised_examples(
    records: list[dict[str, str]],
    tokenizer: ByteLevelBPETokenizer,
    block_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    examples = []

    for record in records:
        prompt_text = format_prompt(record)
        full_text = f"{prompt_text} {record['response']}\n"
        prompt_ids = tokenizer.encode(prompt_text).ids
        ids = tokenizer.encode(full_text).ids
        ids.append(2)

        if len(ids) < 2:
            continue

        ids = ids[:block_size + 1]
        x_ids = ids[:-1]
        y_ids = ids[1:]
        labels = []

        for label_index, token_id in enumerate(y_ids):
            original_token_index = label_index + 1

            if original_token_index < len(prompt_ids):
                labels.append(-100)
            else:
                labels.append(token_id)

        pad_len = block_size - len(x_ids)

        if pad_len > 0:
            x_ids.extend([0] * pad_len)
            labels.extend([-100] * pad_len)

        examples.append((
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        ))

    if not examples:
        raise ValueError("No supervised examples could be built.")

    return examples


def count_label_tokens(examples: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum((labels != -100).sum().item() for _, labels in examples)


def load_base_weights(
    model: GPTLanguageModel,
    checkpoint_path: Path,
    device: str,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    return checkpoint if isinstance(checkpoint, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instruction fine-tune the real DarkMind GPT model from JSONL."
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA.relative_to(ROOT_DIR)))
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=str(DEFAULT_BASE_CHECKPOINT.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)))
    parser.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER.relative_to(ROOT_DIR)))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH.relative_to(ROOT_DIR)))
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    if args.block_size < 2:
        raise ValueError("--block_size must be at least 2")

    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val_ratio must be greater than or equal to 0 and less than 1")

    if args.eval_interval < 1:
        raise ValueError("--eval_interval must be at least 1")

    if args.eval_batches < 1:
        raise ValueError("--eval_batches must be at least 1")

    data_path = resolve_path(args.data)
    base_checkpoint_path = resolve_path(args.base_checkpoint)
    config_path = resolve_path(args.config)
    tokenizer_path = resolve_path(args.tokenizer)
    save_path = resolve_path(args.save_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Instruction data not found: {data_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(load_json(tokenizer_path / "vocab.json"))
    records = load_records(data_path)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    train_examples = build_supervised_examples(
        train_records,
        tokenizer,
        args.block_size,
    )
    val_examples = []

    if val_records:
        val_examples = build_supervised_examples(
            val_records,
            tokenizer,
            args.block_size,
        )

    config = build_config(config_path, vocab_size, args.block_size)
    model = GPTLanguageModel(config).to(device)
    base_checkpoint = load_base_weights(model, base_checkpoint_path, device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.max_steps

    print("=" * 70)
    print("DarkMind instruction fine-tuning")
    print("=" * 70)
    print(f"Run name: {args.run_name}")
    print(f"Data: {data_path}")
    print(f"Base checkpoint: {base_checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Instruction examples: {len(records):,}")
    print(f"Train examples: {len(train_records):,}")
    print(f"Validation examples: {len(val_records):,}")
    print(f"Train supervised label tokens: {count_label_tokens(train_examples):,}")
    print(f"Validation supervised label tokens: {count_label_tokens(val_examples) if val_examples else 0:,}")
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Architecture: real DarkMind GPTLanguageModel from model/gpt.py")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding size: {config.n_embd}")
    print(f"Block size: {config.block_size}")
    print(f"Vocab size: {config.vocab_size:,}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max steps: {total_steps:,}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Eval batches: {args.eval_batches}")
    print(f"Base run name: {base_checkpoint.get('run_name', 'unknown')}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        examples = train_examples if split == "train" else val_examples

        if not examples:
            raise ValueError("Validation data is not available.")

        indices = torch.randint(len(examples), (args.batch_size,))
        x = torch.stack([examples[index][0] for index in indices])
        y = torch.stack([examples[index][1] for index in indices])
        return x.to(device), y.to(device)

    def compute_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        logits, _ = model(xb)
        return F.cross_entropy(
            logits.view(-1, config.vocab_size),
            yb.view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def estimate_losses() -> dict[str, float]:
        model.eval()
        splits = ["train"]

        if val_examples:
            splits.append("val")

        losses = {}

        for split in splits:
            split_losses = []

            for _ in range(args.eval_batches):
                xb, yb = get_batch(split)
                loss = compute_loss(xb, yb)
                split_losses.append(loss.item())

            losses[split] = sum(split_losses) / len(split_losses)

        model.train()
        return losses

    model.train()
    final_step_loss = None
    final_train_loss = None
    final_val_loss = None

    for step in tqdm(range(1, total_steps + 1), desc="instruct training", unit="step"):
        xb, yb = get_batch("train")
        loss = compute_loss(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_step_loss = loss.item()

        if step % args.eval_interval == 0 or step == total_steps:
            losses = estimate_losses()
            final_train_loss = losses["train"]
            final_val_loss = losses.get("val")

            if final_val_loss is None:
                tqdm.write(f"Step {step}: train loss={final_train_loss:.4f}")
            else:
                tqdm.write(
                    f"Step {step}: "
                    f"train loss={final_train_loss:.4f}, "
                    f"val loss={final_val_loss:.4f}"
                )

    if final_train_loss is None:
        final_train_loss = final_step_loss

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_name": args.run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "tokenizer_path": str(tokenizer_path),
            "base_checkpoint": str(base_checkpoint_path),
            "data_path": str(data_path),
            "instruction_count": len(records),
            "train_examples": len(train_records),
            "val_examples": len(val_records),
            "train_label_tokens": count_label_tokens(train_examples),
            "val_label_tokens": count_label_tokens(val_examples) if val_examples else 0,
            "parameter_count": parameter_count,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "final_step_loss": final_step_loss,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "lr": args.lr,
                "val_ratio": args.val_ratio,
                "eval_interval": args.eval_interval,
                "eval_batches": args.eval_batches,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
        },
        save_path,
    )

    print("=" * 70)
    print("Instruction fine-tuning completed.")
    print(f"Checkpoint saved: {save_path}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss: {final_val_loss}")
    print("=" * 70)


if __name__ == "__main__":
    main()
