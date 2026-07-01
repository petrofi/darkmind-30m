from pathlib import Path
import argparse
import hashlib
import json
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

DEFAULT_TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
METADATA_DIR = ROOT_DIR / "checkpoints" / "metadata"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_metadata_path(checkpoint_path: Path) -> Path | None:
    candidates = [
        METADATA_DIR / f"{checkpoint_path.stem}.metadata.json",
        checkpoint_path.with_suffix(".metadata.json"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def tokenizer_state(tokenizer_dir: Path) -> dict:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    vocab_size = None

    if vocab_path.exists():
        vocab_size = len(load_json(vocab_path))

    return {
        "vocab_size": vocab_size,
        "vocab_sha256": sha256_file(vocab_path),
        "merges_sha256": sha256_file(merges_path),
    }


def compare_config(checkpoint_config: dict, config_path: Path) -> list[str]:
    warnings = []

    if not config_path.exists():
        warnings.append(f"Config not found: {config_path}")
        return warnings

    config = load_json(config_path)
    model_config = config.get("model", {})

    for key in ["block_size", "n_layer", "n_head", "n_embd", "dropout"]:
        if key in model_config and checkpoint_config.get(key) != model_config[key]:
            warnings.append(
                f"Config mismatch for {key}: checkpoint={checkpoint_config.get(key)} config={model_config[key]}"
            )

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check tokenizer/config compatibility for a DarkMind checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=str(DEFAULT_TOKENIZER_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    config_path = resolve_path(args.config) if args.config else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    warnings = []
    metadata_path = find_metadata_path(checkpoint_path)
    metadata = load_json(metadata_path) if metadata_path else None
    tokenizer = tokenizer_state(tokenizer_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_vocab_size = checkpoint.get(
        "vocab_size",
        checkpoint_config.get("vocab_size"),
    )

    if tokenizer["vocab_size"] != checkpoint_vocab_size:
        warnings.append(
            f"Tokenizer vocab size mismatch: tokenizer={tokenizer['vocab_size']} checkpoint={checkpoint_vocab_size}"
        )

    if metadata:
        metadata_tokenizer = metadata.get("tokenizer", {})

        if metadata_tokenizer.get("vocab_sha256") != tokenizer["vocab_sha256"]:
            warnings.append("Tokenizer vocab hash differs from checkpoint metadata.")

        if metadata_tokenizer.get("merges_sha256") != tokenizer["merges_sha256"]:
            warnings.append("Tokenizer merges hash differs from checkpoint metadata.")
    else:
        warnings.append("No checkpoint metadata found.")

    if config_path:
        warnings.extend(compare_config(checkpoint_config, config_path))

    compatible = len(warnings) == 0

    print("=" * 70)
    print("DarkMind checkpoint compatibility")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Metadata: {metadata_path if metadata_path else 'not found'}")
    print(f"Checkpoint vocab size: {checkpoint_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer['vocab_size']}")
    print(f"Appears compatible: {compatible}")

    if warnings:
        print("-" * 70)
        print("Warnings:")

        for warning in warnings:
            print(f"- {warning}")

    print("=" * 70)

    if warnings and args.strict:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
