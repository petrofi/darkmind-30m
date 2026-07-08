from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"
DEFAULT_METADATA_DIR = ROOT_DIR / "checkpoints" / "metadata"


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


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    return result.stdout.strip() or None


def tokenizer_info(tokenizer_dir: Path) -> dict:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    vocab_size = None

    if vocab_path.exists():
        vocab_size = len(load_json(vocab_path))

    return {
        "tokenizer_dir": display_path(tokenizer_dir),
        "vocab_size": vocab_size,
        "vocab_path": display_path(vocab_path),
        "merges_path": display_path(merges_path),
        "vocab_sha256": sha256_file(vocab_path),
        "merges_sha256": sha256_file(merges_path),
    }


def build_metadata(
    checkpoint_path: Path,
    config_path: Path,
    tokenizer_dir: Path,
    dataset_path: Path,
    eval_run: Path | None,
) -> dict:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint_path": display_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": display_path(config_path),
        "config_sha256": sha256_file(config_path),
        "tokenizer": tokenizer_info(tokenizer_dir),
        "dataset_path": display_path(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": git_commit_hash(),
        "eval_run": display_path(eval_run) if eval_run else None,
        "eval_run_sha256": sha256_file(eval_run) if eval_run else None,
    }


def default_output_path(checkpoint_path: Path) -> Path:
    return DEFAULT_METADATA_DIR / f"{checkpoint_path.stem}.metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write metadata for a DarkMind checkpoint without modifying it."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=str(DEFAULT_TOKENIZER_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_DATASET_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--eval_run", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    dataset_path = resolve_path(args.dataset_path)
    eval_run = resolve_path(args.eval_run) if args.eval_run else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    output_path = (
        resolve_path(args.output_path)
        if args.output_path
        else default_output_path(checkpoint_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        checkpoint_path,
        config_path,
        tokenizer_dir,
        dataset_path,
        eval_run,
    )
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Metadata saved: {output_path}")
    print(f"Tokenizer vocab size: {metadata['tokenizer']['vocab_size']}")
    print(f"Dataset hash: {metadata['dataset_sha256']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
