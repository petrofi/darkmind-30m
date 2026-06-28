from pathlib import Path
import argparse

from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "corpus_v2.txt"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def main():
    parser = argparse.ArgumentParser(description="Train the DarkMind tokenizer.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_PATH.relative_to(ROOT_DIR)),
        help="Training corpus path.",
    )
    args = parser.parse_args()

    data_path = resolve_path(args.data_path)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("DarkMind tokenizer eğitimi başlıyor...")
    print(f"Veri dosyası: {data_path}")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[str(data_path)],
        vocab_size=8000,
        min_frequency=1,
        special_tokens=[
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )

    tokenizer.save_model(str(TOKENIZER_DIR))

    print("Tokenizer eğitimi tamamlandı.")
    print(f"Kaydedildi: {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
