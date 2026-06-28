from pathlib import Path
import json
from tokenizers import ByteLevelBPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus_v2.txt"
SOURCE_DIR = ROOT_DIR / "data" / "sources"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"


def count_non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    if not VOCAB_PATH.exists() or not MERGES_PATH.exists():
        raise FileNotFoundError("Tokenizer files not found.")

    corpus = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = ByteLevelBPETokenizer(
        str(VOCAB_PATH),
        str(MERGES_PATH),
    )

    encoded = tokenizer.encode(corpus)

    source_files = sorted(SOURCE_DIR.glob("*.txt"))

    print("=" * 70)
    print("DarkMind Corpus Stats")
    print("=" * 70)

    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Source files: {len(source_files)}")
    print(f"Characters: {len(corpus):,}")
    print(f"Lines: {len(corpus.splitlines()):,}")
    print(f"Non-empty lines: {count_non_empty_lines(corpus):,}")
    print(f"Tokens: {len(encoded.ids):,}")

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"Tokenizer vocab size: {len(vocab):,}")

    print("-" * 70)
    print("Source file breakdown:")

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        token_count = len(tokenizer.encode(text).ids)

        print(
            f"- {path.name}: "
            f"chars={len(text):,}, "
            f"non_empty_lines={count_non_empty_lines(text):,}, "
            f"tokens={token_count:,}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()