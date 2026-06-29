from pathlib import Path
import argparse
import random


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "splits"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def parse_documents(text: str) -> list[str]:
    documents = []
    current = []
    in_document = False

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip() == "<s>" and not in_document:
            current = [line]
            in_document = True
            continue

        if in_document:
            current.append(line)

            if line.strip() == "</s>":
                documents.append("\n".join(current).strip())
                current = []
                in_document = False

    if current:
        documents.append("\n".join(current).strip())

    if documents:
        return [document for document in documents if document.strip()]

    stripped = text.strip()
    return [stripped] if stripped else []


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0 "
            f"(got {total})"
        )

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")


def write_split(path: Path, documents: list[str]) -> int:
    text = "\n\n".join(documents)

    if text:
        text += "\n"

    path.write_text(text, encoding="utf-8")
    return len(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build document-level train/val/test splits."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DEFAULT_INPUT_PATH.relative_to(ROOT_DIR)),
        help="Input corpus path with <s>...</s> document blocks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
        help="Output directory for train.txt, val.txt, and test.txt.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_path = resolve_path(args.input_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8-sig")
    documents = parse_documents(text)

    if not documents:
        raise ValueError(f"No documents found in {input_path}")

    rng = random.Random(args.seed)
    rng.shuffle(documents)

    total_documents = len(documents)
    train_count = int(total_documents * args.train_ratio)
    val_count = int(total_documents * args.val_ratio)

    train_docs = documents[:train_count]
    val_docs = documents[train_count:train_count + val_count]
    test_docs = documents[train_count + val_count:]

    split_docs = {
        "train": train_docs,
        "val": val_docs,
        "test": test_docs,
    }

    print("=" * 70)
    print(f"Input corpus: {input_path}")
    print(f"Documents: {total_documents:,}")
    print(f"Output directory: {output_dir}")
    print("-" * 70)

    for split_name, docs in split_docs.items():
        output_path = output_dir / f"{split_name}.txt"
        char_count = write_split(output_path, docs)
        percentage = (len(docs) / total_documents) * 100

        print(
            f"{split_name}: docs={len(docs):,} "
            f"chars={char_count:,} percent={percentage:.2f}% "
            f"path={output_path}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
