from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT_DIR / "data" / "sources"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"


def read_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def collect_input_files(clean_input_dir: Path) -> list[Path]:
    source_files = sorted(SOURCE_DIR.glob("*.txt"))
    cleaned_files = sorted(clean_input_dir.rglob("*.txt"))
    return source_files + cleaned_files


def main():
    parser = argparse.ArgumentParser(
        description="Build corpus_v3 from data/sources and cleaned/deduped text."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=(
            "Optional cleaned input directory override, for example "
            "data/deduped. Defaults to data/cleaned."
        ),
    )
    args = parser.parse_args()

    clean_input_dir = (
        resolve_path(args.input_dir)
        if args.input_dir
        else CLEANED_DIR
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(clean_input_dir)

    if not input_files:
        raise FileNotFoundError(
            f"No .txt files found in {SOURCE_DIR} or {clean_input_dir}"
        )

    documents = []
    total_document_chars = 0

    for path in input_files:
        text = read_document(path)

        if not text:
            continue

        documents.append(f"<s>\n{text}\n</s>")
        total_document_chars += len(text)

        print(f"Added: {path.relative_to(ROOT_DIR)} | chars={len(text):,}")

    corpus = "\n\n".join(documents)
    OUTPUT_PATH.write_text(corpus, encoding="utf-8")

    print("=" * 70)
    print(f"Corpus saved: {OUTPUT_PATH}")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Clean input directory: {clean_input_dir}")
    print(f"Documents: {len(documents):,}")
    print(f"Document characters: {total_document_chars:,}")
    print(f"Final characters: {len(corpus):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
