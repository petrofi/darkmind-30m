from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT_DIR / "data" / "sources"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"


def read_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def collect_input_files() -> list[Path]:
    source_files = sorted(SOURCE_DIR.glob("*.txt"))
    cleaned_files = sorted(CLEANED_DIR.rglob("*.txt"))
    return source_files + cleaned_files


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files()

    if not input_files:
        raise FileNotFoundError(
            f"No .txt files found in {SOURCE_DIR} or {CLEANED_DIR}"
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
    print(f"Documents: {len(documents):,}")
    print(f"Document characters: {total_document_chars:,}")
    print(f"Final characters: {len(corpus):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
