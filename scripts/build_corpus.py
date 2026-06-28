from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT_DIR / "data" / "sources"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "corpus_v2.txt"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main():
    source_files = sorted(SOURCE_DIR.glob("*.txt"))

    if not source_files:
        raise FileNotFoundError(f"No .txt files found in {SOURCE_DIR}")

    parts = []

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        text = clean_text(text)

        if not text:
            continue

        document = f"<s>\n{text}\n</s>"
        parts.append(document)

        print(f"Added: {path.name} | chars={len(text)}")

    corpus = "\n\n".join(parts)
    OUTPUT_PATH.write_text(corpus, encoding="utf-8")

    print("=" * 70)
    print(f"Corpus saved: {OUTPUT_PATH}")
    print(f"Documents: {len(parts)}")
    print(f"Characters: {len(corpus)}")
    print("=" * 70)


if __name__ == "__main__":
    main()