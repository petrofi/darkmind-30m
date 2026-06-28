from collections import Counter
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def collect_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.txt"))

    return [path]


def normalize_duplicate_key(line: str) -> str:
    return " ".join(line.strip().split())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run a simple quality check on a DarkMind dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_PATH.relative_to(ROOT_DIR)),
        help="Dataset file or directory to inspect.",
    )
    args = parser.parse_args()

    target_path = resolve_path(args.path)

    if not target_path.exists():
        raise FileNotFoundError(f"Path not found: {target_path}")

    files = collect_files(target_path)

    if not files:
        raise FileNotFoundError(f"No .txt files found in {target_path}")

    file_texts = []
    file_char_counts = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        file_texts.append(text)
        file_char_counts.append((path, len(text)))

    combined_text = "\n".join(file_texts)
    lines = combined_text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    duplicate_keys = Counter(normalize_duplicate_key(line) for line in non_empty_lines)
    duplicate_line_count = sum(
        count - 1
        for count in duplicate_keys.values()
        if count > 1
    )

    repeated_lines = [
        (line, count)
        for line, count in duplicate_keys.most_common()
        if count > 1
    ][:20]

    print("=" * 70)
    print("DarkMind Dataset Quality Check")
    print("=" * 70)
    print(f"Path: {target_path}")
    print(f"Files inspected: {len(files):,}")
    print(f"Total characters: {len(combined_text):,}")
    print(f"Total lines: {len(lines):,}")
    print(f"Non-empty lines: {len(non_empty_lines):,}")
    print(f"Approx duplicate line count: {duplicate_line_count:,}")

    print("-" * 70)
    print("Top repeated non-empty lines:")

    if repeated_lines:
        for line, count in repeated_lines:
            preview = line if len(line) <= 140 else f"{line[:137]}..."
            print(f"{count:>5}x | {preview}")
    else:
        print("No repeated non-empty lines found.")

    print("-" * 70)
    print("File-level character counts:")

    for path, char_count in file_char_counts:
        print(f"- {display_path(path)}: {char_count:,}")

    print("=" * 70)


if __name__ == "__main__":
    main()
