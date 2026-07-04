from collections import Counter
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT_DIR / "data" / "raw_collected" / "web_text" / "approved"
SUSPICIOUS_MARKERS = [
    "cookie",
    "privacy policy",
    "javascript",
    "advertisement",
    "login",
    "subscribe",
    "all rights reserved",
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def collect_text_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.txt"))

    return [path]


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def inspect_markers(text: str) -> dict[str, int]:
    folded = text.casefold()
    return {
        marker: folded.count(marker.casefold())
        for marker in SUSPICIOUS_MARKERS
        if marker.casefold() in folded
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect approved web text before corpus build."
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH.relative_to(ROOT_DIR)),
        help="Approved web text file or directory.",
    )
    args = parser.parse_args()

    target_path = resolve_path(args.path)

    if not target_path.exists():
        raise FileNotFoundError(f"Path not found: {target_path}")

    files = collect_text_files(target_path)

    if not files:
        raise FileNotFoundError(f"No .txt files found in {target_path}")

    combined_parts = []
    file_char_counts = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        combined_parts.append(text)
        file_char_counts.append((path, len(text)))

    combined_text = "\n".join(combined_parts)
    lines = combined_text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    line_counts = Counter(normalize_line(line) for line in non_empty_lines)
    repeated_lines = [
        (line, count)
        for line, count in line_counts.most_common()
        if count > 1
    ]
    repeated_line_count = sum(count - 1 for _, count in repeated_lines)
    marker_counts = inspect_markers(combined_text)

    print("=" * 70)
    print("DarkMind Approved Web Data Quality Check")
    print("=" * 70)
    print(f"Path: {target_path}")
    print(f"Files inspected: {len(files):,}")
    print(f"Total characters: {len(combined_text):,}")
    print(f"Total lines: {len(lines):,}")
    print(f"Non-empty lines: {len(non_empty_lines):,}")
    print(f"Repeated lines: {repeated_line_count:,}")

    print("-" * 70)
    print("Top repeated lines:")

    if repeated_lines:
        for line, count in repeated_lines[:20]:
            preview = line if len(line) <= 140 else f"{line[:137]}..."
            print(f"{count:>5}x | {preview}")
    else:
        print("No repeated non-empty lines found.")

    print("-" * 70)
    print("Suspicious markers:")

    if marker_counts:
        for marker, count in marker_counts.items():
            print(f"- {marker}: {count}")
    else:
        print("No suspicious markers found.")

    print("-" * 70)
    print("File-level character counts:")

    for path, char_count in file_char_counts:
        print(f"- {display_path(path)}: {char_count:,}")

    print("=" * 70)


if __name__ == "__main__":
    main()
