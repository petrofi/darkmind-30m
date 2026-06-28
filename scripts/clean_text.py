from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw_collected"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"


def normalize_line(line: str, in_code_block: bool) -> str:
    line = line.rstrip()

    if in_code_block:
        return line

    return re.sub(r"[ \t]+", " ", line).strip()


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    in_code_block = False

    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        is_code_fence = stripped.startswith("```")

        line = normalize_line(raw_line, in_code_block or is_code_fence)

        if len(line.strip()) >= 3:
            cleaned_lines.append(line)

        if is_code_fence:
            in_code_block = not in_code_block

    return "\n".join(cleaned_lines).strip()


def clean_file(path: Path) -> tuple[Path, int, int]:
    relative_path = path.relative_to(RAW_DIR)
    output_path = CLEANED_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = path.read_text(encoding="utf-8-sig")
    cleaned = clean_text(raw_text)
    output_path.write_text(cleaned, encoding="utf-8")

    return output_path, len(raw_text), len(cleaned)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.rglob("*.txt"))

    if not raw_files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    total_raw_chars = 0
    total_cleaned_chars = 0

    for path in raw_files:
        output_path, raw_chars, cleaned_chars = clean_file(path)
        total_raw_chars += raw_chars
        total_cleaned_chars += cleaned_chars

        print(
            f"Cleaned: {path.relative_to(ROOT_DIR)} -> "
            f"{output_path.relative_to(ROOT_DIR)} | "
            f"chars={raw_chars:,}->{cleaned_chars:,}"
        )

    print("=" * 70)
    print(f"Files cleaned: {len(raw_files)}")
    print(f"Raw characters: {total_raw_chars:,}")
    print(f"Cleaned characters: {total_cleaned_chars:,}")
    print(f"Output directory: {CLEANED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
