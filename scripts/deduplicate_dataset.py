from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "deduped"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def should_check_duplicate(
    line: str,
    in_code_block: bool,
    min_line_length: int,
) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if in_code_block or stripped.startswith("```"):
        return False

    if len(stripped) < min_line_length:
        return False

    return True


def deduplicate_text(
    text: str,
    seen_lines: set[str],
    min_line_length: int,
) -> tuple[str, int, int, int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    kept_lines = []
    duplicate_lines_removed = 0
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        is_code_fence = stripped.startswith("```")
        check_duplicate = should_check_duplicate(
            line,
            in_code_block,
            min_line_length,
        )

        if check_duplicate:
            if line in seen_lines:
                duplicate_lines_removed += 1

                if is_code_fence:
                    in_code_block = not in_code_block

                continue

            seen_lines.add(line)

        kept_lines.append(line)

        if is_code_fence:
            in_code_block = not in_code_block

    return (
        "\n".join(kept_lines).strip(),
        len(lines),
        len(kept_lines),
        duplicate_lines_removed,
    )


def process_file(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    seen_lines: set[str],
    min_line_length: int,
) -> tuple[int, int, int]:
    relative_path = input_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8-sig")
    deduped, original_lines, kept_lines, removed_lines = deduplicate_text(
        text,
        seen_lines,
        min_line_length,
    )

    output_path.write_text(deduped, encoding="utf-8")

    print(
        f"Deduped: {input_path.relative_to(ROOT_DIR)} -> "
        f"{output_path.relative_to(ROOT_DIR)} | "
        f"lines={original_lines:,}->{kept_lines:,} "
        f"removed={removed_lines:,}"
    )

    return original_lines, kept_lines, removed_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove exact duplicate non-empty lines from cleaned data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR.relative_to(ROOT_DIR)),
        help="Input directory containing cleaned .txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
        help="Output directory for deduplicated .txt files.",
    )
    parser.add_argument(
        "--min_line_length",
        type=int,
        default=20,
        help="Only deduplicate non-code lines with at least this many characters.",
    )
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    input_files = sorted(input_dir.rglob("*.txt")) if input_dir.exists() else []

    seen_lines: set[str] = set()
    total_original_lines = 0
    total_kept_lines = 0
    total_removed_lines = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        original_lines, kept_lines, removed_lines = process_file(
            input_path,
            input_dir,
            output_dir,
            seen_lines,
            args.min_line_length,
        )
        total_original_lines += original_lines
        total_kept_lines += kept_lines
        total_removed_lines += removed_lines

    print("=" * 70)
    print(f"Files processed: {len(input_files):,}")
    print(f"Original lines: {total_original_lines:,}")
    print(f"Kept lines: {total_kept_lines:,}")
    print(f"Duplicate lines removed: {total_removed_lines:,}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
