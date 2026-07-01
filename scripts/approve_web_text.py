from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "web_text"
    / "approved"
    / "web_text_approved_v01.txt"
)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def append_approved_text(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Input file is empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = output_path.exists() and output_path.stat().st_size > 0

    with output_path.open("a", encoding="utf-8") as file:
        if needs_separator:
            file.write("\n\n")

        file.write("=" * 70)
        file.write("\n")
        file.write(f"WEB SOURCE REVIEWED: {input_path}\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(text)
        file.write("\n")

    print("=" * 70)
    print("Approved web text appended.")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Metadata was not approved automatically.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append reviewed web text to approved raw training input."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Reviewed pending web text file.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        help="Approved web text output file.",
    )
    parser.add_argument(
        "--approve_all",
        action="store_true",
        help="Append the whole input file. Use only after human review.",
    )
    args = parser.parse_args()

    print("WARNING: Only approve content whose license and quality you reviewed.")

    if not args.approve_all:
        print("Refusing to approve without --approve_all.")
        print("Review the text and metadata first, then rerun with --approve_all.")
        raise SystemExit(1)

    append_approved_text(
        input_path=resolve_path(args.input_path),
        output_path=resolve_path(args.output_path),
    )


if __name__ == "__main__":
    main()
