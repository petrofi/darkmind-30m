from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "qa_pairs"
    / "self_improvement_approved_v01.txt"
)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def append_reviewed_candidates(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_text = input_path.read_text(encoding="utf-8").strip()

    if not input_text:
        raise ValueError(f"Input file is empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = output_path.exists() and output_path.stat().st_size > 0

    with output_path.open("a", encoding="utf-8") as file:
        if needs_separator:
            file.write("\n\n")

        file.write(input_text)
        file.write("\n")

    print("=" * 70)
    print("Approved candidates appended.")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append reviewed correction candidates to raw training data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Reviewed candidate file to approve.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        help="Raw collected output file for approved examples.",
    )
    parser.add_argument(
        "--approve_all",
        action="store_true",
        help="Append the whole input file. Use only after human review.",
    )
    args = parser.parse_args()

    print("WARNING: Only approve candidates you have reviewed.")

    if not args.approve_all:
        print("Refusing to auto-approve without --approve_all.")
        print("Review the candidate file first, then rerun with --approve_all.")
        raise SystemExit(1)

    append_reviewed_candidates(
        input_path=resolve_path(args.input_path),
        output_path=resolve_path(args.output_path),
    )


if __name__ == "__main__":
    main()
