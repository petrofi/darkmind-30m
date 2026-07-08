from pathlib import Path
import argparse
import ast
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT_DIR / "data" / "raw_collected" / "python_examples"
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def collect_files(path: Path, max_files: int | None) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
    else:
        files = [path]

    if max_files is not None:
        files = files[:max_files]

    return files


def extract_code_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in CODE_BLOCK_RE.finditer(text)
    ]


def preview_code(code: str, max_lines: int = 8) -> str:
    lines = code.splitlines()
    preview = "\n".join(lines[:max_lines])

    if len(lines) > max_lines:
        preview += "\n..."

    return preview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Syntax-check Python code blocks in dataset files."
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH.relative_to(ROOT_DIR)),
        help="File or directory containing dataset .txt files.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional maximum number of files to inspect.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if syntax failures exist.",
    )
    args = parser.parse_args()

    target_path = resolve_path(args.path)

    if not target_path.exists():
        raise FileNotFoundError(f"Path not found: {target_path}")

    files = collect_files(target_path, args.max_files)

    if not files:
        raise FileNotFoundError(f"No .txt files found in {target_path}")

    code_blocks_found = 0
    syntax_passed = 0
    failures = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        code_blocks = extract_code_blocks(text)

        for block_index, code in enumerate(code_blocks, start=1):
            code_blocks_found += 1

            try:
                ast.parse(code)
                syntax_passed += 1
            except SyntaxError as exc:
                failures.append(
                    {
                        "file": path,
                        "block_index": block_index,
                        "message": str(exc),
                        "preview": preview_code(code),
                    }
                )

    print("=" * 70)
    print("DarkMind Python Example Validation")
    print("=" * 70)
    print(f"Files inspected: {len(files):,}")
    print(f"Code blocks found: {code_blocks_found:,}")
    print(f"Syntax passed: {syntax_passed:,}")
    print(f"Syntax failed: {len(failures):,}")

    if failures:
        print("-" * 70)
        print("Failures:")

        for failure in failures:
            try:
                display_path = failure["file"].relative_to(ROOT_DIR)
            except ValueError:
                display_path = failure["file"]

            print(f"- File: {display_path}")
            print(f"  Block: {failure['block_index']}")
            print(f"  Error: {failure['message']}")
            print("  Preview:")
            print(failure["preview"])

    print("=" * 70)

    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
