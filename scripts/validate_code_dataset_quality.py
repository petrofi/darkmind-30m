from collections import Counter
from pathlib import Path
import argparse
import ast
import re
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "python_examples"
    / "turkish_code_factory_v01.txt"
)

UNSAFE_PATTERNS = [
    ("rm -rf", re.compile(re.escape("rm -rf"), re.IGNORECASE)),
    ("del /s", re.compile(re.escape("del /s"), re.IGNORECASE)),
    ("format c:", re.compile(re.escape("format c:"), re.IGNORECASE)),
    ("os.remove", re.compile(re.escape("os.remove"), re.IGNORECASE)),
    ("shutil.rmtree", re.compile(re.escape("shutil.rmtree"), re.IGNORECASE)),
    ("subprocess", re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("eval(", re.compile(re.escape("eval("), re.IGNORECASE)),
    ("exec(", re.compile(re.escape("exec("), re.IGNORECASE)),
    ("requests.post", re.compile(re.escape("requests.post"), re.IGNORECASE)),
    ("token", re.compile(r"\btoken\b", re.IGNORECASE)),
    ("password", re.compile(r"\bpassword\b", re.IGNORECASE)),
    ("credential", re.compile(r"\bcredential\b", re.IGNORECASE)),
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def split_examples(text: str) -> list[str]:
    parts = re.split(r"(?=^# bucket: )", text, flags=re.MULTILINE)
    return [part.strip() for part in parts if part.strip()]


def extract_prompt(example: str) -> str | None:
    match = re.search(r"^Kullanıcı:\s*(.*)$", example, flags=re.MULTILINE)

    if not match:
        return None

    return match.group(1).strip()


def extract_answer(example: str) -> str:
    marker = "Asistan:"

    if marker not in example:
        return ""

    return example.split(marker, 1)[1].strip()


def extract_python_blocks(example: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```python\s*\n(.*?)```",
            example,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]


def find_unsafe_hits(text: str) -> list[tuple[str, int]]:
    hits = []

    for label, pattern in UNSAFE_PATTERNS:
        count = len(pattern.findall(text))

        if count:
            hits.append((label, count))

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Turkish code instruction dataset quality."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_PATH.relative_to(ROOT_DIR)),
        help="Dataset path to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when quality errors are found.",
    )
    args = parser.parse_args()

    dataset_path = resolve_path(args.path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    text = dataset_path.read_text(encoding="utf-8-sig")
    examples = split_examples(text)
    prompt_counts = Counter()
    warnings = []
    syntax_failed = []
    syntax_passed = 0
    code_block_count = 0
    missing_user = 0
    missing_assistant = 0
    empty_answers = 0
    very_long_answers = 0

    if text.count("```") % 2 != 0:
        warnings.append("Code fence count is not balanced.")

    unsafe_hits = find_unsafe_hits(text)

    for index, example in enumerate(examples, start=1):
        prompt = extract_prompt(example)
        answer = extract_answer(example)

        if prompt:
            prompt_counts[prompt] += 1
        else:
            missing_user += 1

        if "Asistan:" not in example:
            missing_assistant += 1

        if not answer:
            empty_answers += 1

        if len(answer) > 2500:
            very_long_answers += 1

        for code in extract_python_blocks(example):
            code_block_count += 1

            try:
                ast.parse(code)
                syntax_passed += 1
            except SyntaxError as exc:
                syntax_failed.append((index, exc.lineno, exc.msg))

    duplicate_prompts = [
        (prompt, count)
        for prompt, count in prompt_counts.items()
        if count > 1
    ]

    hard_error_count = (
        len(syntax_failed)
        + len(unsafe_hits)
        + len(duplicate_prompts)
        + missing_user
        + missing_assistant
        + empty_answers
    )

    print("=" * 70)
    print("Turkish Code Dataset Quality")
    print("=" * 70)
    print(f"Path: {dataset_path}")
    print(f"Total examples: {len(examples):,}")
    print(f"Code blocks: {code_block_count:,}")
    print(f"Syntax passed: {syntax_passed:,}")
    print(f"Syntax failed: {len(syntax_failed):,}")
    print(f"Unsafe hits: {sum(count for _, count in unsafe_hits):,}")
    print(f"Duplicate prompts: {len(duplicate_prompts):,}")
    print(f"Examples without Kullanıcı: {missing_user:,}")
    print(f"Examples without Asistan: {missing_assistant:,}")
    print(f"Empty answers: {empty_answers:,}")
    print(f"Very long answers: {very_long_answers:,}")

    if unsafe_hits:
        print("-" * 70)
        print("Unsafe keyword hits:")

        for label, count in unsafe_hits:
            print(f"- {label}: {count}")

    if syntax_failed:
        print("-" * 70)
        print("Syntax failures:")

        for example_index, line_number, message in syntax_failed[:20]:
            print(f"- example={example_index} line={line_number}: {message}")

    if duplicate_prompts:
        print("-" * 70)
        print("Duplicate prompts:")

        for prompt, count in duplicate_prompts[:20]:
            print(f"- {count}x | {prompt}")

    if warnings:
        print("-" * 70)
        print("Warnings:")

        for warning in warnings:
            print(f"- {warning}")
    else:
        print("-" * 70)
        print("Warnings: none")

    print("=" * 70)

    if args.strict and (hard_error_count or warnings):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
