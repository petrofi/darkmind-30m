from pathlib import Path
import argparse
import json
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "splits" / "train.txt"
DEFAULT_EVAL_PATHS = [
    ROOT_DIR / "data" / "evals" / "darkmind_eval_v02.jsonl",
    ROOT_DIR / "data" / "evals" / "darkmind_code_eval_v01.jsonl",
]

PROMPT_FIELDS = ("prompt", "question", "instruction", "input")
ANSWER_FIELDS = (
    "expected_answer",
    "answer",
    "target",
    "completion",
    "response",
)
ANSWER_LIST_FIELDS = ("accepted_phrases", "expected_phrases")
KEYWORD_FIELDS = ("expected_keywords",)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}")

    return rows


def as_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]

    return []


def iter_eval_snippets(row: dict, min_answer_length: int):
    for field in PROMPT_FIELDS:
        for value in as_strings(row.get(field)):
            value = value.strip()

            if value:
                yield field, value

    for field in ANSWER_FIELDS:
        for value in as_strings(row.get(field)):
            value = value.strip()

            if len(value) >= min_answer_length:
                yield field, value

    for field in ANSWER_LIST_FIELDS:
        for value in as_strings(row.get(field)):
            value = value.strip()

            if len(value) >= min_answer_length:
                yield field, value

    for field in KEYWORD_FIELDS:
        for value in as_strings(row.get(field)):
            value = value.strip()

            if len(value) >= min_answer_length:
                yield field, value


def preview(text: str, limit: int = 100) -> str:
    one_line = " ".join(text.split())

    if len(one_line) <= limit:
        return one_line

    return one_line[:limit - 3] + "..."


def check_eval_file(
    train_text: str,
    eval_path: Path,
    min_answer_length: int,
) -> list[dict]:
    matches = []
    rows = load_jsonl(eval_path)

    for index, row in enumerate(rows, start=1):
        eval_id = row.get("id", f"row_{index}")

        for field, snippet in iter_eval_snippets(row, min_answer_length):
            if snippet in train_text:
                matches.append({
                    "eval_path": eval_path,
                    "id": eval_id,
                    "field": field,
                    "snippet": snippet,
                })

    return matches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether eval prompts or answer snippets leak into train text."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(DEFAULT_TRAIN_PATH.relative_to(ROOT_DIR)),
        help="Train split path.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        action="append",
        default=None,
        help="Eval JSONL path. Can be repeated.",
    )
    parser.add_argument(
        "--min_answer_length",
        type=int,
        default=12,
        help="Minimum answer snippet length checked for leakage.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero code if suspicious matches are found.",
    )
    args = parser.parse_args()

    train_path = resolve_path(args.train_path)

    if args.eval_path:
        eval_paths = [resolve_path(path) for path in args.eval_path]
    else:
        eval_paths = [path for path in DEFAULT_EVAL_PATHS if path.exists()]

    if not train_path.exists():
        print(f"WARNING: train file not found: {train_path}")
        return 1 if args.strict else 0

    train_text = train_path.read_text(encoding="utf-8-sig")
    all_matches = []

    for eval_path in eval_paths:
        if not eval_path.exists():
            print(f"WARNING: eval file not found, skipped: {eval_path}")
            continue

        matches = check_eval_file(
            train_text,
            eval_path,
            args.min_answer_length,
        )
        all_matches.extend(matches)

    print("=" * 70)
    print(f"Train file: {train_path}")
    print(f"Eval files checked: {len(eval_paths):,}")
    print(f"Suspicious matches: {len(all_matches):,}")

    if all_matches:
        print("-" * 70)
        print("Possible leakage:")

        for match in all_matches:
            relative_eval = match["eval_path"].relative_to(ROOT_DIR)
            print(
                f"- {relative_eval} | id={match['id']} | "
                f"field={match['field']} | {preview(match['snippet'])}"
            )

        print("-" * 70)
        print("WARNING: Review these matches before training or reporting eval results.")
    else:
        print("No exact prompt/snippet matches found.")

    print("=" * 70)

    if all_matches and args.strict:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
