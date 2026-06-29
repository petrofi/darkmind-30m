from collections import Counter, defaultdict
from pathlib import Path
import argparse
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed_quality"
    / "scored"
    / "scored_dataset_v01.txt"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "curriculum"
DEFAULT_FINAL_OUTPUT = ROOT_DIR / "data" / "processed" / "corpus_curriculum_v01.txt"

BUCKETS = [
    "chat_basics",
    "identity",
    "fallback",
    "python_code",
    "coding_errors",
    "llm_concepts",
    "data_pipeline",
    "cuda_gpu",
    "project_workflow",
    "unknown",
]

FINAL_ORDER = [
    "chat_basics",
    "identity",
    "fallback",
    "llm_concepts",
    "python_code",
    "coding_errors",
    "project_workflow",
    "data_pipeline",
    "cuda_gpu",
    "unknown",
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def split_scored_examples(text: str) -> list[str]:
    parts = re.split(r"\n\s*---\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    return [part.strip() for part in parts if part.strip()]


def strip_scoring_headers(example: str) -> str:
    lines = []

    for line in example.splitlines():
        if line.startswith("# source:"):
            continue

        if line.startswith("# score:"):
            continue

        if line.startswith("# positives:"):
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def normalize_for_bucket(example: str) -> str:
    return example.casefold()


def categorize(example: str) -> str:
    lowered = normalize_for_bucket(example)

    if any(term in lowered for term in ["merhaba", "selam", "nasılsın", "naber"]):
        return "chat_basics"

    if any(term in lowered for term in ["sen kimsin", "darkmind nedir", "hazır bir model", "chatgpt gibi"]):
        return "identity"

    if any(term in lowered for term in ["bilmiyorum", "emin değil", "uydurmam", "yeterli bilgi"]):
        return "fallback"

    if any(term in lowered for term in ["typeerror", "valueerror", "filenotfounderror", "indentationerror", "traceback", "hata"]):
        return "coding_errors"

    if any(term in lowered for term in ["python", "append", "remove", "sorted", "liste", "sözlük", "json", "pathlib", "```python", "def "]):
        return "python_code"

    if any(term in lowered for term in ["tokenizer", "transformer", "dil modeli", "llm", "embedding", "checkpoint"]):
        return "llm_concepts"

    if any(term in lowered for term in ["clean_text", "corpus", "dataset", "veri pipeline", "data pipeline", "manifest", "dedup"]):
        return "data_pipeline"

    if any(term in lowered for term in ["cuda", "gpu", "torch.cuda", "rtx"]):
        return "cuda_gpu"

    if any(term in lowered for term in ["git", "commit", "push", "venv", "powershell", "readme", "eval", "experiment"]):
        return "project_workflow"

    return "unknown"


def wrap_document(example: str) -> str:
    return f"<s>\n{example.strip()}\n</s>"


def write_bucket(path: Path, examples: list[str]) -> int:
    text = "\n\n".join(wrap_document(example) for example in examples)

    if text:
        text += "\n"

    path.write_text(text, encoding="utf-8")
    return len(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a curriculum-style DarkMind corpus from scored examples."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DEFAULT_INPUT_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--final_output",
        type=str,
        default=str(DEFAULT_FINAL_OUTPUT.relative_to(ROOT_DIR)),
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input_path)
    output_dir = resolve_path(args.output_dir)
    final_output = resolve_path(args.final_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Scored dataset not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    scored_text = input_path.read_text(encoding="utf-8-sig")
    scored_examples = split_scored_examples(scored_text)
    buckets: dict[str, list[str]] = defaultdict(list)

    for scored_example in scored_examples:
        example = strip_scoring_headers(scored_example)
        bucket = categorize(example)
        buckets[bucket].append(example)

    bucket_counts = Counter({bucket: len(buckets[bucket]) for bucket in BUCKETS})
    bucket_char_counts = {}

    for bucket in BUCKETS:
        bucket_path = output_dir / f"{bucket}.txt"
        bucket_char_counts[bucket] = write_bucket(bucket_path, buckets[bucket])

    final_documents = []

    for bucket in FINAL_ORDER:
        final_documents.extend(wrap_document(example) for example in buckets[bucket])

    final_text = "\n\n".join(final_documents)

    if final_text:
        final_text += "\n"

    final_output.write_text(final_text, encoding="utf-8")

    print("=" * 70)
    print("DarkMind Curriculum Dataset")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Final output: {final_output}")
    print("Bucket counts:")

    for bucket in BUCKETS:
        print(
            f"- {bucket}: {bucket_counts[bucket]:,} "
            f"chars={bucket_char_counts[bucket]:,}"
        )

    print(f"Final character count: {len(final_text):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
