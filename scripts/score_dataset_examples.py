from collections import Counter
from pathlib import Path
import argparse
import ast
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "cleaned"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed_quality" / "scored"
DEFAULT_REJECTED_DIR = ROOT_DIR / "data" / "processed_quality" / "rejected"
DEFAULT_REPORT_PATH = (
    ROOT_DIR
    / "data"
    / "processed_quality"
    / "reports"
    / "dataset_score_report.md"
)

OUTPUT_FILE_NAME = "scored_dataset_v01.txt"
REJECTED_FILE_NAME = "rejected_dataset_v01.txt"

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_WORDS = {
    "ve",
    "bir",
    "için",
    "nasıl",
    "nedir",
    "neden",
    "kullanılır",
    "örnek",
    "hata",
    "değer",
    "metin",
}
TECHNICAL_TERMS = {
    "python",
    "fonksiyon",
    "liste",
    "sözlük",
    "json",
    "pathlib",
    "try",
    "except",
    "class",
    "tokenizer",
    "cuda",
    "pytorch",
    "torch",
    "transformer",
    "checkpoint",
    "corpus",
    "eval",
    "model",
    "dataset",
    "veri",
    "git",
}
PROJECT_TERMS = {
    "darkmind",
    "tokenizer",
    "cuda",
    "pytorch",
    "python",
    "transformer",
    "checkpoint",
    "corpus",
    "eval",
    "pipeline",
    "dataset",
}
BOILERPLATE_TERMS = {
    "cookie",
    "cookies",
    "privacy policy",
    "gizlilik politikası",
    "login",
    "log in",
    "sign in",
    "subscribe",
    "newsletter",
    "reklam",
    "advertisement",
    "terms of service",
    "kullanım şartları",
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_key(text: str) -> str:
    return " ".join(text.casefold().split())


def split_dialogue_examples(text: str) -> list[str]:
    examples = []
    current = []
    in_code_block = False

    for line in text.splitlines():
        stripped = line.strip()
        is_code_fence = stripped.startswith("```")
        starts_example = (
            not in_code_block
            and (
                stripped.startswith("# bucket:")
                or stripped.startswith("Kullanıcı:")
                or stripped.startswith("KullanÄ±cÄ±:")
            )
        )

        if starts_example and current:
            examples.append("\n".join(current).strip())
            current = []

        current.append(line)

        if is_code_fence:
            in_code_block = not in_code_block

    if current:
        examples.append("\n".join(current).strip())

    return [example for example in examples if example]


def split_plain_examples(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [part.strip() for part in parts if part.strip()]


def split_examples(text: str) -> list[str]:
    text = normalize_text(text)

    if not text:
        return []

    if "Kullanıcı:" in text or "KullanÄ±cÄ±:" in text or "# bucket:" in text:
        return split_dialogue_examples(text)

    return split_plain_examples(text)


def contains_turkish_text(text: str) -> bool:
    lowered = text.casefold()

    if any(char in text for char in TURKISH_CHARS):
        return True

    words = set(re.findall(r"\b\w+\b", lowered))
    return bool(words & TURKISH_WORDS)


def count_terms(text: str, terms: set[str]) -> int:
    lowered = text.casefold()
    return sum(1 for term in terms if term in lowered)


def extract_python_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```python\s*\n(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]


def has_valid_short_python_block(text: str) -> bool:
    for block in extract_python_blocks(text):
        if len(block) > 2000:
            continue

        try:
            ast.parse(block)
            return True
        except SyntaxError:
            continue

    return False


def answer_text(example: str) -> str:
    if "Asistan:" not in example:
        return ""

    return example.split("Asistan:", 1)[1].strip()


def question_text(example: str) -> str:
    for marker in ["Kullanıcı:", "KullanÄ±cÄ±:"]:
        if marker in example:
            after_marker = example.split(marker, 1)[1]
            return after_marker.split("\n", 1)[0].strip()

    return ""


def repeated_line_count(example: str) -> int:
    lines = [
        " ".join(line.strip().split())
        for line in example.splitlines()
        if line.strip()
    ]
    counts = Counter(lines)
    return sum(count - 1 for count in counts.values() if count > 1)


def too_many_urls(example: str) -> bool:
    return len(re.findall(r"https?://|www\.", example, flags=re.IGNORECASE)) > 2


def has_boilerplate(example: str) -> bool:
    lowered = example.casefold()
    return any(term in lowered for term in BOILERPLATE_TERMS)


def has_unfinished_dialogue(example: str) -> bool:
    stripped = example.rstrip()
    return stripped.endswith("Kullanıcı:") or stripped.endswith("Asistan:")


def answer_repeats_question(example: str) -> bool:
    question = normalize_key(question_text(example))
    answer = normalize_key(answer_text(example))

    if not question or not answer:
        return False

    return answer == question or answer.startswith(question) and len(answer) < len(question) + 20


def score_example(
    example: str,
    min_chars: int,
    max_chars: int,
    seen_examples: set[str],
) -> tuple[bool, int, list[str], list[str]]:
    score = 0
    positives = []
    negatives = []
    key = normalize_key(example)

    if key in seen_examples:
        negatives.append("duplicated exact example")
        return False, score, positives, negatives

    if len(example) < min_chars:
        negatives.append("too short")

    if len(example) > max_chars:
        negatives.append("too long")

    if example.count("```") % 2 != 0:
        negatives.append("broken code fences")

    if too_many_urls(example):
        negatives.append("too many URLs")

    if has_boilerplate(example):
        negatives.append("web boilerplate")

    if has_unfinished_dialogue(example):
        negatives.append("unfinished dialogue")

    if repeated_line_count(example) >= 3:
        negatives.append("repeated lines")

    if example.count("Kullanıcı:") > 1 or example.count("KullanÄ±cÄ±:") > 1:
        negatives.append("multiple user markers inside example")

    answer = answer_text(example)

    if "Asistan:" in example and not answer:
        negatives.append("empty answer")

    if answer_repeats_question(example):
        negatives.append("answer repeats question")

    if contains_turkish_text(example):
        score += 2
        positives.append("Turkish text")

    technical_hits = count_terms(example, TECHNICAL_TERMS)

    if technical_hits:
        score += min(technical_hits, 3)
        positives.append("technical terms")

    if "Kullanıcı:" in example and "Asistan:" in example:
        score += 2
        positives.append("clear dialogue structure")

    if has_valid_short_python_block(example):
        score += 2
        positives.append("valid short Python code block")

    if answer and len(answer) >= 20:
        score += 1
        positives.append("answer/explanation content")

    project_hits = count_terms(example, PROJECT_TERMS)

    if project_hits:
        score += min(project_hits, 2)
        positives.append("project relevant concepts")

    score -= len(negatives) * 2

    hard_reasons = {
        "duplicated exact example",
        "too short",
        "too long",
        "broken code fences",
        "too many URLs",
        "web boilerplate",
        "empty answer",
        "answer repeats question",
    }
    hard_reject = any(reason in hard_reasons for reason in negatives)
    accepted = score >= 2 and not hard_reject

    if accepted:
        seen_examples.add(key)

    return accepted, score, positives, negatives


def render_scored_example(
    source_path: Path,
    score: int,
    positives: list[str],
    example: str,
) -> str:
    return (
        f"# source: {display_path(source_path)}\n"
        f"# score: {score}\n"
        f"# positives: {', '.join(positives) if positives else 'none'}\n\n"
        f"{example.strip()}"
    )


def render_rejected_example(
    source_path: Path,
    score: int,
    reasons: list[str],
    example: str,
) -> str:
    return (
        f"# source: {display_path(source_path)}\n"
        f"# score: {score}\n"
        f"# rejection_reasons: {', '.join(reasons) if reasons else 'low score'}\n\n"
        f"{example.strip()}"
    )


def write_report(
    report_path: Path,
    total_examples: int,
    accepted_examples: list[str],
    rejected_examples: list[str],
    rejection_reasons: Counter,
    source_counts: Counter,
    accepted_chars: int,
    rejected_chars: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DarkMind Dataset Score Report",
        "",
        f"- Total examples: {total_examples:,}",
        f"- Accepted examples: {len(accepted_examples):,}",
        f"- Rejected examples: {len(rejected_examples):,}",
        f"- Accepted characters: {accepted_chars:,}",
        f"- Rejected characters: {rejected_chars:,}",
        "",
        "## Rejection Reasons",
        "",
    ]

    if rejection_reasons:
        for reason, count in rejection_reasons.most_common():
            lines.append(f"- {reason}: {count:,}")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Source Files", ""])

    for source, count in source_counts.most_common(20):
        lines.append(f"- {source}: {count:,}")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score cleaned DarkMind dataset examples before final training."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--rejected_dir",
        type=str,
        default=str(DEFAULT_REJECTED_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=str(DEFAULT_REPORT_PATH.relative_to(ROOT_DIR)),
    )
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=4000)
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    rejected_dir = resolve_path(args.rejected_dir)
    report_path = resolve_path(args.report_path)
    output_path = output_dir / OUTPUT_FILE_NAME
    rejected_path = rejected_dir / REJECTED_FILE_NAME

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.rglob("*.txt")) if input_dir.exists() else []
    seen_examples: set[str] = set()
    accepted_examples = []
    rejected_examples = []
    rejection_reasons = Counter()
    source_counts = Counter()

    for path in input_files:
        text = path.read_text(encoding="utf-8-sig")
        examples = split_examples(text)

        for example in examples:
            source_counts[display_path(path)] += 1
            accepted, score, positives, negatives = score_example(
                example,
                args.min_chars,
                args.max_chars,
                seen_examples,
            )

            if accepted:
                accepted_examples.append(
                    render_scored_example(path, score, positives, example)
                )
            else:
                reasons = negatives or ["low score"]
                rejected_examples.append(
                    render_rejected_example(path, score, reasons, example)
                )

                for reason in reasons:
                    rejection_reasons[reason] += 1

    output_path.write_text(
        "\n\n---\n\n".join(accepted_examples) + ("\n" if accepted_examples else ""),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "\n\n---\n\n".join(rejected_examples) + ("\n" if rejected_examples else ""),
        encoding="utf-8",
    )

    accepted_chars = sum(len(example) for example in accepted_examples)
    rejected_chars = sum(len(example) for example in rejected_examples)
    total_examples = len(accepted_examples) + len(rejected_examples)

    write_report(
        report_path,
        total_examples,
        accepted_examples,
        rejected_examples,
        rejection_reasons,
        source_counts,
        accepted_chars,
        rejected_chars,
    )

    print("=" * 70)
    print("DarkMind Dataset Example Scoring")
    print("=" * 70)
    print(f"Input files: {len(input_files):,}")
    print(f"Total examples: {total_examples:,}")
    print(f"Accepted examples: {len(accepted_examples):,}")
    print(f"Rejected examples: {len(rejected_examples):,}")
    print(f"Accepted characters: {accepted_chars:,}")
    print(f"Rejected characters: {rejected_chars:,}")
    print(f"Scored output: {output_path}")
    print(f"Rejected output: {rejected_path}")
    print(f"Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
