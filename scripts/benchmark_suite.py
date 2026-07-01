from pathlib import Path
import argparse
from datetime import datetime
import json
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "benchmarks"
DEFAULT_EVAL_PATHS = [
    ROOT_DIR / "data" / "evals" / "darkmind_eval_v02.jsonl",
    ROOT_DIR / "data" / "evals" / "darkmind_code_eval_v02.jsonl",
]


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


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def collect_default_eval_paths() -> list[Path]:
    return [path for path in DEFAULT_EVAL_PATHS if path.exists()]


def run_command(command: list[str]) -> None:
    print("=" * 70)
    print(" ".join(command))
    print("=" * 70)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def summarize_eval_run(eval_path: Path, eval_run_path: Path) -> dict:
    rows = load_jsonl(eval_run_path)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0

    return {
        "eval_path": display_path(eval_path),
        "eval_run_path": display_path(eval_run_path),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
    }


def write_json_summary(path: Path, summary: dict) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown_summary(path: Path, summary: dict) -> None:
    lines = [
        f"# Benchmark Suite - {summary['checkpoint']}",
        "",
        f"- Timestamp: {summary['timestamp']}",
        f"- Overall weighted pass rate: {summary['overall_pass_rate']:.2%}",
        f"- Total eval items: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        "",
        "## Eval Runs",
        "",
        "| Eval file | Eval run | Total | Passed | Failed | Pass rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for item in summary["evals"]:
        lines.append(
            f"| `{item['eval_path']}` | `{item['eval_run_path']}` | "
            f"{item['total']} | {item['passed']} | {item['failed']} | "
            f"{item['pass_rate']:.2%} |"
        )

    lines.extend([
        "",
        "A benchmark result is evidence, not a guarantee of general ability.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DarkMind benchmark evals and summarize pass rates."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
    )
    parser.add_argument(
        "--eval_paths",
        type=str,
        action="append",
        default=None,
        help="Eval JSONL path. Can be repeated.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    timestamp = timestamp_for_filename()

    if args.eval_paths:
        eval_paths = [resolve_path(path) for path in args.eval_paths]
    else:
        eval_paths = collect_default_eval_paths()

    if not eval_paths:
        raise FileNotFoundError("No eval paths found for benchmark suite.")

    eval_summaries = []

    for eval_path in eval_paths:
        eval_output_dir = (
            output_dir
            / "eval_runs"
            / f"{checkpoint_path.stem}_{timestamp}"
            / eval_path.stem
        )
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        run_command([
            sys.executable,
            str(ROOT_DIR / "scripts" / "eval_model.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--eval_path",
            str(eval_path),
            "--output_dir",
            str(eval_output_dir),
        ])

        eval_run_paths = sorted(eval_output_dir.glob("*.jsonl"))

        if not eval_run_paths:
            raise FileNotFoundError(f"No eval run JSONL written to {eval_output_dir}")

        eval_summaries.append(summarize_eval_run(eval_path, eval_run_paths[-1]))

    total = sum(item["total"] for item in eval_summaries)
    passed = sum(item["passed"] for item in eval_summaries)
    failed = total - passed
    overall_pass_rate = (passed / total) if total else 0.0

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": display_path(checkpoint_path),
        "total": total,
        "passed": passed,
        "failed": failed,
        "overall_pass_rate": overall_pass_rate,
        "evals": eval_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"benchmark_{checkpoint_path.stem}_{timestamp}.json"
    markdown_path = output_dir / f"benchmark_{checkpoint_path.stem}_{timestamp}.md"
    write_json_summary(json_path, summary)
    write_markdown_summary(markdown_path, summary)

    print("=" * 70)
    print(f"Benchmark JSON: {json_path}")
    print(f"Benchmark Markdown: {markdown_path}")
    print(f"Overall weighted pass rate: {overall_pass_rate:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
