from collections import Counter
from pathlib import Path
import argparse
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT_DIR / "data" / "self_improvement" / "runs"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_rows(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

    return rows


def summarize(path: Path) -> dict:
    rows = load_rows(path)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed", False))
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0
    failing_categories = Counter(
        row.get("category", "unknown")
        for row in rows
        if not row.get("passed", False)
    )

    return {
        "filename": path.name,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "failing_categories": failing_categories,
    }


def format_categories(counter: Counter) -> str:
    if not counter:
        return "-"

    return ", ".join(
        f"{category}:{count}"
        for category, count in sorted(counter.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latest DarkMind eval runs.")
    parser.add_argument(
        "--runs_dir",
        default=str(DEFAULT_RUNS_DIR.relative_to(ROOT_DIR)),
        help="Directory containing eval_run_*.jsonl files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of latest runs to show.",
    )
    args = parser.parse_args()

    runs_dir = resolve_path(args.runs_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_files = sorted(runs_dir.glob("eval_run_*.jsonl"), key=lambda path: path.stat().st_mtime)
    run_files = run_files[-args.limit:]

    print("=" * 70)
    print("DarkMind Latest Eval Summary")
    print("=" * 70)

    if not run_files:
        print(f"No eval_run_*.jsonl files found in {runs_dir}")
        print("=" * 70)
        return

    print("filename | total | passed | failed | pass rate | categories failing")

    for path in run_files:
        row = summarize(path)
        print(
            f"{row['filename']} | "
            f"{row['total']} | "
            f"{row['passed']} | "
            f"{row['failed']} | "
            f"{row['pass_rate']:.2f}% | "
            f"{format_categories(row['failing_categories'])}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
