from collections import Counter
from pathlib import Path
import argparse
import json


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_eval_run(path: Path) -> dict[str, dict]:
    rows = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            rows[row["id"]] = row

    return rows


def pass_rate(rows: dict[str, dict]) -> tuple[int, int, int, float]:
    total = len(rows)
    passed = sum(1 for row in rows.values() if row.get("passed", False))
    failed = total - passed
    rate = (passed / total * 100) if total else 0.0
    return total, passed, failed, rate


def category_summary(rows: dict[str, dict]) -> dict[str, tuple[int, int, int, float]]:
    categories = sorted({row.get("category", "unknown") for row in rows.values()})
    summary = {}

    for category in categories:
        category_rows = {
            row_id: row
            for row_id, row in rows.items()
            if row.get("category", "unknown") == category
        }
        summary[category] = pass_rate(category_rows)

    return summary


def print_id_list(title: str, ids: list[str]) -> None:
    print(title)

    if not ids:
        print("- none")
        return

    for item_id in ids:
        print(f"- {item_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DarkMind eval run JSONL files.")
    parser.add_argument("--before", required=True, help="Earlier eval_run_*.jsonl path.")
    parser.add_argument("--after", required=True, help="Later eval_run_*.jsonl path.")
    args = parser.parse_args()

    before_path = resolve_path(args.before)
    after_path = resolve_path(args.after)
    before = load_eval_run(before_path)
    after = load_eval_run(after_path)

    before_total, before_passed, before_failed, before_rate = pass_rate(before)
    after_total, after_passed, after_failed, after_rate = pass_rate(after)
    delta = after_rate - before_rate

    common_ids = sorted(set(before) & set(after))
    newly_passed = [
        item_id
        for item_id in common_ids
        if not before[item_id].get("passed", False) and after[item_id].get("passed", False)
    ]
    newly_failed = [
        item_id
        for item_id in common_ids
        if before[item_id].get("passed", False) and not after[item_id].get("passed", False)
    ]
    still_failing = [
        item_id
        for item_id in common_ids
        if not before[item_id].get("passed", False) and not after[item_id].get("passed", False)
    ]

    print("=" * 70)
    print("DarkMind Eval Run Comparison")
    print("=" * 70)
    print(f"Before: {before_path}")
    print(f"After:  {after_path}")
    print(f"Before pass rate: {before_passed}/{before_total} ({before_rate:.2f}%)")
    print(f"After pass rate:  {after_passed}/{after_total} ({after_rate:.2f}%)")
    print(f"Delta: {delta:+.2f} percentage points")
    print(f"Before failed: {before_failed}")
    print(f"After failed: {after_failed}")
    print("-" * 70)
    print_id_list("Newly passed IDs:", newly_passed)
    print("-" * 70)
    print_id_list("Newly failed IDs:", newly_failed)
    print("-" * 70)
    print_id_list("Still failing IDs:", still_failing)

    before_categories = category_summary(before)
    after_categories = category_summary(after)
    all_categories = sorted(set(before_categories) | set(after_categories))

    print("-" * 70)
    print("Category summary:")
    print("category | before | after | delta")

    for category in all_categories:
        before_stats = before_categories.get(category, (0, 0, 0, 0.0))
        after_stats = after_categories.get(category, (0, 0, 0, 0.0))
        category_delta = after_stats[3] - before_stats[3]
        print(
            f"{category} | "
            f"{before_stats[1]}/{before_stats[0]} ({before_stats[3]:.2f}%) | "
            f"{after_stats[1]}/{after_stats[0]} ({after_stats[3]:.2f}%) | "
            f"{category_delta:+.2f}"
        )

    new_only = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))

    if new_only or removed:
        print("-" * 70)
        print(f"IDs only in after: {len(new_only)}")
        print(f"IDs only in before: {len(removed)}")

    category_failures = Counter(row.get("category", "unknown") for row in after.values() if not row.get("passed", False))

    if category_failures:
        print("-" * 70)
        print("After failures by category:")
        for category, count in sorted(category_failures.items()):
            print(f"- {category}: {count}")

    print("=" * 70)


if __name__ == "__main__":
    main()
