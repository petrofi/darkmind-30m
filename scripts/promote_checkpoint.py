from pathlib import Path
import argparse
from datetime import datetime
import json
import shutil
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
PROMOTED_DIR = ROOT_DIR / "checkpoints" / "promoted"


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


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def compute_pass_rate(eval_run_path: Path) -> tuple[int, int, float]:
    rows = load_jsonl(eval_run_path)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    pass_rate = (passed / total) if total else 0.0

    return total, passed, pass_rate


def write_metadata(
    path: Path,
    candidate_checkpoint: Path,
    eval_run: Path,
    min_pass_rate: float,
    total: int,
    passed: int,
    pass_rate: float,
) -> None:
    metadata = {
        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_checkpoint": display_path(candidate_checkpoint),
        "eval_run": display_path(eval_run),
        "min_pass_rate": min_pass_rate,
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a checkpoint only when an eval run passes a threshold."
    )
    parser.add_argument("--candidate_checkpoint", type=str, required=True)
    parser.add_argument("--eval_run", type=str, required=True)
    parser.add_argument("--min_pass_rate", type=float, default=0.70)
    parser.add_argument(
        "--promoted_name",
        type=str,
        default="darkmind_current_best.pt",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata file to copy alongside the promoted checkpoint.",
    )
    args = parser.parse_args()

    candidate_checkpoint = resolve_path(args.candidate_checkpoint)
    eval_run = resolve_path(args.eval_run)

    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")

    if not eval_run.exists():
        raise FileNotFoundError(f"Eval run not found: {eval_run}")

    total, passed, pass_rate = compute_pass_rate(eval_run)

    print("=" * 70)
    print(f"Eval run: {eval_run}")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Pass rate: {pass_rate:.2%}")
    print(f"Required pass rate: {args.min_pass_rate:.2%}")

    if total == 0 or pass_rate < args.min_pass_rate:
        print("-" * 70)
        print("Promotion refused. The candidate did not meet the threshold.")
        print("=" * 70)
        return 1

    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    promoted_path = PROMOTED_DIR / args.promoted_name
    shutil.copy2(candidate_checkpoint, promoted_path)

    if args.metadata:
        metadata_path = resolve_path(args.metadata)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        copied_metadata_path = (
            PROMOTED_DIR
            / f"{promoted_path.stem}_metadata{metadata_path.suffix}"
        )
        shutil.copy2(metadata_path, copied_metadata_path)
    else:
        copied_metadata_path = PROMOTED_DIR / f"{promoted_path.stem}_metadata.json"
        write_metadata(
            copied_metadata_path,
            candidate_checkpoint,
            eval_run,
            args.min_pass_rate,
            total,
            passed,
            pass_rate,
        )

    print("-" * 70)
    print(f"Promoted checkpoint: {promoted_path}")
    print(f"Promotion metadata: {copied_metadata_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
