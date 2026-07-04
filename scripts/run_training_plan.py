from pathlib import Path
import argparse
from datetime import datetime
import json
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "corpus_curriculum_v01.txt"
DEFAULT_EVAL_PATHS = [
    ROOT_DIR / "data" / "evals" / "darkmind_eval_v02.jsonl",
    ROOT_DIR / "data" / "evals" / "darkmind_code_eval_v02.jsonl",
]
REPORT_DIR = ROOT_DIR / "reports" / "training"


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str]) -> None:
    print("=" * 70)
    print(command_text(command))
    print("=" * 70)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def summarize_eval_run(path: Path) -> dict:
    rows = load_jsonl(path)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0

    return {
        "path": path,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
    }


def collect_default_eval_paths() -> list[Path]:
    return [path for path in DEFAULT_EVAL_PATHS if path.exists()]


def determine_checkpoint_path(
    config: dict,
    run_name: str,
    checkpoint_override: str | None,
) -> Path:
    if checkpoint_override:
        return resolve_path(checkpoint_override)

    if config.get("checkpoint_path"):
        return resolve_path(config["checkpoint_path"])

    best_path = ROOT_DIR / "checkpoints" / f"{run_name}_best.pt"

    if best_path.exists():
        return best_path

    return ROOT_DIR / "checkpoints" / f"{run_name}.pt"


def write_report(
    path: Path,
    run_name: str,
    config_path: Path,
    data_path: Path,
    checkpoint_path: Path,
    skip_training: bool,
    eval_summaries: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Training Plan Report - {run_name}",
        "",
        f"- Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Config: `{display_path(config_path)}`",
        f"- Data path: `{display_path(data_path)}`",
        f"- Checkpoint: `{display_path(checkpoint_path)}`",
        f"- Skip training: `{skip_training}`",
        "",
        "## Eval Results",
        "",
    ]

    if not eval_summaries:
        lines.append("No evals were run.")
    else:
        lines.append("| Eval run | Total | Passed | Failed | Pass rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")

        for summary in eval_summaries:
            lines.append(
                f"| `{display_path(summary['path'])}` | "
                f"{summary['total']} | {summary['passed']} | "
                f"{summary['failed']} | {summary['pass_rate']:.2%} |"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "This report records commands and eval outputs only. It does not claim model improvement unless the eval results support it.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a selected DarkMind config, then run evals and write a report."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH.relative_to(ROOT_DIR)),
        help="Training corpus path passed to tokenizer and training.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        action="append",
        default=None,
        help="Eval JSONL path. Can be repeated.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip tokenizer/model training and only run evals.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint override used for eval.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    data_path = resolve_path(args.data_path)
    config = load_json(config_path)
    run_name = config["run_name"]
    timestamp = timestamp_for_filename()

    if args.eval_path:
        eval_paths = [resolve_path(path) for path in args.eval_path]
    else:
        eval_paths = collect_default_eval_paths()

    if not args.skip_training:
        run_command([
            sys.executable,
            str(ROOT_DIR / "scripts" / "train_tokenizer.py"),
            "--data_path",
            str(data_path),
        ])
        run_command([
            sys.executable,
            str(ROOT_DIR / "scripts" / "train_from_config.py"),
            "--config",
            str(config_path),
            "--data_path",
            str(data_path),
        ])

    checkpoint_path = determine_checkpoint_path(
        config,
        run_name,
        args.checkpoint,
    )

    eval_summaries = []

    for eval_path in eval_paths:
        eval_output_dir = (
            REPORT_DIR
            / "eval_runs"
            / f"{run_name}_{timestamp}"
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

        eval_summaries.append(summarize_eval_run(eval_run_paths[-1]))

    report_path = REPORT_DIR / f"training_plan_{run_name}_{timestamp}.md"
    write_report(
        report_path,
        run_name,
        config_path,
        data_path,
        checkpoint_path,
        args.skip_training,
        eval_summaries,
    )

    print("=" * 70)
    print(f"Training plan report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
