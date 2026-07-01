from datetime import datetime, timezone
from pathlib import Path
import argparse
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "experiment_registry.jsonl"


def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a DarkMind experiment registry JSONL record."
    )
    parser.add_argument("--experiment_id", required=True, help="Experiment id, e.g. exp_012.")
    parser.add_argument("--name", required=True, help="Human-readable experiment name.")
    parser.add_argument(
        "--dataset",
        default="data/processed/corpus_v3.txt",
        help="Dataset path used by the experiment.",
    )
    parser.add_argument(
        "--config",
        default="configs/darkmind_30m_1000step.json",
        help="Training config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/darkmind_30m_1000step.pt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--eval_path",
        action="append",
        default=[],
        help="Evaluation JSONL path. Can be repeated.",
    )
    parser.add_argument("--notes", default="", help="Optional notes.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(ROOT_DIR)),
        help="Experiment registry JSONL output path.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc).astimezone()
    record = {
        "experiment_id": args.experiment_id,
        "name": args.name,
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "evals": args.eval_path,
        "notes": args.notes,
    }

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("Experiment logged.")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Experiment: {args.experiment_id} - {args.name}")


if __name__ == "__main__":
    main()
