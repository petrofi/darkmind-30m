from pathlib import Path
import argparse
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_EVAL_PATH = ROOT_DIR / "data" / "evals" / "darkmind_code_eval_v01.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from eval_model import DEFAULT_OUTPUT_DIR, resolve_path, run_evaluation
from generate_correction_candidates import default_output_path, generate_candidates


def print_next_steps(candidate_file: Path) -> None:
    print("=" * 70)
    print("Manual coding eval next steps")
    print("=" * 70)
    print("1. Review pending coding correction candidates:")
    print(f"   {candidate_file}")
    print()
    print("2. If useful, approve manually after review:")
    print(
        "   python scripts/approve_candidates.py "
        f"--input_path {candidate_file} --approve_all"
    )
    print()
    print("3. Rebuild only after approval:")
    print("   python scripts/clean_text.py")
    print("   python scripts/build_dataset_from_raw.py")
    print("   python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt")
    print("=" * 70)
    print("This script did not approve, rebuild, tokenize, or train.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Turkish coding eval and generate pending corrections."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Example: checkpoints/darkmind_30m_1000step.pt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for eval generation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k sampling value for eval generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Maximum number of new tokens per eval answer.",
    )
    args = parser.parse_args()

    run_path = run_evaluation(
        checkpoint_path=resolve_path(args.checkpoint),
        eval_path=CODE_EVAL_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    candidate_file = generate_candidates(
        run_path=run_path,
        output_path=default_output_path(),
    )

    print_next_steps(candidate_file)


if __name__ == "__main__":
    main()
