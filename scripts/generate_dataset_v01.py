from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from generate_coding_error_examples import main as generate_coding_error_examples
from generate_python_examples import main as generate_python_examples
from generate_qa_variants import main as generate_qa_variants


NEXT_COMMANDS = [
    "python scripts/clean_text.py",
    "python scripts/build_dataset_from_raw.py",
    "python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt",
    "python scripts/train_tokenizer.py --data_path data/processed/corpus_v3.txt",
    "python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_v3.txt",
]


def main() -> None:
    print("=" * 70)
    print("DarkMind deterministic dataset generation v01")
    print("=" * 70)

    generate_python_examples()
    generate_coding_error_examples()
    generate_qa_variants()

    print("=" * 70)
    print("Next recommended commands:")

    for command in NEXT_COMMANDS:
        print(command)

    print("=" * 70)
    print(f"Repository: {ROOT_DIR}")


if __name__ == "__main__":
    main()
