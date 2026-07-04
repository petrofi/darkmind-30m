from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
CURRICULUM_CORPUS = ROOT_DIR / "data" / "processed" / "corpus_curriculum_v01.txt"


def run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *args]

    print("=" * 70)
    print(" ".join(command))
    print("=" * 70)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def main() -> None:
    run_script("clean_text.py")
    run_script("score_dataset_examples.py")
    run_script("build_curriculum_dataset.py")
    run_script("dataset_quality_check.py", "--path", str(CURRICULUM_CORPUS))

    print("=" * 70)
    print("Automated dataset preparation completed.")
    print("Next commands:")
    print("python scripts/train_tokenizer.py --data_path data/processed/corpus_curriculum_v01.txt")
    print("python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_curriculum_v01.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
