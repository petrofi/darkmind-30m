from pathlib import Path
import shutil
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
CORPUS_V3_PATH = ROOT_DIR / "data" / "processed" / "corpus_v3.txt"
CURRICULUM_PATH = ROOT_DIR / "data" / "processed" / "corpus_curriculum_v01.txt"


def run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *args]

    print("=" * 70)
    print(" ".join(command))
    print("=" * 70)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def script_exists(script_name: str) -> bool:
    return (SCRIPT_DIR / script_name).exists()


def ensure_curriculum_corpus_from_v3() -> None:
    if not CORPUS_V3_PATH.exists():
        raise FileNotFoundError(f"Fallback corpus not found: {CORPUS_V3_PATH}")

    CURRICULUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CORPUS_V3_PATH, CURRICULUM_PATH)
    print("=" * 70)
    print(
        "Created curriculum corpus fallback from corpus_v3: "
        f"{CURRICULUM_PATH}"
    )
    print("=" * 70)


def main() -> None:
    run_script("generate_turkish_code_data_factory.py")
    run_script("validate_code_dataset_quality.py", "--strict")
    run_script("generate_code_unit_eval.py")
    run_script("clean_text.py")

    if script_exists("score_dataset_examples.py"):
        run_script("score_dataset_examples.py")
    else:
        print("score_dataset_examples.py not found; skipped.")

    if script_exists("build_curriculum_dataset.py"):
        run_script("build_curriculum_dataset.py")
    else:
        print("build_curriculum_dataset.py not found; using build_dataset_from_raw.py.")
        run_script("build_dataset_from_raw.py")
        ensure_curriculum_corpus_from_v3()

    run_script("dataset_quality_check.py", "--path", str(CURRICULUM_PATH))

    print("=" * 70)
    print("Code data preparation cycle completed.")
    print("Next commands:")
    print("python scripts/train_tokenizer.py --data_path data/processed/corpus_curriculum_v01.txt")
    print("python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_curriculum_v01.txt")
    print("python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m_1000step.pt --eval_path data/evals/darkmind_code_eval_v02.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
