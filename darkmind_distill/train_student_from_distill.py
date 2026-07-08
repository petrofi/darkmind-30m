from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "darkmind_distill" / "data" / "darkmind_qwen_distill_v0_1.jsonl"
BASE_CHECKPOINT = ROOT_DIR / "models" / "darkmind-30m-10k-step15000.pt"
SAVE_PATH = ROOT_DIR / "models" / "darkmind-30m-qwen-distill-v0.1.pt"
INSPECT_SCRIPT = ROOT_DIR / "darkmind_distill" / "inspect_distill_dataset.py"
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_instruct_jsonl.py"


def training_command() -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data",
        str(DATA_PATH),
        "--base_checkpoint",
        str(BASE_CHECKPOINT),
        "--save_path",
        str(SAVE_PATH),
        "--epochs",
        "2",
        "--batch_size",
        "4",
        "--block_size",
        "256",
        "--max_steps",
        "1200",
        "--lr",
        "0.00002",
        "--val_ratio",
        "0.1",
        "--eval_interval",
        "100",
        "--eval_batches",
        "20",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DarkMind student from inspected Qwen distillation data.")
    parser.add_argument("--confirm_train", action="store_true")
    args = parser.parse_args()

    inspect_cmd = [sys.executable, str(INSPECT_SCRIPT)]
    print("Running dataset inspection first...")
    inspection = subprocess.run(inspect_cmd, cwd=ROOT_DIR)
    if inspection.returncode != 0:
        raise SystemExit("Inspection failed. Training will not start.")

    cmd = training_command()
    print("=" * 70)
    print("Training command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    print("=" * 70)

    if not args.confirm_train:
        print("Pass --confirm_train to actually start training.")
        return

    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
