from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT_DIR / "darkmind_distill" / "prompts" / "eval_prompts.md"
GENERATE_SCRIPT = ROOT_DIR / "scripts" / "generate_from_checkpoint.py"


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    language = "unknown"
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("## "):
                language = stripped[3:].strip().lower()
                continue
            if stripped.startswith("- "):
                prompts.append({"language": language, "prompt": stripped[2:].strip()})
    return prompts


def extract_answer(stdout: str) -> str:
    marker = "ASSISTANT ANSWER 1:"
    if marker not in stdout:
        return stdout.strip()
    answer = stdout.split(marker, 1)[1]
    if "=" * 20 in answer:
        answer = answer.split("=" * 20, 1)[0]
    return answer.strip()


def repeated_text(text: str) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 9:
        return False
    grams = Counter(tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2))
    return any(count >= 3 for count in grams.values())


def flags_for(prompt: str, language: str, output: str) -> dict[str, bool]:
    lower_prompt = prompt.lower()
    lower_output = output.lower()
    identity_prompt = any(word in lower_prompt for word in ["kimsin", "chatgpt", "openai", "qwen", "who are you", "bist du", "eres", "es-tu"])
    identity_ok = (not identity_prompt) or ("darkmind" in lower_output and "chatgpt" not in lower_output.replace("chatgpt değil", ""))
    identity_leak = (not identity_prompt) and ("ben darkmind" in lower_output or lower_output.startswith("darkmind"))
    empty_output = len(output.strip()) < 10
    language_mismatch = False
    if language == "turkish" and re.search(r"\b(the|and|what is|docker is)\b", lower_output):
        language_mismatch = True
    if language == "english" and re.search(r"\b(nedir|değildir|geliştirilen)\b", lower_output):
        language_mismatch = True
    return {
        "identity_ok": identity_ok,
        "identity_leak": identity_leak,
        "empty_output": empty_output,
        "language_mismatch": language_mismatch,
        "repeated_text": repeated_text(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DarkMind student outputs on fixed multilingual prompts.")
    parser.add_argument("--checkpoint", type=str, default="models/darkmind-30m-qwen-distill-v0.1.pt")
    parser.add_argument("--out", type=str, default="darkmind_distill/reports/qwen_distill_v0_1_eval.md")
    args = parser.parse_args()

    checkpoint = ROOT_DIR / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    out_path = ROOT_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)
    prompts = load_prompts(PROMPTS_PATH)
    rows: list[dict[str, object]] = []

    for row in prompts:
        cmd = [
            sys.executable,
            str(GENERATE_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            row["prompt"],
            "--chat_format",
            "--max_new_tokens",
            "120",
            "--temperature",
            "0.3",
            "--top_k",
            "40",
            "--top_p",
            "0.9",
            "--repetition_penalty",
            "1.15",
            "--no_repeat_ngram_size",
            "3",
        ]
        result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        output = extract_answer(result.stdout)
        rows.append({
            "language": row["language"],
            "prompt": row["prompt"],
            "output": output,
            "returncode": result.returncode,
            "flags": flags_for(row["prompt"], row["language"], output),
        })

    lines = ["# DarkMind Qwen Distill Eval", ""]
    for index, row in enumerate(rows, start=1):
        flags = ", ".join(f"{key}={value}" for key, value in row["flags"].items())
        lines.extend([
            f"## {index}. {row['language']}",
            "",
            f"**Prompt:** {row['prompt']}",
            "",
            f"**Flags:** `{flags}`",
            "",
            "```text",
            str(row["output"]),
            "```",
            "",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Eval report written: {out_path}")


if __name__ == "__main__":
    main()
