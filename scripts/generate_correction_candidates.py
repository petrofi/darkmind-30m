from datetime import datetime
from pathlib import Path
import argparse
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "data" / "self_improvement" / "runs"
PENDING_REVIEW_DIR = ROOT_DIR / "data" / "self_improvement" / "pending_review"


CATEGORY_CORRECTIONS = {
    "identity": "Ben DarkMind. Türkçe odaklı küçük bir dil modeli geliştirme projesiyim ve hâlâ geliştirme aşamasındayım.",
    "fallback": "Bu konuda yeterli bilgiye sahip olmayabilirim. Emin değilsem uydurmak yerine sınırlı olduğumu söylemem daha doğru olur.",
    "tokenizer": "Tokenizer, metni modelin işleyebileceği tokenlara ve token ID'lerine dönüştüren bileşendir.",
    "cuda": "CUDA, NVIDIA GPU üzerinde derin öğrenme işlemlerini hızlandırmak için kullanılan altyapıdır.",
    "overfitting": "Overfitting, modelin eğitim verisini ezberleyip yeni örneklerde zayıf genelleme yapmasıdır.",
    "checkpoint": "Checkpoint, model ağırlıklarını, config bilgisini ve eğitim durumunu daha sonra kullanmak için kaydeden dosyadır.",
    "python_basic": "Python'da küçük ve doğru örneklerle ilerlemek gerekir; fonksiyonlar def ile tanımlanır, koşullar if/else ile yazılır ve listeler üzerinde döngü kurulabilir.",
    "data_pipeline": "Data pipeline; ham veriyi toplama, temizleme, corpus oluşturma, tokenizer eğitme ve modeli eğitime hazırlama sürecidir.",
    "coding_errors": "Kod hatalarında önce hata mesajını okumalı, aktif sanal ortamı, doğru Python yorumlayıcısını, paketleri ve dosya yolunu kontrol etmelisin.",
    "chat_basics": "Merhaba. Ben DarkMind. Sınırlı bir Türkçe demo modeliyim ve özellikle yazılım, yapay zeka ve proje geliştirme konularında yardımcı olmaya çalışırım.",
}


HEADER = (
    "These are candidate examples. Review before adding to training data.\n"
    "Do not approve blindly. Keep only examples that are correct, useful, and safe."
)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def latest_eval_run() -> Path:
    run_files = sorted(RUNS_DIR.glob("eval_run_*.jsonl"))

    if not run_files:
        raise FileNotFoundError(f"No eval_run_*.jsonl files found in {RUNS_DIR}")

    return run_files[-1]


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PENDING_REVIEW_DIR / f"correction_candidates_{timestamp}.txt"


def load_run(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

    return rows


def correction_for_category(category: str) -> str:
    return CATEGORY_CORRECTIONS.get(
        category,
        "Bu konuda kısa, güvenli ve emin olunan bir cevap vermeliyim; emin değilsem sınırlı olduğumu söylemeliyim.",
    )


def render_candidate(prompt: str, answer: str) -> str:
    return f"Kullanıcı: {prompt}\nAsistan: {answer}"


def generate_candidates(run_path: Path, output_path: Path) -> Path:
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    rows = load_run(run_path)
    failed_rows = [row for row in rows if not row.get("passed", False)]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        HEADER,
        f"Source eval run: {run_path}",
        f"Failed items: {len(failed_rows)}",
    ]

    for row in failed_rows:
        answer = correction_for_category(row.get("category", ""))
        parts.append(render_candidate(row["prompt"], answer))

    if not failed_rows:
        parts.append("No failed eval items found. No correction examples generated.")

    output_text = "\n\n".join(parts).strip() + "\n"
    output_path.write_text(output_text, encoding="utf-8")

    print("=" * 70)
    print("DarkMind Correction Candidates")
    print("=" * 70)
    print(f"Run file: {run_path}")
    print(f"Failed items: {len(failed_rows)}")
    print(f"Output file: {output_path}")
    print("Review this file before approving anything.")
    print("=" * 70)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pending-review correction examples from failed evals."
    )
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="Eval run JSONL path. Defaults to the latest eval_run_*.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Pending review candidate output path.",
    )
    args = parser.parse_args()

    run_path = resolve_path(args.run_path) if args.run_path else latest_eval_run()
    output_path = (
        resolve_path(args.output_path)
        if args.output_path
        else default_output_path()
    )

    generate_candidates(run_path, output_path)


if __name__ == "__main__":
    main()
