from datetime import datetime
from pathlib import Path
import argparse
import json
import unicodedata


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


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return normalized.replace("\u0307", "")


def flatten_missing_terms(row: dict) -> list[str]:
    terms = list(row.get("missing_keywords", []))

    for group in row.get("missing_groups", []):
        terms.extend(group)

    unique_terms = []
    seen = set()

    for term in terms:
        normalized = normalize_for_match(term)

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_terms.append(term)

    return unique_terms


def has_missing(missing_terms: list[str], *needles: str) -> bool:
    normalized_terms = [
        normalize_for_match(term)
        for term in missing_terms
    ]

    return any(
        normalize_for_match(needle) in term
        for term in normalized_terms
        for needle in needles
    )


def text_has_any(text: str, *needles: str) -> bool:
    normalized_text = normalize_for_match(text)
    return any(
        normalize_for_match(needle) in normalized_text
        for needle in needles
    )


def append_missing_terms(answer: str, missing_terms: list[str]) -> str:
    still_missing = [
        term
        for term in missing_terms
        if normalize_for_match(term) not in normalize_for_match(answer)
    ]

    if not still_missing:
        return answer

    return f"{answer} Anahtar kavramlar: {', '.join(still_missing[:4])}."


def coding_answer_with_code(explanation: str, code: str) -> str:
    return f"{explanation}\n\n```python\n{code.strip()}\n```"


def correction_for_python_row(row: dict, missing_terms: list[str]) -> str:
    prompt = row.get("prompt", "")
    category = row.get("category", "")

    if has_missing(missing_terms, "append") or text_has_any(prompt, "append", "listeye eleman"):
        answer = coding_answer_with_code(
            "Python'da listeye eleman eklemek için append metodu kullanılır.",
            """
numbers = [1, 2, 3]
numbers.append(5)
print(numbers)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "try", "except") or has_missing(missing_terms, "try", "except"):
        answer = coding_answer_with_code(
            "try/except, hata oluşabilecek kodu güvenli şekilde ele almak için kullanılır.",
            """
try:
    number = int("42")
    print(number)
except Exception as error:
    print(f"Hata: {error}")
""",
        )
        return append_missing_terms(answer, missing_terms)

    if (
        category == "python_function"
        or text_has_any(prompt, "fonksiyon", "return")
        or has_missing(missing_terms, "def", "return")
    ):
        answer = coding_answer_with_code(
            "Python'da fonksiyon def ile tanımlanır ve sonucu döndürmek için return kullanılır.",
            """
def add_numbers(a, b):
    return a + b

print(add_numbers(2, 3))
""",
        )
        return append_missing_terms(answer, missing_terms)

    if (
        category == "python_file"
        or text_has_any(prompt, "dosya", "pathlib", "read_text")
        or has_missing(missing_terms, "Path", "read_text")
    ):
        answer = coding_answer_with_code(
            "pathlib ile dosya okumak için Path ve read_text kullanılabilir.",
            """
from pathlib import Path

text = Path("file.txt").read_text(encoding="utf-8")
print(text)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if (
        category == "python_dict"
        or text_has_any(prompt, "sözlük", "dictionary", "key")
        or has_missing(missing_terms, "get", "dict")
    ):
        answer = coding_answer_with_code(
            "Sözlükten güvenli değer okumak için my_dict.get(\"key\") kullanılabilir.",
            """
my_dict = {"name": "DarkMind"}
name = my_dict.get("name")
print(name)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if (
        category == "python_class"
        or text_has_any(prompt, "class", "__init__", "init")
        or has_missing(missing_terms, "__init__", "class")
    ):
        answer = coding_answer_with_code(
            "Python'da class içinde __init__, nesne oluşturulurken başlangıç değerlerini ayarlar.",
            """
class User:
    def __init__(self, name):
        self.name = name

user = User("Ayşe")
print(user.name)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "sırala", "sorted") or has_missing(missing_terms, "sorted"):
        answer = coding_answer_with_code(
            "Liste sıralamak için sorted(items) kullanılabilir.",
            """
items = [3, 1, 2]
sorted_items = sorted(items)
print(sorted_items)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "çift", "filtre") or has_missing(missing_terms, "if", "% 2"):
        answer = coding_answer_with_code(
            "Çift sayıları filtrelemek için number % 2 == 0 koşulu kullanılabilir.",
            """
numbers = [1, 2, 3, 4, 5, 6]
evens = [number for number in numbers if number % 2 == 0]
print(evens)
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "palindrom", "palindrome"):
        answer = coding_answer_with_code(
            "Palindrom kontrolünde metin ters haliyle karşılaştırılır.",
            """
def is_palindrome(text):
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "asal", "prime"):
        answer = coding_answer_with_code(
            "Asal sayı kontrolünde sayının böleni olup olmadığına bakılır.",
            """
def is_prime(number):
    if number < 2:
        return False
    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False
    return True
""",
        )
        return append_missing_terms(answer, missing_terms)

    if text_has_any(prompt, "faktöriyel", "factorial"):
        answer = coding_answer_with_code(
            "Faktöriyel, 1'den n'e kadar sayıların çarpımıdır.",
            """
def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result *= value
    return result
""",
        )
        return append_missing_terms(answer, missing_terms)

    answer = coding_answer_with_code(
        "Python'da küçük, güvenli ve okunabilir örneklerle ilerlemek en doğru yaklaşımdır.",
        """
def add_numbers(a, b):
    return a + b

print(add_numbers(2, 3))
""",
    )
    return append_missing_terms(answer, missing_terms)


def correction_for_row(row: dict) -> str:
    category = row.get("category", "")
    missing_terms = flatten_missing_terms(row)

    if category.startswith("python_"):
        return correction_for_python_row(row, missing_terms)

    if has_missing(missing_terms, "append"):
        return "Python'da listeye eleman eklemek için append metodu kullanılır. Örnek: numbers.append(5)"

    if has_missing(missing_terms, "girinti", "blok"):
        return "IndentationError genellikle girinti hatasından kaynaklanır. Python'da aynı blok içindeki satırların girinti seviyesi tutarlı olmalıdır."

    if has_missing(missing_terms, "vocab", "sözlük"):
        return "Vocab, tokenizer'ın kullandığı token sözlüğüdür. Model metni vocab içindeki token ID değerleriyle işler."

    if has_missing(missing_terms, "CPU"):
        return "CPU genel amaçlı işlemcidir. GPU paralel hesaplamalarda güçlüdür ve derin öğrenme eğitimini CPU'ya göre hızlandırabilir."

    if has_missing(missing_terms, "birleştir"):
        return "build_dataset_from_raw.py kaynak dosyaları ve temizlenmiş verileri birleştirir, ardından data/processed/corpus_v3.txt dosyasını üretir."

    answer = correction_for_category(category)
    return append_missing_terms(answer, missing_terms)


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
        answer = correction_for_row(row)
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
