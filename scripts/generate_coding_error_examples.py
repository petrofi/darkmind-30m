from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "coding_notes"
    / "coding_error_examples_v01.txt"
)


ERROR_TOPICS = [
    {
        "error": "ModuleNotFoundError: No module named 'torch'",
        "reason": "aktif Python ortamında torch paketinin kurulu olmadığını",
        "fix": "Sanal ortamın aktif olduğundan emin olmalı ve paketi doğru ortamda kurmalısın.",
    },
    {
        "error": "ImportError: cannot import name 'GPTLanguageModel'",
        "reason": "import edilen ismin dosyada bulunmadığını veya yanlış yerden import edildiğini",
        "fix": "Sınıf adını, dosya yolunu ve sys.path ayarını kontrol etmelisin.",
    },
    {
        "error": "SyntaxError: invalid syntax",
        "reason": "Python kodunda yazım kuralı hatası olduğunu",
        "fix": "Hatanın gösterdiği satırdaki parantez, iki nokta ve tırnak işaretlerini kontrol etmelisin.",
    },
    {
        "error": "IndentationError: expected an indented block",
        "reason": "if, for, while veya def satırından sonra girinti beklediğini",
        "fix": "Blok içindeki satırları aynı girinti seviyesiyle yazmalısın.",
    },
    {
        "error": "NameError: name 'tokenizer' is not defined",
        "reason": "tokenizer değişkeninin kullanılmadan önce tanımlanmadığını",
        "fix": "Değişkeni önce oluşturmalı veya doğru isimle çağırmalısın.",
    },
    {
        "error": "TypeError: unsupported operand type(s)",
        "reason": "uyumsuz veri tipleriyle işlem yapılmaya çalışıldığını",
        "fix": "Değerlerin tipini kontrol etmeli ve gerekirse int, float veya str dönüşümü yapmalısın.",
    },
    {
        "error": "ValueError: invalid literal for int()",
        "reason": "sayıya çevrilemeyen bir metnin int() içine verildiğini",
        "fix": "Girdiyi doğrulamalı veya dönüşümü try/except ile güvenli hale getirmelisin.",
    },
    {
        "error": "FileNotFoundError: No such file or directory",
        "reason": "okunmak istenen dosyanın belirtilen konumda bulunmadığını",
        "fix": "Dosya yolunu Path ile kontrol etmeli ve çalışma dizinini doğrulamalısın.",
    },
    {
        "error": "PermissionError: permission denied",
        "reason": "dosya veya klasöre erişim izni olmadığını",
        "fix": "Dosyanın başka programda açık olmadığını ve yazma izninin bulunduğunu kontrol etmelisin.",
    },
    {
        "error": "IndexError: list index out of range",
        "reason": "listenin olmayan bir indeksine erişilmeye çalışıldığını",
        "fix": "İndeksi kullanmadan önce listenin uzunluğunu kontrol etmelisin.",
    },
    {
        "error": "KeyError: 'config'",
        "reason": "sözlükte olmayan bir anahtarın okunmaya çalışıldığını",
        "fix": "Anahtar adını kontrol etmeli veya get() ile varsayılan değer kullanmalısın.",
    },
    {
        "error": "AttributeError: object has no attribute 'generate'",
        "reason": "nesnede çağrılan özellik veya metodun bulunmadığını",
        "fix": "Nesnenin tipini ve kullandığın metod adını kontrol etmelisin.",
    },
    {
        "error": "fatal: not a git repository",
        "reason": "komutun Git deposu olmayan bir klasörde çalıştırıldığını",
        "fix": "Önce proje kök dizinine gitmeli veya doğru klasörde git init yapılmış mı bakmalısın.",
    },
    {
        "error": "git push rejected",
        "reason": "uzak depoda yerelde olmayan commitler bulunduğunu veya branch koruması olduğunu",
        "fix": "Önce remote değişiklikleri anlamalı, gerekirse pull veya rebase yapmalı ve çakışmaları çözmelisin.",
    },
    {
        "error": "venv aktif değil",
        "reason": "komutların beklenen sanal ortam yerine sistem Python'unda çalıştığını",
        "fix": "PowerShell'de .\\.venv\\Scripts\\Activate.ps1 ile ortamı aktifleştirmelisin.",
    },
    {
        "error": "torch.cuda.is_available() False dönüyor",
        "reason": "PyTorch'un CUDA'yı görmediğini veya CUDA destekli kurulum olmadığını",
        "fix": "GPU sürücüsünü, CUDA destekli PyTorch kurulumunu ve doğru Python ortamını kontrol etmelisin.",
    },
    {
        "error": "yanlış Python interpreter kullanılıyor",
        "reason": "paketlerin kurulu olduğu ortam ile çalışan Python yorumlayıcısının farklı olduğunu",
        "fix": "where python, py --version ve venv yolunu kontrol ederek aynı ortamı kullanmalısın.",
    },
    {
        "error": "checkpoint/tokenizer mismatch",
        "reason": "modelin eğitildiği tokenizer ile üretimde kullanılan tokenizer'ın farklı olabileceğini",
        "fix": "Checkpoint ile aynı tokenizer klasörünü kullanmalı ve vocab boyutlarını karşılaştırmalısın.",
    },
    {
        "error": "JSONDecodeError in config",
        "reason": "JSON config dosyasında virgül, tırnak veya parantez hatası olduğunu",
        "fix": "Config dosyasını JSON formatına göre kontrol etmeli ve sondaki gereksiz virgülleri kaldırmalısın.",
    },
    {
        "error": "Windows path problem",
        "reason": "ters slash karakterlerinin veya göreli yolların yanlış yorumlandığını",
        "fix": "Pathlib kullanmalı ve yolları proje kökünden güvenli şekilde birleştirmelisin.",
    },
    {
        "error": "RuntimeError: size mismatch for token_embedding.weight",
        "reason": "checkpoint içindeki vocab boyutu ile mevcut tokenizer vocab boyutunun uyuşmadığını",
        "fix": "Tokenizer'ı değiştirdiysen modeli yeniden eğitmeli veya doğru checkpoint-tokenizer çiftini kullanmalısın.",
    },
]


QUESTION_TEMPLATES = [
    "{error} hatası alıyorum. Ne yapmalıyım?",
    "Python'da {error} neden olur?",
    "{error} hatasını nasıl çözebilirim?",
]


def build_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []

    for topic in ERROR_TOPICS:
        answer = (
            f"Bu hata {topic['reason']} gösterir. "
            f"{topic['fix']}"
        )

        for template in QUESTION_TEMPLATES:
            question = template.format(error=topic["error"])
            examples.append((question, answer))

    return examples


def render_example(question: str, answer: str) -> str:
    return f"Kullanıcı: {question}\nAsistan: {answer}"


def main() -> None:
    examples = build_examples()

    if len(examples) < 60:
        raise ValueError(f"Expected at least 60 examples, got {len(examples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n\n".join(
        render_example(question, answer)
        for question, answer in examples
    )
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print(f"Generated examples: {len(examples)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
