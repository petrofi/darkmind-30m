from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "corpus_v1.txt"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

print("DarkMind tokenizer eğitimi başlıyor...")
print(f"Veri dosyası: {DATA_PATH}")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[str(DATA_PATH)],
    vocab_size=8000,
    min_frequency=1,
    special_tokens=[
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

tokenizer.save_model(str(TOKENIZER_DIR))

print("Tokenizer eğitimi tamamlandı.")
print(f"Kaydedildi: {TOKENIZER_DIR}")