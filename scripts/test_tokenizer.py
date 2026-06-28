from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"

vocab_path = TOKENIZER_DIR / "vocab.json"
merges_path = TOKENIZER_DIR / "merges.txt"

tokenizer = ByteLevelBPETokenizer(
    str(vocab_path),
    str(merges_path),
)

text = "DarkMind sıfırdan eğitilen küçük bir Türkçe yapay zeka modelidir."

encoded = tokenizer.encode(text)

print("Metin:")
print(text)

print("\nToken ID'leri:")
print(encoded.ids)

print("\nTokenlar:")
print(encoded.tokens)

print("\nDecode:")
print(tokenizer.decode(encoded.ids))