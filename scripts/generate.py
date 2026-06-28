from pathlib import Path
import sys

import torch
from tokenizers import ByteLevelBPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel


TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "darkmind_tiny.pt"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    str(VOCAB_PATH),
    str(MERGES_PATH),
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

config = GPTConfig(**checkpoint["config"])

model = GPTLanguageModel(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "DarkMind"
encoded = tokenizer.encode(prompt)

idx = torch.tensor([encoded.ids], dtype=torch.long).to(device)

with torch.no_grad():
    generated = model.generate(
        idx,
        max_new_tokens=120,
        temperature=0.8,
        top_k=50,
    )

output_ids = generated[0].tolist()
text = tokenizer.decode(output_ids)

print("=" * 70)
print("PROMPT:")
print(prompt)
print("=" * 70)
print("DARKMIND OUTPUT:")
print(text)
print("=" * 70)