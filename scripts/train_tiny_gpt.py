from pathlib import Path
import sys
import json

import torch
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


DATA_PATH = ROOT_DIR / "data" / "corpus_v1.txt"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("DarkMind Tiny GPT eğitimi başlıyor")
print("=" * 70)
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = ByteLevelBPETokenizer(
    str(VOCAB_PATH),
    str(MERGES_PATH),
)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

print(f"Vocab size: {vocab_size}")

text = DATA_PATH.read_text(encoding="utf-8")
encoded = tokenizer.encode(text)
ids = encoded.ids

print(f"Raw token count: {len(ids)}")

# Veri çok küçük olduğu için şimdilik eğitim pipeline testi için çoğaltıyoruz.
# Gerçek veri setinde bunu yapmayacağız.
while len(ids) < 50_000:
    ids = ids + ids

tokens = torch.tensor(ids, dtype=torch.long)

train_ratio = 0.9
split_idx = int(len(tokens) * train_ratio)

train_data = tokens[:split_idx]
val_data = tokens[split_idx:]

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
)

batch_size = 32
max_steps = 500
eval_interval = 50
learning_rate = 3e-4

model = GPTLanguageModel(config).to(device)

print(f"Model parameters: {count_parameters(model):,}")
print(f"Block size: {config.block_size}")
print(f"Layers: {config.n_layer}")
print(f"Heads: {config.n_head}")
print(f"Embedding size: {config.n_embd}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - config.block_size - 1, (batch_size,))

    x = torch.stack([
        data[i:i + config.block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + config.block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        total_loss = 0.0
        eval_iters = 10

        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses


model.train()

progress = tqdm(range(1, max_steps + 1))

for step in progress:
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    progress.set_description(f"step {step} loss {loss.item():.4f}")

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"\nStep {step}: "
            f"train loss={losses['train']:.4f}, "
            f"val loss={losses['val']:.4f}"
        )

checkpoint_path = CHECKPOINT_DIR / "darkmind_tiny.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": vocab_size,
    },
    checkpoint_path,
)

print("=" * 70)
print("Eğitim tamamlandı.")
print(f"Checkpoint kaydedildi: {checkpoint_path}")
print("=" * 70)