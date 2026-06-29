# DarkMind Instruction Tuning

DarkMind-30M base pretraining teaches the model to continue text. This is useful for fluency, but it does not reliably teach the model to answer a user, follow a request, or stay in a clean dialogue format.

Instruction tuning is the next small supervised step. It shows the model examples in a fixed prompt/answer format:

```json
{"prompt":"Python nedir?","response":"Python, okunabilir sözdizimine sahip yüksek seviyeli bir programlama dilidir."}
```

During training, each row is formatted as:

```text
Kullanıcı: Python nedir?
Asistan: Python, okunabilir sözdizimine sahip yüksek seviyeli bir programlama dilidir.
```

## Train

Run a small local instruction fine-tune from the current base checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_seed.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 10 --batch_size 4 --block_size 256 --max_steps 1000 --save_path models/darkmind-30m-instruct-v0.1.pt
```

For a smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_seed.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 3 --batch_size 4 --block_size 256 --max_steps 100 --save_path models/darkmind-30m-instruct-smoke.pt
```

## Generate

Use `--chat_format` so the prompt matches the instruction template:

```powershell
.\.venv\Scripts\python.exe scripts\generate_from_checkpoint.py --checkpoint models/darkmind-30m-instruct-v0.1.pt --prompt "Python nedir?" --chat_format --max_new_tokens 80 --temperature 0.3 --top_k 40 --top_p 0.9 --repetition_penalty 1.15 --no_repeat_ngram_size 3
```

## Limitations

This is a 28M parameter local experiment, not a production assistant. A small seed dataset can teach basic answer style, but it can also overfit, memorize examples, or fail on prompts outside the dataset. It should not be described as ChatGPT-level, broadly knowledgeable, or production-ready.

The next useful steps are larger clean Turkish instruction data, held-out evaluation prompts, before/after comparison, and careful safety/quality review before adding examples back into training.
