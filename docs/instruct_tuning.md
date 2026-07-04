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

For the larger v0.2 smoke dataset:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_v0_2.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 3 --batch_size 4 --block_size 256 --max_steps 300 --lr 0.00005 --val_ratio 0.1 --eval_interval 50 --eval_batches 10 --save_path models/darkmind-30m-instruct-v0.2-smoke.pt
```

## Generate

Use `--chat_format` so the prompt matches the instruction template:

```powershell
.\.venv\Scripts\python.exe scripts\generate_from_checkpoint.py --checkpoint models/darkmind-30m-instruct-v0.1.pt --prompt "Python nedir?" --chat_format --max_new_tokens 80 --temperature 0.3 --top_k 40 --top_p 0.9 --repetition_penalty 1.15 --no_repeat_ngram_size 3
```

With `--chat_format`, the script internally wraps the prompt as:

```text
Kullanıcı: Python nedir?
Asistan:
```

The generation script prints both `RAW OUTPUT` and `ASSISTANT ANSWER`. Use the assistant answer for quick reading, and inspect raw output when debugging repeated turns, special tokens, or tokenizer artifacts.

## Eval

Run the fixed instruction prompt set:

```powershell
.\.venv\Scripts\python.exe scripts\eval_instruct_prompts.py --checkpoint models/darkmind-30m-instruct-v0.2-smoke.pt --eval data/eval/darkmind_eval_prompts.jsonl --out reports/instruct_eval_outputs.md
```

The report is written to `reports/instruct_eval_outputs.md`.

## Instruct v0.3 Dataset Pipeline

Prepare a controlled v0.3 dataset:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_instruct_data.py --max_examples 10000 --out data/instruct/darkmind_instruct_v0_3.jsonl
```

Inspect the prepared dataset:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_instruct_data.py --data data/instruct/darkmind_instruct_v0_3.jsonl
```

Train the recommended v0.3 run:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_v0_3.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 3 --batch_size 4 --block_size 256 --max_steps 2000 --lr 0.00003 --val_ratio 0.1 --eval_interval 100 --eval_batches 20 --save_path models/darkmind-30m-instruct-v0.3.pt
```

For smoke tests, use `--max_examples 1000`. Smoke preparation intentionally skips slow external dataset downloads and uses local identity/seed/fallback data so the pipeline stays fast and testable.

Attempted external sources for the full pipeline:

- TFLai/Turkish-Alpaca
- malhajar/OpenOrca-tr
- malhajar/alpaca-gpt4-tr
- CohereLabs/aya_dataset
- merve/turkish_instructions
- atasoglu/instruction-turkish
- afkfatih/turkish-distilled-5K
- alibayram/diyalog-dataset

The script skips unavailable, gated, unsupported, unclear, or slow sources safely and records reasons in the `.meta.json` file next to the output.

## Limitations

This is a 28M parameter local experiment, not a production assistant. A small seed dataset can teach basic answer style, but it can also overfit, memorize examples, or fail on prompts outside the dataset. It should not be described as ChatGPT-level, broadly knowledgeable, or production-ready.

The next useful steps are larger clean Turkish instruction data, held-out evaluation prompts, before/after comparison, and careful safety/quality review before adding examples back into training.

Known v0.2 smoke limitations:

- Identity behavior improved on some prompts, but related prompts can still fail.
- Turkish special characters may appear corrupted in generation outputs.
- Some answers still drift into base-pretraining text such as event lists.
- More data and longer controlled SFT are needed before judging the model seriously.

Known v0.3 smoke limitations:

- The smoke dataset is local-only by design; it is not a substitute for the full 10k mixed dataset.
- Identity behavior improved, but answers can still contain irrelevant prefixes.
- Programming and support answers are still unstable.
- The model can still continue into base-pretraining text after special tokens.
