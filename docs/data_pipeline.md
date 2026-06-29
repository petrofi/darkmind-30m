# DarkMind Data Pipeline

This document describes the small-scale, legal, streaming data pipeline for DarkMind-30M. It is designed for local experiments, not for downloading huge web corpora.

## Supported Sources

The script `scripts/prepare_pretraining_data.py` supports these source names:

- `wikipedia`: Turkish and English Wikipedia through Hugging Face streaming when accessible.
- `fineweb`: `HuggingFaceFW/fineweb-edu` using a small streaming sample/config when available.
- `redpajama`: `togethercomputer/RedPajama-Data-V2` using streaming when available.
- `stack`: `bigcode/the-stack-v2` for code data when accessible and permitted.

If a dataset requires authentication, gated access, license acceptance, or the config names have changed, the script skips that source and continues with the remaining sources.

## Legal And Safety Notes

- Do not scrape random websites.
- Do not use Reddit data for training.
- Do not include private or personal data.
- Do not include passwords, API keys, emails, phone numbers, addresses, tokens, or credentials.
- Respect dataset licenses and terms before using any downloaded or streamed data for training.
- Big datasets such as FineWeb, RedPajama, and The Stack should not be fully downloaded to a normal local machine.

## Smoke Test

Install data dependencies:

```powershell
pip install -r requirements-data.txt
```

Run a tiny streaming sample:

```powershell
python scripts/prepare_pretraining_data.py --max_docs 100 --out data/processed/pretrain_smoke.jsonl
```

This writes JSONL rows like:

```json
{"text": "...", "source": "wikipedia_tr", "language": "tr"}
```

## Small Real Preparation

Run a small local-scale preparation:

```powershell
python scripts/prepare_pretraining_data.py --max_docs 50000 --out data/processed/pretrain_corpus.jsonl
```

The default mix prioritizes Turkish text, then English general/technical text, then code data if accessible. If code data is unavailable, the remaining selected sources still run.

If a streaming run is interrupted, resume without duplicating exact texts:

```powershell
python scripts/prepare_pretraining_data.py --max_docs 50000 --out data/processed/pretrain_corpus.jsonl --resume
```

## Training From JSONL

For a conservative local test:

```powershell
python scripts/train_from_jsonl.py --data data/processed/pretrain_corpus.jsonl --epochs 1 --batch_size 4 --block_size 256 --save_path models/darkmind-30m-pretrain.pt
```

This uses the existing tokenizer and GPT model implementation. A 30M-quality run needs more data, more steps, better evaluation, and careful checkpoint tracking.

To train with a validation split:

```powershell
python scripts/train_from_jsonl.py --data data/processed/pretrain_corpus.jsonl --epochs 1 --batch_size 4 --block_size 256 --max_steps 1000 --val_ratio 0.05 --eval_interval 100 --eval_batches 20 --save_path models/darkmind-30m-pretrain.pt
```

The script splits tokenized data into train and validation sections, prints train and validation loss during training, and stores the final train and validation loss in the checkpoint metadata.

## Verifying The Real Model Architecture

`scripts/train_from_jsonl.py` reads `configs/darkmind_30m_1000step.json` by default and instantiates the real `GPTLanguageModel` from `model/gpt.py`. It should print the config path, tokenizer path, layer count, head count, embedding size, block size, vocab size, and parameter count before training starts.

Run a one-step architecture check with the smoke dataset:

```powershell
python scripts/train_from_jsonl.py --data data/processed/pretrain_smoke.jsonl --epochs 1 --batch_size 4 --block_size 256 --max_steps 1 --save_path models/darkmind-30m-arch-check.pt
```

With the current `configs/darkmind_30m_1000step.json`, the expected architecture lines are:

- Layers: 8
- Heads: 8
- Embedding size: 512
- Block size: 256

The exact parameter count depends on the tokenizer vocabulary size. If the script reports a much smaller model, check that `--config`, `--tokenizer`, and any manual `--n_layer`, `--n_head`, or `--n_embd` overrides are not pointing to a smaller experiment.

## Generating From A Checkpoint

Generate a quick sample from a trained checkpoint:

```powershell
python scripts/generate_from_checkpoint.py --checkpoint models/darkmind-30m-10k.pt --config configs/darkmind_30m_1000step.json --tokenizer tokenizer/darkmind-tokenizer --prompt "Merhaba, sen kimsin?" --max_new_tokens 100 --temperature 0.8 --top_k 50
```

The generation script uses the real `GPTLanguageModel` from `model/gpt.py`. If the checkpoint contains its own embedded model config, that config is used so the saved weights match the instantiated model. The `--config` argument is still useful for checkpoints that only contain a raw state dict.

## Comparing Outputs Before And After Training

Use the same prompt and decoding settings for each checkpoint:

```powershell
python scripts/generate_from_checkpoint.py --checkpoint models/before.pt --prompt "Merhaba, sen kimsin?" --max_new_tokens 100 --temperature 0.8 --top_k 50
python scripts/generate_from_checkpoint.py --checkpoint models/after.pt --prompt "Merhaba, sen kimsin?" --max_new_tokens 100 --temperature 0.8 --top_k 50
```

Compare both the output text and the training metadata. A better checkpoint should usually show lower validation loss and more coherent generations, but lower train loss alone is not enough because it can also indicate overfitting.

## Cleaning And Filtering

The preparation script:

- removes HTML
- normalizes whitespace
- removes repeated boilerplate lines
- filters very short documents
- filters too many URLs
- filters too many non-text characters
- filters emails, phone-like strings, and secret-related patterns
- deduplicates exact normalized texts with hashing
- keeps Turkish characters intact
- uses `langdetect` when installed and falls back to lightweight heuristics

## Expected Limitations

DarkMind-30M is a small model. Even clean 50k-document samples are tiny compared with serious LLM pretraining. This pipeline is useful for controlled experiments and learning, but it does not make the model production-ready or broadly capable.
