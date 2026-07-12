# DarkMind-30M

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20ready-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#project-status)

DarkMind-30M is a Turkish-focused small language model research project. It includes a local tokenizer, GPT-style decoder-only Transformer models, data preparation tools, instruction-tuning experiments, evaluation scripts, and a teacher-student distillation track.

This repository is intended for learning, controlled experimentation, and transparent iteration. It is not a production assistant and should not be treated as a high-stakes model.

## Project Status

DarkMind is an active research prototype.

- Architecture: GPT-style decoder-only Transformer implemented in PyTorch.
- Scale: 30M-family experiments, roughly 27M-30M parameters depending on tokenizer vocabulary size.
- Primary language target: Turkish, with software-assistant behavior as the main applied track.
- Current training strategy: continue from compatible checkpoints and use inspected instruction/distillation data.
- Safety posture: no blind self-training, no unreviewed web scraping, no high-stakes use claims.

Important limitation: existing checkpoints and tokenizer compatibility matter. Do not replace the tokenizer for checkpoint-compatible pilot experiments unless the experiment explicitly calls for it.

## Repository Layout

```text
.
|-- configs/              Model and training configuration files
|-- data/                 Local corpora, processed text, eval prompts, and generated data
|-- darkmind_assistant/   Assistant/runtime experiments
|-- darkmind_distill/     Qwen teacher-student distillation pipeline
|-- darkmind_v2/          Phase 0 tokenizer, corpus, and base-eval validation infrastructure
|-- docs/                 Experiment notes and pipeline documentation
|-- experiments/          Experiment index and historical notes
|-- model/                GPT model implementation
|-- scripts/              Training, data, eval, tokenizer, and utility scripts
|-- tests/                Unit tests
|-- tokenizer/            Local ByteLevel BPE tokenizer files
```

Large runtime artifacts such as checkpoints, generated JSONL datasets, local configs, logs, and reports are intentionally ignored by git.

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the unit tests:

```powershell
python -m pytest tests
```

Check that a Python file compiles:

```powershell
python -m py_compile scripts/train_from_config.py model/gpt.py
```

## Model Training

Train from a config and a plain text corpus:

```powershell
python scripts/train_from_config.py `
  --config configs/darkmind_30m_1000step.json `
  --data_path data/processed/corpus_v3.txt
```

Train from JSONL pretraining data:

```powershell
python scripts/train_from_jsonl.py `
  --data data/processed/pretrain_corpus.jsonl `
  --epochs 1 `
  --batch_size 4 `
  --block_size 256 `
  --save_path models/darkmind-30m-pretrain.pt
```

Instruction fine-tuning from JSONL:

```powershell
python scripts/train_instruct_jsonl.py `
  --data data/instruct/darkmind_instruct_seed.jsonl `
  --base_checkpoint models/darkmind-30m-10k-step15000.pt `
  --epochs 2 `
  --batch_size 4 `
  --block_size 256 `
  --max_steps 250 `
  --lr 0.000015 `
  --save_path models/darkmind-30m-instruct-v0.1.pt
```

For GPU-only training, use:

```powershell
python scripts/train_instruct_jsonl.py --require-cuda ...
```

With `--require-cuda`, the script refuses to fall back to CPU/RAM training if PyTorch cannot see CUDA.

## Inference

Generate text from a checkpoint:

```powershell
python scripts/generate_from_checkpoint.py `
  --checkpoint models/darkmind-30m-10k-step15000.pt `
  --prompt "Kullanici: Python'da liste nedir?\nAsistan:" `
  --max_new_tokens 120 `
  --temperature 0.8 `
  --top_k 50
```

Run the older chat demo flow:

```powershell
python scripts/chat_demo.py `
  --checkpoint checkpoints/darkmind_30m.pt `
  --temperature 0.8 `
  --top_k 50 `
  --max_new_tokens 120
```

## Data Pipeline

Clean raw local text:

```powershell
python scripts/clean_text.py
```

Build the combined corpus:

```powershell
python scripts/build_dataset_from_raw.py
```

Run quality checks:

```powershell
python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt
```

Build train/validation/test splits:

```powershell
python scripts/build_train_val_test.py
```

Train the tokenizer on a reviewed corpus:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/splits/train.txt
```

Check eval leakage:

```powershell
python scripts/check_eval_leakage.py --eval_path data/evals/darkmind_eval_v02.jsonl
```

## Streaming Data

Optional streaming-data dependencies:

```powershell
pip install -r requirements-data.txt
```

Smoke test Hugging Face streaming ingestion:

```powershell
python scripts/prepare_pretraining_data.py `
  --max_docs 100 `
  --out data/processed/pretrain_smoke.jsonl
```

Prepare a small local-scale corpus:

```powershell
python scripts/prepare_pretraining_data.py `
  --max_docs 50000 `
  --out data/processed/pretrain_corpus.jsonl
```

Do not download full FineWeb, RedPajama, The Stack, or gated datasets locally unless the licensing, storage, and compute plan has been reviewed.

## Instruction and Self-Improvement Pipeline

DarkMind includes a human-in-the-loop self-improvement workflow. It evaluates a checkpoint, creates candidate correction examples, and waits for explicit review before examples enter training data.

Run evaluation prompts:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m.pt
```

Generate correction candidates:

```powershell
python scripts/generate_correction_candidates.py
```

Run the convenience loop:

```powershell
python scripts/self_improve_loop.py --checkpoint checkpoints/darkmind_30m.pt
```

Approve candidates only after manual review:

```powershell
python scripts/approve_candidates.py `
  --input_path data/self_improvement/pending_review/correction_candidates_YYYYMMDD_HHMMSS.txt `
  --approve_all
```

## Teacher-Student Distillation

The `darkmind_distill/` directory contains a Qwen teacher-student distillation workflow. The goal is to generate synthetic instruction-response data with a stronger local teacher model, inspect it strictly, then train DarkMind as the student only when quality gates pass.

Expected local teacher server:

```text
http://localhost:1234/v1
model: local-model
```

Generate a small smoke dataset:

```powershell
python darkmind_distill/generate_qwen_distill_dataset.py --smoke
```

Generate controlled chunks:

```powershell
python darkmind_distill/generate_qwen_distill_dataset.py --max-new 100
python darkmind_distill/generate_qwen_distill_dataset.py --target-total 300
python darkmind_distill/generate_qwen_distill_dataset.py --categories programming,debugging --languages tr,en --max-new 50
```

Inspect staged milestones:

```powershell
python darkmind_distill/inspect_distill_dataset.py --target-mode staged --min-total 300
python darkmind_distill/inspect_distill_dataset.py --target-mode full
```

Current Pilot500 TR/EN v2 workflow:

```powershell
python darkmind_distill/build_pilot500_tr_en_v2.py --batch-size 3
python darkmind_distill/audit_pilot500_tr_en_v2.py
python darkmind_distill/audit_multilingual_tokenizer.py `
  --data darkmind_distill/data/darkmind_qwen_distill_pilot500_tr_en_v2.jsonl `
  --report darkmind_distill/reports/pilot500_tr_en_v2_tokenizer_audit.md `
  --block-size 256
```

Training is intentionally gated. Do not train on a distillation dataset until duplicate, contamination, tokenizer, eval-overlap, and distribution checks pass.

## DarkMind-30M Pilot500 Failure Analysis

Pilot500 TR/EN v2 was a controlled teacher-student distillation experiment, not a production release. The student used a 28,127,232 parameter DarkMind model and a 500-example Turkish-English Qwen teacher dataset.

Final training metrics improved numerically, with train loss `3.4495` and validation loss `3.7017`, but generation quality failed. The base checkpoint failed `8/8` deterministic greedy generation tests, and the Pilot500 student produced mixed-script, corrupted, and semantically invalid output. The tokenizer audit found mojibake and encoding artifacts in the vocabulary, including forms like `TÃƒÂ¼rkiye` and `KullanÃ„Â±cÃ„Â±`.

The SFT formatting, prompt masking, response supervision, EOS supervision, and label alignment were audited and verified as correct. The Pilot500 student checkpoint was not continued. This failure is kept as an engineering milestone and led to the DarkMind v2 base-pipeline redesign.

See:

- [darkmind_distill/reports/pilot500_failure_diagnosis.md](darkmind_distill/reports/pilot500_failure_diagnosis.md)
- [docs/darkmind-v2-base-pipeline.md](docs/darkmind-v2-base-pipeline.md)
- [darkmind_v2/README.md](darkmind_v2/README.md)

DarkMind v2 Phase 1B has since built and validated the 50M-character tokenizer corpus, compared four SentencePiece candidates, and frozen Candidate D BPE 24k. No DarkMind v2 model training has started; the next step is tiny base smoke planning with tied embeddings.

## Evaluation

Core evaluation scripts:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m.pt
python scripts/eval_instruct_prompts.py --checkpoint models/darkmind-30m-instruct-v0.1.pt
python scripts/compare_eval_runs.py --before reports/before.jsonl --after reports/after.jsonl
```

Pilot500 evaluation:

```powershell
python darkmind_distill/run_pilot500_eval.py eval `
  --prompts darkmind_distill/data/pilot500_tr_en_v2_eval_prompts.jsonl `
  --checkpoint models/darkmind-30m-10k-step15000.pt `
  --out darkmind_distill/reports/pilot500_tr_en_v2_base_eval.md `
  --json-out darkmind_distill/reports/pilot500_tr_en_v2_base_eval.jsonl
```

## Safety and Data Policy

DarkMind follows a conservative research workflow:

- No blind internet scraping.
- No Reddit training data.
- No automatic training on model outputs.
- No unreviewed self-improvement examples.
- No high-stakes medical, legal, financial, or security-sensitive claims.
- No claims of ChatGPT-level capability.
- Human review is required before generated examples enter training.

See also:

- [MODEL_CARD.md](MODEL_CARD.md)
- [ROADMAP.md](ROADMAP.md)
- [docs/data_pipeline.md](docs/data_pipeline.md)
- [docs/experiments.md](docs/experiments.md)
- [darkmind_distill/README.md](darkmind_distill/README.md)

## Development Notes

Check repository status:

```powershell
git status --short --branch
```

Run tests:

```powershell
python -m pytest tests
```

Compile changed Python files:

```powershell
python -m py_compile scripts/train_instruct_jsonl.py darkmind_distill/generate_qwen_distill_dataset.py
```

Sync release artifacts to Hugging Face only after reviewing the target repository and files:

```powershell
python scripts/hf_hub_sync.py --help
```

## License

This project is released under the [MIT License](LICENSE).

## Contributing

Contributions should preserve the research discipline of the project: small scoped changes, documented experiments, explicit data provenance, and honest evaluation. See [CONTRIBUTING.md](CONTRIBUTING.md).
