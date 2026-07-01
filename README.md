# DarkMind-30M

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/petrofi/darkmind-30m?style=social)

> 🇹🇷 Türkçe odaklı mini LLM projesi — Sıfırdan tokenizer, decoder-only Transformer ve eğitim pipeline'ı.

DarkMind is a Turkish-focused small LLM research and learning project, not a production-grade assistant.

Project references:

- [MODEL_CARD.md](MODEL_CARD.md)
- [ROADMAP.md](ROADMAP.md)
- [experiments/README.md](experiments/README.md)

## Data Pipeline

Clean raw collected text files:

```powershell
python scripts/clean_text.py
```

Build the combined v3 corpus from `data/sources/*.txt` and `data/cleaned/**/*.txt`:

```powershell
python scripts/build_dataset_from_raw.py
```

Run a dataset quality check:

```powershell
python scripts/dataset_quality_check.py
```

Train the tokenizer on `corpus_v3.txt`:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_v3.txt
```

Train the model with the v3 corpus:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m.json --data_path data/processed/corpus_v3.txt
```

Run the interactive chat demo:

```powershell
python scripts/chat_demo.py --checkpoint checkpoints/darkmind_30m.pt --temperature 0.8 --top_k 50 --max_new_tokens 120
```

## Streaming Pretraining Data

Install optional data-ingestion dependencies:

```powershell
pip install -r requirements-data.txt
```

Run a tiny smoke test with Hugging Face streaming sources:

```powershell
python scripts/prepare_pretraining_data.py --max_docs 100 --out data/processed/pretrain_smoke.jsonl
```

Prepare a small local-scale pretraining corpus:

```powershell
python scripts/prepare_pretraining_data.py --max_docs 50000 --out data/processed/pretrain_corpus.jsonl
```

Train from JSONL with conservative local defaults:

```powershell
python scripts/train_from_jsonl.py --data data/processed/pretrain_corpus.jsonl --epochs 1 --batch_size 4 --block_size 256 --save_path models/darkmind-30m-pretrain.pt
```

Do not download full FineWeb, RedPajama, or The Stack locally. Use small streaming samples, respect licenses, and skip gated datasets unless access is explicitly approved. See [docs/data_pipeline.md](docs/data_pipeline.md).

## Automated Dataset Processing

Run the full safe data preparation pipeline:

```powershell
python scripts/auto_prepare_dataset.py
```

Score cleaned examples and write accepted/rejected outputs:

```powershell
python scripts/score_dataset_examples.py
```

Build the curriculum corpus from scored examples:

```powershell
python scripts/build_curriculum_dataset.py
```

Check the final curriculum corpus:

```powershell
python scripts/dataset_quality_check.py --path data/processed/corpus_curriculum_v01.txt
```

Train tokenizer manually after reviewing the dataset:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_curriculum_v01.txt
```

Train the model manually:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_curriculum_v01.txt
```

See [docs/experiment_016_automated_dataset_processing.md](docs/experiment_016_automated_dataset_processing.md) for scoring logic, rejected examples, curriculum ordering, and limitations.

## Dataset Quality and Splits

Build or refresh the dataset source manifest:

```powershell
python scripts/build_dataset_manifest.py
```

Clean raw data, deduplicate cleaned text, and rebuild `corpus_v3.txt` from deduped data:

```powershell
python scripts/clean_text.py
python scripts/deduplicate_dataset.py
python scripts/build_dataset_from_raw.py --input_dir data/deduped
```

Build document-level train/validation/test splits:

```powershell
python scripts/build_train_val_test.py
```

Check whether eval prompts or expected snippets leaked into train data:

```powershell
python scripts/check_eval_leakage.py --eval_path data/evals/darkmind_eval_v02.jsonl
```

Train tokenizer only on the train split:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/splits/train.txt
```

Train the model with explicit train/validation files:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --train_path data/processed/splits/train.txt --val_path data/processed/splits/val.txt
```

See [docs/experiment_013_dataset_quality_and_splits.md](docs/experiment_013_dataset_quality_and_splits.md) for the reasoning and limitations.

## Dataset Generation

Generate deterministic Turkish Python coding examples:

```powershell
python scripts/generate_python_examples.py
```

Generate Turkish coding error explanation examples:

```powershell
python scripts/generate_coding_error_examples.py
```

Generate QA variants for core DarkMind concepts:

```powershell
python scripts/generate_qa_variants.py
```

Run all dataset generators:

```powershell
python scripts/generate_dataset_v01.py
```

Rebuild `corpus_v3.txt` after generating and cleaning raw data:

```powershell
python scripts/clean_text.py
python scripts/build_dataset_from_raw.py
python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt
```

Train tokenizer with `corpus_v3.txt`:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_v3.txt
```

Train model with `corpus_v3.txt`:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_v3.txt
```

Run chat demo:

```powershell
python scripts/chat_demo.py --checkpoint checkpoints/darkmind_30m.pt --temperature 0.8 --top_k 50 --max_new_tokens 120
```

## Self-Improvement Pipeline

This is not fully autonomous self-training. DarkMind does not blindly train on its own outputs. The loop only finds weak answers and creates candidate examples for human review.

Run evaluation prompts against a checkpoint:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m.pt
```

Generate correction candidates from the latest eval run:

```powershell
python scripts/generate_correction_candidates.py
```

Run the convenience loop. This runs eval and creates pending candidates, but does not approve, rebuild, or train:

```powershell
python scripts/self_improve_loop.py --checkpoint checkpoints/darkmind_30m.pt
```

Review the pending candidate file manually. If the examples are correct and useful, approve them explicitly:

```powershell
python scripts/approve_candidates.py --input_path data/self_improvement/pending_review/correction_candidates_YYYYMMDD_HHMMSS.txt --approve_all
```

Rebuild the normal dataset after approval:

```powershell
python scripts/clean_text.py
python scripts/build_dataset_from_raw.py
python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt
```

Then tokenizer and model training remain manual:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_v3.txt
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_v3.txt
```

## Safe Web Data Collection

DarkMind web data collection is allowlist-based. Do not scrape random websites. A source must be added to `configs/web_sources_allowlist.json`, its license must be reviewed, and fetched text must stay in `pending_review` until a human approves it.

Check whether a URL is allowed:

```powershell
python scripts/check_web_source_policy.py --url "https://example.com/page"
```

Fetch one allowlisted URL into pending review with metadata:

```powershell
python scripts/fetch_web_text.py --url "https://example.com/page"
```

Review the fetched text manually:

```powershell
notepad data/raw_collected/web_text/pending_review/<file>.txt
```

Approve only after checking license and quality:

```powershell
python scripts/approve_web_text.py --input_path data/raw_collected/web_text/pending_review/<file>.txt --approve_all
```

Inspect approved web data before corpus build:

```powershell
python scripts/web_data_quality_check.py
```

Clean and rebuild corpus after approval:

```powershell
python scripts/clean_text.py
python scripts/build_dataset_from_raw.py
```

`scripts/clean_text.py` intentionally skips `data/raw_collected/web_text/pending_review/` so unreviewed web text does not enter training data automatically.

## Turkish Coding Ability Track

Generate the stronger Turkish Python instruction dataset:

```powershell
python scripts/generate_python_instruction_v02.py
```

Validate Python code blocks without executing them:

```powershell
python scripts/validate_python_examples.py --strict
```

Rebuild corpus after generation:

```powershell
python scripts/clean_text.py
python scripts/build_dataset_from_raw.py
```

Train tokenizer and model manually:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_v3.txt
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_v3.txt
```

Run the Turkish coding eval:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m_1000step.pt --eval_path data/evals/darkmind_code_eval_v01.jsonl
```

Run the safe coding self-improvement loop. This only evaluates and creates pending review candidates:

```powershell
python scripts/run_code_eval_cycle.py --checkpoint checkpoints/darkmind_30m_1000step.pt
```

## Turkish Code Data Factory

Generate, validate, clean, and rebuild the larger Turkish coding dataset without training the model:

```powershell
python scripts/auto_code_dataset_cycle.py
```

Validate the generated coding dataset directly:

```powershell
python scripts/validate_code_dataset_quality.py --strict
```

Train tokenizer manually after reviewing the generated corpus:

```powershell
python scripts/train_tokenizer.py --data_path data/processed/corpus_curriculum_v01.txt
```

Train the model manually:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --data_path data/processed/corpus_curriculum_v01.txt
```

Run the stronger Turkish coding eval:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m_1000step.pt --eval_path data/evals/darkmind_code_eval_v02.jsonl
```

See [docs/experiment_017_turkish_code_data_factory.md](docs/experiment_017_turkish_code_data_factory.md) for safety checks, syntax validation, and limitations.

## Inference Quality and Checkpoint Metadata

Check whether a checkpoint appears compatible with the current tokenizer:

```powershell
python scripts/check_checkpoint_compatibility.py --checkpoint checkpoints/darkmind_30m_1000step.pt
```

Write checkpoint metadata without modifying the checkpoint:

```powershell
python scripts/write_checkpoint_metadata.py --checkpoint checkpoints/darkmind_30m_1000step.pt --config configs/darkmind_30m_1000step.json --eval_run data/self_improvement/runs/<eval_run>.jsonl
```

Run a small deterministic inference smoke suite:

```powershell
python scripts/run_inference_suite.py --checkpoint checkpoints/darkmind_30m_1000step.pt --preset deterministic
```

Run the chat demo with cleaner inference controls:

```powershell
python scripts/chat_demo.py --checkpoint checkpoints/darkmind_30m_1000step.pt --temperature 0.2 --top_k 5 --max_new_tokens 80 --repetition_penalty 1.1
```

See [docs/experiment_015_inference_quality_and_versioning.md](docs/experiment_015_inference_quality_and_versioning.md) for inference cleanup, metadata, presets, chat logging, and limitations.

## Experiment Tracking

Log a real experiment after training/evaluation:

```powershell
python scripts/log_experiment.py --experiment_id exp_012 --name "LLM development framework baseline" --eval_path data/evals/darkmind_eval_v02.jsonl --eval_path data/evals/darkmind_code_eval_v01.jsonl --notes "Fill with real run details after evaluation."
```

Compare two eval runs:

```powershell
python scripts/compare_eval_runs.py --before data/self_improvement/runs/eval_run_BEFORE.jsonl --after data/self_improvement/runs/eval_run_AFTER.jsonl
```

Show the latest eval run summaries:

```powershell
python scripts/latest_eval_summary.py --limit 10
```

Longer 30M training config:

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_longer_train.json
```

Experimental 50M config warning:

```powershell
python scripts/train_from_config.py --config configs/darkmind_50m_experimental.json
```

The 50M config may be too heavy for the current machine. Test carefully with short runs before any long training.

## Training Strategy and Model Selection

Run the safest next curriculum training plan for the RTX 4060 Laptop GPU:

```powershell
python scripts/run_training_plan.py --config configs/darkmind_30m_curriculum_3000step.json --data_path data/processed/corpus_curriculum_v01.txt --eval_path data/evals/darkmind_eval_v02.jsonl --eval_path data/evals/darkmind_code_eval_v02.jsonl
```

Benchmark the best checkpoint from that run:

```powershell
python scripts/benchmark_suite.py --checkpoint checkpoints/darkmind_30m_curriculum_3000step_best.pt
```

Promote a checkpoint only after reviewing a real eval run:

```powershell
python scripts/promote_checkpoint.py --candidate_checkpoint checkpoints/darkmind_30m_curriculum_3000step_best.pt --eval_run data/self_improvement/runs/<eval_run>.jsonl --min_pass_rate 0.70
```

The 30M curriculum path should be tried before the experimental 50M config. The 50M config may be heavy for laptop GPU VRAM and should only be run after the 30M benchmark results justify it.

See [docs/experiment_018_training_strategy_and_model_selection.md](docs/experiment_018_training_strategy_and_model_selection.md) for the recommended model-selection order.

DarkMind-30M, Türkçe odaklı küçük bir dil modeli geliştirme projesidir.

Bu proje, hazır bir modeli fine-tune etmek yerine sıfırdan bir mini decoder-only Transformer mimarisi kurmayı, kendi tokenizer'ını eğitmeyi ve kendi training loop'u ile model eğitmeyi amaçlar.

## Hedef

- Türkçe odaklı mini LLM geliştirmek
- Kendi tokenizer'ını eğitmek
- Decoder-only Transformer mimarisini sıfırdan yazmak
- CUDA destekli eğitim pipeline'ı kurmak
- Küçük modelden başlayarak DarkMind-5M ve DarkMind-30M seviyesine ilerlemek

## Mevcut Durum

- Byte-Level BPE tokenizer eğitildi
- Tiny GPT mimarisi yazıldı
- CUDA aktif edildi
- İlk checkpoint üretildi
- Eğitim pipeline'ı başarıyla çalıştırıldı

## Sistem

- Python: 3.13.12
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- RAM: 32 GB

## Proje Yapısı

```txt
darkmind-30m/
├── configs/
├── data/
├── docs/
├── model/
├── scripts/
├── tokenizer/
├── training/
├── checkpoints/
└── README.md
