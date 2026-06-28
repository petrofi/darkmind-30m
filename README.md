# DarkMind-30M

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Stars](https://img.shields.io/github/stars/petrofi/darkmind-30m?style=social)

> 🇹🇷 Türkçe odaklı mini LLM projesi — Sıfırdan tokenizer, decoder-only Transformer ve eğitim pipeline'ı.


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
