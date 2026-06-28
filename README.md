# DarkMind-30M

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
