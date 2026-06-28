# Experiment 001 — DarkMind Tiny GPT

## Amaç

Bu deneyin amacı, DarkMind projesi için sıfırdan eğitilen ilk decoder-only Transformer pipeline'ını çalıştırmaktır.

## Sistem

- Python: 3.13.12
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- RAM: 32 GB
- PyTorch: 2.11.0+cu128
- CUDA: Aktif

## Bileşenler

- Tokenizer: Byte-Level BPE
- Model tipi: Decoder-only Transformer
- Training loop: Sıfırdan yazıldı
- Checkpoint: checkpoints/darkmind_tiny.pt

## Model Konfigürasyonu

- block_size: 128
- n_layer: 4
- n_head: 4
- n_embd: 256
- dropout: 0.1

## Notlar

Bu model kalite amacıyla değil, pipeline doğrulama amacıyla eğitildi.
Veri seti çok küçük olduğu için modelin ürettiği metinler anlamlı olmayabilir.
Bu deneyin başarısı, tokenizer + model + training loop + checkpoint sürecinin çalışmasıdır.

## Sonuç

DarkMind için ilk sıfırdan eğitilmiş mini GPT pipeline'ı başarıyla çalıştırıldı.