# DarkMind-30M

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