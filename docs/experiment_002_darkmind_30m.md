**# Experiment 002 — DarkMind 30M GPU Training**



**## Amaç**



**Bu deneyin amacı, DarkMind projesinde daha büyük bir decoder-only Transformer konfigürasyonunun RTX 4060 Laptop GPU üzerinde çalışıp çalışmadığını test etmektir.**



**## Sistem**



**- Python: 3.13.12**

**- PyTorch: 2.11.0+cu128**

**- CUDA: Aktif**

**- GPU: NVIDIA GeForce RTX 4060 Laptop GPU**

**- RAM: 32 GB**



**## Model Konfigürasyonu**



**- block\_size: 256**

**- n\_layer: 8**

**- n\_head: 8**

**- n\_embd: 512**

**- dropout: 0.1**

**- vocab\_size: 705**



**## Eğitim Konfigürasyonu**



**- batch\_size: 8**

**- max\_steps: 300**

**- eval\_interval: 50**

**- learning\_rate: 0.0003**

**- min\_tokens: 200000**



**## Sonuçlar**



**- Model parameters: 25,712,128**

**- Final train loss: 0.006427645403891802**

**- Final val loss: 0.009647576045244933**

**- Eğitim süresi: yaklaşık 15 saniye**

**- Checkpoint: checkpoints/darkmind\_30m.pt**



**## Değerlendirme**



**Model başarıyla CUDA üzerinde eğitildi ve checkpoint üretildi.**



**Bu deney kalite testi değildir. Kullanılan veri seti çok küçük olduğu için modelin düşük loss değeri gerçek dil öğreniminden çok veri tekrarını ve ezberlemeyi gösterir.**



**Bu deneyin ana başarısı, DarkMind projesinde 25M+ parametreli bir sıfırdan GPT modelinin yerel GPU üzerinde çalıştırılabilmesidir.**



**## Sonraki Adım**



**Bir sonraki aşamada veri seti büyütülecek ve DarkMind Corpus v0.2 hazırlanacaktır.**

