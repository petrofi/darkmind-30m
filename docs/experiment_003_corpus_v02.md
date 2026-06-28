**# Experiment 003 — DarkMind Corpus v0.2**



**## Amaç**



**Bu deneyin amacı, DarkMind projesinde ilk yapılandırılmış Türkçe corpus sürümünü oluşturmak, tokenizer'ı yeni corpus ile yeniden eğitmek ve DarkMind 30M konfigürasyonunu bu veriyle test etmektir.**



**## Corpus**



**Kaynak dosyalar:**



**- data/sources/ai\_notes\_tr.txt**

**- data/sources/emotion\_notes\_tr.txt**

**- data/sources/python\_notes\_tr.txt**

**- data/sources/qa\_samples\_tr.txt**



**Corpus builder:**



**- scripts/build\_corpus.py**



**Çıktı:**



**- data/processed/corpus\_v2.txt**



**## Corpus Sonuçları**



**- Documents: 4**

**- Characters: 7246**

**- Original token count: 1293**



**## Tokenizer**



**- Tokenizer type: Byte-Level BPE**

**- Yeni vocab size: 1589**



**## Model**



**- Run name: darkmind\_30m**

**- Model parameters: 26,164,736**

**- Block size: 256**

**- Layers: 8**

**- Heads: 8**

**- Embedding size: 512**

**- Batch size: 8**



**## GPU Eğitimi**



**- Device: cuda**

**- GPU: NVIDIA GeForce RTX 4060 Laptop GPU**

**- Max steps: 300**

**- Final train loss: 0.037350621074438095**

**- Final val loss: 0.04313364624977112**

**- Eğitim süresi: yaklaşık 16 saniye**



**## Generation Testleri**



**Prompt: Bir dil modeli**



**Model, veri kalitesi, tokenizer ve Transformer mimarisi hakkında corpus içeriğinden devam üretebildi.**



**Prompt: İnsan bazen**



**Model, motivasyon ve kişisel gelişim temalı metinlerden devam üretebildi.**



**Prompt: DarkMind projesinin amacı**



**Model, soru-cevap formatında DarkMind ve Darklove ilişkisini açıklayan metin üretebildi.**



**## Değerlendirme**



**Bu deneyde model hâlâ gerçek anlamda genelleme yapan güçlü bir dil modeli değildir. Veri seti küçük olduğu için model büyük ölçüde corpus içeriğini öğrenip devam ettirmektedir.**



**Ancak bu deneyin başarısı, DarkMind projesinin uçtan uca çalışan bir araştırma pipeline'ına dönüşmesidir:**



**veri kaynakları → corpus builder → tokenizer eğitimi → 26M parametreli model eğitimi → checkpoint generation → GitHub kaydı.**



**## Sonraki Adım**



**Bir sonraki hedef DarkMind Corpus v0.3 olacaktır. Hedef, corpus boyutunu 50.000+ karakter seviyesine çıkarmak ve daha çeşitli Türkçe örnekler eklemektir.**

