**# Experiment 004 — DarkMind Corpus v0.3**



**## Amaç**



**Bu deneyin amacı, DarkMind corpus'unu daha çeşitli Türkçe veri kaynaklarıyla büyütmek, tokenizer'ı yeniden eğitmek ve DarkMind 30M modelini güncellenmiş veriyle test etmektir.**



**## Corpus**



**Kaynak dosya sayısı: 10**



**Kaynak türleri:**



**- Yapay zeka notları**

**- Transformer notları**

**- Python notları**

**- Model eğitimi notları**

**- Kod hataları ve hata ayıklama notları**

**- DarkMind proje günlüğü**

**- Darklove güvenlik notları**

**- Duygu ve motivasyon notları**

**- Soru-cevap örnekleri**

**- Mini diyaloglar**



**## Corpus İstatistikleri**



**- Characters: 26,715**

**- Lines: 447**

**- Non-empty lines: 360**

**- Tokens after tokenizer retrain: 4,630**

**- Tokenizer vocab size: 3,108**



**## Model**



**- Run name: darkmind\_30m**

**- Model parameters: 26,942,464**

**- Block size: 256**

**- Layers: 8**

**- Heads: 8**

**- Embedding size: 512**

**- Batch size: 8**

**- Max steps: 300**



**## Eğitim**



**- Device: cuda**

**- GPU: NVIDIA GeForce RTX 4060 Laptop GPU**

**- Final train loss: 0.2155756890773773**

**- Final val loss: 0.2150833487510681**



**## Generation Testleri**



**Prompt: Bir hata alınca**



**Model, hata alma ve proje hedefleriyle ilgili metinler üretmiştir ancak cümle geçişlerinde karışmalar görülmüştür.**



**Prompt: DarkMind hazır bir model mi**



**Model, hazır model olmadığı ve sıfırdan geliştirildiği fikrini üretebilmiştir ancak bazı cümlelerde tekrar ve anlam karışması vardır.**



**Prompt: Duygusal yapay zeka**



**Model, duygusal yapay zeka güvenliği hakkında daha temiz ve anlamlı bir çıktı üretmiştir.**



**## Değerlendirme**



**Corpus v0.3 ile veri çeşitliliği artmıştır. Tokenizer vocabulary büyümüş ve model parametre sayısı yaklaşık 26.9M seviyesine çıkmıştır.**



**Model hâlâ güçlü bir genelleme yapmamaktadır. Bazı promptlarda corpus parçalarını karıştırarak üretim yapmaktadır. Bunun temel nedeni veri setinin hâlâ küçük olması ve eğitim süresinin kısa tutulmasıdır.**



**Bu deneyin başarısı, DarkMind projesinin artık daha geniş kaynak dosyalarıyla ölçülebilir bir corpus geliştirme döngüsüne sahip olmasıdır.**



**## Sonraki Adım**



**Bir sonraki aşamada eğitim adımı 300'den 1000'e çıkarılarak aynı corpus üzerinde daha uzun eğitim yapılacaktır. Ayrıca generation çıktılarında özel token kırpma ve daha kontrollü sampling ayarları iyileştirilecektir.**

