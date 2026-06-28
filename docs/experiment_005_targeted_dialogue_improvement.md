**# Experiment 005 — Targeted Dialogue Improvement**



**## Amaç**



**Bu deneyin amacı, DarkMind modelinin bazı teknik kavramları karıştırdığı durumları tespit etmek ve hedefli soru-cevap verisi ekleyerek bu davranışı düzeltmektir.**



**Özellikle şu karışıklıklar gözlemlenmiştir:**



**\* CUDA ile tokenizer kavramlarının karışması**

**\* Overfitting sorusunda causal mask cevabına kayması**



**## Başlangıç Problemi**



**Model, düşük temperature ve düşük top-k değerleriyle test edildiğinde bile bazı sorularda yanlış konuya kaymıştır.**



**Örnek sorunlu testler:**



**```txt**

**Kullanıcı: CUDA ile tokenizer aynı şey mi?**

**Asistan: Hayır. Darklove daha çok ürün ve duygusal yapay zeka tarafını temsil eder...**



**Kullanıcı: Overfitting nedir?**

**Asistan: Causal mask, modelin gelecekteki tokenları görmesini engeller...**

**```**



**Bu sonuç, modelin bazı yakın formatlı teknik soru-cevap bloklarını karıştırdığını göstermiştir.**



**## Yapılan Değişiklik**



**`data/sources/mini\_dialogues\_tr.txt` dosyasına hedefli ve karşılaştırmalı soru-cevap örnekleri eklenmiştir.**



**Eklenen veri türleri:**



**\* CUDA nedir?**

**\* Tokenizer nedir?**

**\* CUDA ile tokenizer aynı şey midir?**

**\* CUDA ile tokenizer arasındaki fark nedir?**

**\* Overfitting nedir?**

**\* Overfitting ile causal mask aynı şey midir?**

**\* Loss düşükse model kesin iyi midir?**

**\* Model neden bazı sorularda yanlış konuya kayar?**



**Bu yaklaşım, modele sadece doğru cevabı değil, kavramlar arasındaki farkı da öğretmeyi amaçlamıştır.**



**## Corpus v0.4 İstatistikleri**



**\* Source files: 10**

**\* Characters: 32,907**

**\* Lines: 537**

**\* Non-empty lines: 420**

**\* Tokens: 5,741**

**\* Tokenizer vocab size: 3,314**



**`mini\_dialogues\_tr.txt` dosyası bu deneyde önemli ölçüde büyümüştür:**



**\* mini\_dialogues\_tr.txt characters: 10,188**

**\* mini\_dialogues\_tr.txt non-empty lines: 100**

**\* mini\_dialogues\_tr.txt tokens: 1,822**



**## Model Eğitimi**



**Kullanılan config:**



**```txt**

**configs/darkmind\_30m\_1000step.json**

**```**



**Model bilgileri:**



**\* Run name: darkmind\_30m\_1000step**

**\* Device: cuda**

**\* GPU: NVIDIA GeForce RTX 4060 Laptop GPU**

**\* Model parameters: 27,047,936**

**\* Vocab size: 3,314**

**\* Block size: 256**

**\* Layers: 8**

**\* Heads: 8**

**\* Embedding size: 512**

**\* Batch size: 8**

**\* Max steps: 1000**



**Final sonuçlar:**



**\* Final train loss: 0.03320951610803604**

**\* Final val loss: 0.02507846988737583**



**## Test Sonuçları**



**Test komutu:**



**```powershell**

**python scripts/generate\_from\_checkpoint.py --checkpoint checkpoints/darkmind\_30m\_1000step.pt --dialogue "CUDA ile tokenizer aynı şey mi?" --temperature 0.2 --top\_k 5 --max\_new\_tokens 80**

**```**



**Çıktı:**



**```txt**

**Kullanıcı: CUDA ile tokenizer aynı şey mi?**

**Asistan: Hayır. CUDA hesaplama hızlandırma teknolojisidir. Tokenizer ise metni modelin anlayabileceği token parçalarına dönüştürür.**

**```**



**Test komutu:**



**```powershell**

**python scripts/generate\_from\_checkpoint.py --checkpoint checkpoints/darkmind\_30m\_1000step.pt --dialogue "Overfitting nedir?" --temperature 0.2 --top\_k 5 --max\_new\_tokens 80**

**```**



**Çıktı:**



**```txt**

**Kullanıcı: Overfitting nedir?**

**Asistan: Overfitting, modelin veriyi anlamak yerine ezberlemesi durumudur. Böyle bir model eğitim verisinde iyi görünür ama yeni sorularda zayıf cevap verebilir.**

**```**



**## Değerlendirme**



**Bu deney başarılı olmuştur.**



**Önce modelin yanlış cevap verdiği alanlar tespit edilmiştir. Daha sonra bu alanlara özel hedefli ve karşılaştırmalı veri eklenmiştir. Yeniden tokenizer eğitimi ve model eğitimi sonrasında test edilen iki kavram da doğru cevaplanmıştır.**



**Bu deney, küçük bir dil modelinde hedefli veri iyileştirmenin model davranışını değiştirebildiğini göstermektedir.**



**## Not**



**Bu sonuçlar modelin genel olarak güçlü bir dil modeli olduğu anlamına gelmez. Corpus hâlâ küçüktür ve model birçok konuda genelleme yapmakta zorlanabilir. Ancak bu deney, veri odaklı iyileştirme döngüsünün çalıştığını göstermesi açısından önemlidir.**



**## Sonraki Adım**



**Bir sonraki aşamada:**



**\* Daha fazla teknik mini diyalog eklenecek**

**\* Soru-cevap formatı genişletilecek**

**\* DarkMind için küçük bir demo CLI hazırlanacak**

**\* Kullanıcıdan soru alıp dialogue mode ile cevap veren basit bir terminal arayüzü geliştirilecek**



