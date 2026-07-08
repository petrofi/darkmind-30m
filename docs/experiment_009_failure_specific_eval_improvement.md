# Experiment 009 - Failure-Specific Eval Improvement

## Goal

DarkMind eval sistemini daha adil hale getirmek. `darkmind_eval_v01.jsonl` bazı doğru veya kabul edilebilir cevapları sadece bire bir keyword eşleşmesi olmadığı için başarısız sayıyordu. Bu deney, gerçek model hataları ile yanlış negatifleri ayırmayı hedefler.

## Why Pass Rate Plateaued At 58%

Pass rate yaklaşık `%58` civarında takıldı çünkü iki farklı hata türü aynı sepete düştü:

1. Gerçek model hataları: Model yanlış, eksik veya konu dışı cevap veriyor.
2. False negative hatalar: Model doğru anlamı veriyor ama beklenen kelimeyi bire bir kullanmıyor.

Örneğin "Her şeyi biliyor musun?" sorusunda "sadece eğitildiğim veri kadar cevap üretebilirim" kabul edilebilir bir sınırlılık cevabıdır. Ama v01 sadece "sınırlı" kelimesini aradığı için bunu fail sayabiliyordu.

## False Negatives vs Real Failures

False negative örnekleri:

- "sınırlı" yerine "sadece eğitildiğim veri kadar"
- "genelleme" yerine "yeni örneklerde zayıf performans"
- "token sözlüğü" yerine "vocab"
- "birleştir" yerine "kaynak dosyaları ve temiz verileri corpus haline getirir"

Gerçek failure örnekleri:

- Python `append` sorusunda `append` hiç geçmiyor.
- `IndentationError` cevabında `girinti` veya `blok` anlatılmıyor.
- `CPU` ve `GPU` farkı karıştırılıyor.
- `build_dataset_from_raw.py` için `corpus_v3.txt` üretimi söylenmiyor.

## Eval v02 Design

`data/evals/darkmind_eval_v02.jsonl` üç sinyal kullanır:

- `expected_keywords`: v01 ile uyumlu klasik keyword kontrolü.
- `accepted_phrases`: Cevapta kabul edilebilir alternatif ifade varsa pass sinyali.
- `keyword_groups`: Her gruptan en az bir eşleşme bekleyen esnek eş anlamlı kontrolü.

Eval result JSONL artık şunları da yazar:

- `matched_phrases`
- `keyword_groups`
- `matched_keyword_groups`
- `missing_groups`

Bu sayede "kelime tutmadı" ile "kavram gerçekten eksik" ayrımı daha görünür hale gelir.

## Targeted Corrections

`data/raw_collected/qa_pairs/targeted_eval_corrections_v01.txt` belirli zayıf alanlar için elle hedeflenmiş kısa örnekler içerir:

- `append`
- `IndentationError`
- `vocab`
- `CPU vs GPU`
- `build_dataset_from_raw.py`
- `DarkMind ChatGPT gibi mi?`
- `validation loss`
- fallback ve overfitting davranışı

Her örnek için küçük harf ve noktalama varyantları eklenmiştir.

## Expected Result

Eval v02 ile yanlış negatiflerin azalması beklenir. Bu, pass rate'in yapay biçimde şişmesi için değil, gerçekten kabul edilebilir cevapları fail saymamak içindir. Gerçek eksikler hâlâ candidate correction üretimine düşmelidir.

## Limitations

- Keyword ve phrase eşleşmesi hâlâ yüzeyseldir.
- `accepted_phrases` fazla geniş yazılırsa zayıf cevapları geçirebilir.
- `keyword_groups` anlamı tam ölçmez; sadece beklenen kavram ailesini kontrol eder.
- Hedefli correction dosyası küçük ve elle yazılmıştır; genel kod yeteneği için yeterli değildir.
- Model eğitimi yapılmadan bu değişiklikler sadece eval ve dataset hazırlığı seviyesinde etki eder.

## Next Steps

1. `darkmind_eval_v02.jsonl` ile mevcut checkpoint'i değerlendir.
2. Fail olan sonuçlarda `missing_groups` alanını incele.
3. Gerçek hataları pending correction candidate olarak üret.
4. Candidate dosyasını insan incelemesinden geçir.
5. Onaylanan örnekleri data pipeline'a ekle.
6. Corpus'u yeniden oluştur ve tokenizer/model eğitimini manuel çalıştır.
