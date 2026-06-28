# Experiment 007 - Dataset Generation v01

## Goal

DarkMind'in veri setini elle küçük dosyalar düzenleme aşamasından daha tekrarlanabilir bir akışa taşımak. Bu deneyde internetten veri çekilmez; bunun yerine deterministik, Türkçe odaklı ve okunabilir template tabanlı örnekler üretilir.

## Generated Files

- `data/raw_collected/python_examples/python_examples_v01.txt`
- `data/raw_collected/coding_notes/coding_error_examples_v01.txt`
- `data/raw_collected/qa_pairs/generated_qa_variants_v01.txt`

Bu dosyalar `scripts/generate_dataset_v01.py` ile tek komutta üretilebilir. Ardından `scripts/clean_text.py` ve `scripts/build_dataset_from_raw.py` çalıştırılarak `data/processed/corpus_v3.txt` yeniden oluşturulur.

## Expected Improvement

- Basit Python görevlerinde daha tutarlı cevap formatı.
- Kod bloklarını `Asistan:` sonrasında üretme eğilimi.
- Sık görülen Python, Git, CUDA, checkpoint ve Windows path hatalarında daha kısa ve doğru açıklamalar.
- DarkMind kimliği, tokenizer, Transformer, corpus, checkpoint ve fallback davranışı gibi temel kavramlarda daha kararlı cevaplar.
- Küçük model için daha hedefli ve tekrar üretilebilir eğitim verisi.

## Limitations

- Veriler template tabanlıdır; gerçek dünyadaki çeşitliliği tam temsil etmez.
- Örnekler kısa ve öğreticidir; büyük projelerde gereken karmaşık kod muhakemesini öğretmez.
- Çok tekrar eden kalıplar overfitting riskini artırabilir.
- Üretilen veri internetten doğrulanmış güncel bilgi içermez.
- Modelin cevap kalitesi hâlâ tokenizer, model kapasitesi, eğitim süresi ve veri temizliğiyle sınırlıdır.

## Why This Is Not Real General Intelligence

Bu deney modele gerçek genel zeka kazandırmaz. Sadece belirli prompt-cevap kalıplarını, temel Python örneklerini ve proje kavramlarını daha sık görmesini sağlar. Model hâlâ eğitim verisindeki örüntülere göre token tahmin eder; anlamayı, doğrulamayı veya bağımsız akıl yürütmeyi garanti etmez.

## Next Steps

1. Generator çıktısını kalite kontrolünden geçir.
2. `corpus_v3.txt` içinde tekrar eden satırları ve zayıf örnekleri incele.
3. Tokenizer'ı `corpus_v3.txt` ile yeniden eğit.
4. `configs/darkmind_30m_1000step.json` ile kısa bir eğitim koşusu yap.
5. Chat demo üzerinde aynı test sorularını önceki checkpoint ile karşılaştır.
6. Kod verisi için daha fazla güvenli, küçük ve açıklamalı Python örneği ekle.
