# Experiment 011 - Turkish Coding Track

## Goal

DarkMind'in Türkçe Python kodlama yeteneğini ölçülebilir ve güvenli şekilde geliştirmek. Hedef; küçük doğru programlar yazabilen, Python hatalarını açıklayabilen, dosya/JSON/sözlük/liste gibi temel konularda Türkçe yardımcı olabilen bir öğrenme odaklı model hattı kurmaktır.

## Why Coding Dataset Is Separate

Kod verisi normal sohbet verisinden farklıdır. Küçük syntax hataları bile modelin kötü alışkanlık edinmesine neden olabilir. Bu yüzden Python örnekleri ayrı üretilir, code block syntax validation ile kontrol edilir ve coding eval dosyasıyla ayrıca ölçülür.

## Generated Files

- `scripts/generate_python_instruction_v02.py`
- `data/raw_collected/python_examples/python_instruction_v02.txt`
- `scripts/validate_python_examples.py`
- `data/evals/darkmind_code_eval_v01.jsonl`
- `data/raw_collected/qa_pairs/targeted_code_corrections_v01.txt`
- `scripts/run_code_eval_cycle.py`

## Validation Method

`scripts/validate_python_examples.py` dataset dosyalarındaki ` ```python ` code blocklarını bulur ve `ast.parse` ile syntax kontrolü yapar. Kod çalıştırılmaz. Bu, güvenli bir ilk kalite kapısıdır.

## Eval Method

`data/evals/darkmind_code_eval_v01.jsonl` Python listeleri, fonksiyonlar, hatalar, dosya okuma, sözlükler, döngüler, class, algoritmalar ve açıklama becerisini ölçer. `eval_model.py` içindeki esnek scoring alanları kullanılır:

- `expected_keywords`
- `accepted_phrases`
- `keyword_groups`

## Current Limitations

- Örnekler Python odaklıdır; farklı diller henüz kapsam dışıdır.
- Syntax validation kodu çalıştırmaz, sadece parse eder.
- Kodların gerçekten doğru çıktıyı üretmesi ayrı test altyapısı gerektirir.
- Dataset deterministik olduğu için bazı kalıplar tekrar edebilir.
- Model eğitilmeden bu dosyalar sadece veri hazırlığı ve eval altyapısı sağlar.

## Next Steps Toward Better Code Generation

1. Python instruction v02 datasetini üret.
2. Code block syntax validation çalıştır.
3. Corpus'u yeniden oluştur.
4. Tokenizer ve modeli manuel eğit.
5. Coding eval ile pass rate ölç.
6. Failed eval sonuçlarından pending correction candidates üret.
7. İnsan incelemesinden geçen correction örneklerini dataset'e ekle.
8. Daha sonra küçük çalıştırılabilir unit test datasetleri için ayrı güvenli doğrulama hattı tasarla.
