# Experiment 008 - Self-Improvement Pipeline

## Goal

DarkMind için güvenli bir self-improvement akışı kurmak. Amaç, modelin zayıf cevaplarını değerlendirme promptlarıyla bulmak, bu zayıflıklar için kural tabanlı düzeltme adayları üretmek ve bu adayları insan incelemesine bırakmaktır.

## Why Blind Self-Training Is Dangerous

Bir modelin kendi ürettiği cevapları doğrudan eğitim verisine eklemek hataları büyütebilir. Yanlış bilgi, uydurma açıklamalar, bozuk format ve güvenli olmayan öneriler sonraki eğitimlerde daha güçlü hale gelebilir. Bu yüzden DarkMind kendi çıktılarını otomatik olarak doğru kabul etmez.

## Human-In-The-Loop Design

1. `scripts/eval_model.py` eval promptlarını çalıştırır.
2. Model cevapları JSONL run dosyasına kaydedilir.
3. Basit keyword kurallarıyla zayıf cevaplar işaretlenir.
4. `scripts/generate_correction_candidates.py` başarısız örnekler için kısa ve güvenli aday cevaplar üretir.
5. Adaylar `data/self_improvement/pending_review/` altında bekler.
6. İnsan adayları okur, düzeltir veya reddeder.
7. Sadece açık onaydan sonra `scripts/approve_candidates.py --approve_all` ile raw dataset'e eklenir.
8. Corpus yeniden oluşturulur.
9. Tokenizer ve model eğitimi manuel çalıştırılır.

## Files Added

- `data/evals/darkmind_eval_v01.jsonl`
- `data/self_improvement/runs/`
- `data/self_improvement/pending_review/`
- `data/self_improvement/approved/`
- `data/self_improvement/rejected/`
- `scripts/eval_model.py`
- `scripts/generate_correction_candidates.py`
- `scripts/approve_candidates.py`
- `scripts/self_improve_loop.py`

## Eval Format

Eval dosyası JSONL formatındadır. Her satır bir değerlendirme maddesidir:

```json
{"id":"tokenizer_001","prompt":"Tokenizer nedir?","expected_keywords":["metni","token","model"],"category":"tokenizer"}
```

Eval sonucu da JSONL olarak kaydedilir:

```json
{"id":"tokenizer_001","prompt":"Tokenizer nedir?","answer":"...","expected_keywords":["metni","token","model"],"matched_keywords":["token"],"missing_keywords":["metni","model"],"passed":false,"category":"tokenizer"}
```

## Correction Candidate Format

Correction adayları düz metin olarak üretilir:

```txt
These are candidate examples. Review before adding to training data.

Kullanıcı: Tokenizer nedir?
Asistan: Tokenizer, metni modelin işleyebileceği tokenlara ve token ID'lerine dönüştüren bileşendir.
```

Bu dosyalar doğrudan eğitim verisi sayılmaz. Önce insan tarafından incelenmelidir.

## How To Run

Eval ve aday üretimini ayrı ayrı çalıştır:

```powershell
python scripts/eval_model.py --checkpoint checkpoints/darkmind_30m.pt
python scripts/generate_correction_candidates.py
```

Veya convenience loop kullan:

```powershell
python scripts/self_improve_loop.py --checkpoint checkpoints/darkmind_30m.pt
```

İnceledikten sonra onayla:

```powershell
python scripts/approve_candidates.py --input_path data/self_improvement/pending_review/correction_candidates_YYYYMMDD_HHMMSS.txt --approve_all
```

Corpus'u yeniden kur:

```powershell
python scripts/clean_text.py
python scripts/build_dataset_from_raw.py
python scripts/dataset_quality_check.py --path data/processed/corpus_v3.txt
```

## Limitations

- Keyword tabanlı eval yüzeyseldir; doğru cevabı kaçırabilir veya kötü cevabı geçirebilir.
- Correction adayları kural tabanlıdır; bağlama özel mükemmel cevap üretmez.
- İnsan incelemesi şarttır.
- Bu sistem gerçek genel zeka veya otomatik araştırma yeteneği sağlamaz.
- Tokenizer ve model eğitimi hâlâ manuel kalite kontrol gerektirir.

## Next Steps

1. Eval dosyasını daha fazla kategori ve daha gerçekçi prompt ile genişlet.
2. Candidate dosyalarını düzenlemek için küçük bir review checklist ekle.
3. Approved ve rejected kararlarını ayrı kayıt dosyalarında tut.
4. Eval pass rate'i eğitim koşuları arasında karşılaştır.
5. Kötü cevap tiplerine göre daha hedefli dataset generatorları ekle.
