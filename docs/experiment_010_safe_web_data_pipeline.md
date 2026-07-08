# Experiment 010 - Safe Web Data Pipeline

## Goal

DarkMind için kontrollü, allowlist tabanlı ve insan incelemeli bir web verisi toplama hattı kurmak. Amaç rastgele internet kazımak değil; lisansı ve kaynağı izlenebilir, Türkçe odaklı ve kaliteli metinleri zamanla güvenli şekilde dataset hattına eklemektir.

## Why Internet Data Needs Control

İnternet verisi karışık, gürültülü ve hukuki açıdan risklidir. Rastgele web scraping; telifli metinleri, login/cookie/banner gürültüsünü, reklamları, düşük kaliteli sayfaları ve yanlış bilgileri eğitim verisine taşıyabilir. Küçük bir modelde bu hatalar daha da belirgin hale gelebilir.

## Allowlist Design

Kaynaklar `configs/web_sources_allowlist.json` içinde tanımlanır:

- `name`: kaynak adı
- `base_url`: izin verilen alan adı
- `allowed_paths`: izin verilen path kapsamı
- `license`: lisans bilgisi
- `usage_status`: örneğin `pending_review` veya `approved`
- `notes`: insan notları

`scripts/check_web_source_policy.py` URL'nin allowlist içinde olup olmadığını kontrol eder ve sayfayı fetch etmez.

## Pending Review Flow

1. URL allowlist'e eklenir.
2. Lisans ve kullanım amacı insan tarafından kontrol edilir.
3. `scripts/fetch_web_text.py` tek sayfayı çeker, text dosyasını `pending_review` altına yazar.
4. Metadata JSON `metadata` klasörüne yazılır.
5. İnsan text ve metadata dosyasını inceler.
6. Sadece uygunsa `scripts/approve_web_text.py --approve_all` ile approved dosyasına eklenir.
7. `scripts/web_data_quality_check.py` approved web datasını inceler.
8. Normal `clean_text.py` ve `build_dataset_from_raw.py` hattı çalıştırılır.

`clean_text.py` pending review ve rejected web text dosyalarını otomatik işlemez.

## Licensing Limitations

Bir sayfanın herkesçe erişilebilir olması eğitim için kullanılabileceği anlamına gelmez. Kaynak lisansı, kullanım şartları ve robots.txt dikkate alınmalıdır. Lisans belirsizse veri `pending_review` olarak kalmalı ve eğitim corpus'una eklenmemelidir.

## Why Reddit Is Excluded

Reddit verisi kullanıcı üretimli, bağlamı karışık ve lisans/kullanım açısından sorunlu olabilir. Bu pipeline Reddit'i hedeflemez ve eğitim verisi için Reddit datası kullanmaz.

## Why Web Data Alone Will Not Make The Model Intelligent

Web verisi miktarı artsa bile model otomatik olarak güvenilir veya zeki hale gelmez. Veri temizliği, lisans uygunluğu, tekrar kontrolü, hedefli eval, model kapasitesi ve eğitim ayarları hâlâ belirleyicidir. Bu pipeline sadece daha kontrollü veri toplama altyapısı sağlar.

## Next Steps For Turkish Open Data Sources

1. Türkçe ve açık lisanslı kaynak adaylarını listele.
2. Her kaynak için lisans ve kullanım koşullarını not et.
3. Sadece izinli pathleri allowlist'e ekle.
4. Küçük batchlerle fetch yap.
5. Metadata ve kalite kontrol raporlarını sakla.
6. Approved web text için tekrar ve boilerplate temizliğini iyileştir.
7. Eval sonuçlarıyla web datasının gerçekten fayda sağlayıp sağlamadığını ölç.
