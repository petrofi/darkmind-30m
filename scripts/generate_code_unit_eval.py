from pathlib import Path
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "data" / "evals" / "darkmind_code_eval_v02.jsonl"


def item(
    category: str,
    prompt: str,
    expected_keywords: list[str],
    keyword_groups: list[list[str]] | None = None,
    accepted_phrases: list[str] | None = None,
) -> dict:
    return {
        "id": f"{category}_{item.counter:03d}",
        "prompt": prompt,
        "expected_keywords": expected_keywords,
        "accepted_phrases": accepted_phrases or [],
        "keyword_groups": keyword_groups or [[keyword] for keyword in expected_keywords],
        "category": category,
    }


item.counter = 0


def add(items: list[dict], category: str, prompt: str, expected_keywords: list[str], keyword_groups=None, accepted_phrases=None) -> None:
    item.counter += 1
    items.append(item(category, prompt, expected_keywords, keyword_groups, accepted_phrases))


def build_eval_items() -> list[dict]:
    items: list[dict] = []
    rows = {
        "beginner_python": [
            ("Python'da değişken nasıl tanımlanır?", ["değişken"], [["=", "değişken"]]),
            ("print ne işe yarar?", ["print"], [["ekrana", "yazdır"], ["print"]]),
            ("f-string nedir?", ["f-string"], [["f-string"], ["değişken", "metin"]]),
            ("type() ne gösterir?", ["type"], [["type"], ["tip", "tür"]]),
            ("Python'da iki sayı nasıl toplanır?", ["+"], [["+", "topla"]]),
            ("yorum satırı ne işe yarar?", ["#"], [["#", "yorum"]]),
            ("Bilmiyorsan Python sorusunda ne yaparsın?", ["uydurmam"], [["uydurmam", "emin değil"], ["güvenli", "kontrol"]]),
            ("basit bool değişken örneği ver", ["True"], [["True", "False", "bool"]]),
        ],
        "functions": [
            ("iki sayıyı toplayan fonksiyon yaz", ["def", "return"], [["def"], ["return"], ["+", "topla"]]),
            ("return ne işe yarar?", ["return"], [["return"], ["döndürür", "sonuç"]]),
            ("varsayılan parametre nedir?", ["varsayılan"], [["varsayılan", "default"], ["parametre"]]),
            ("fonksiyon ne zaman kullanılır?", ["tekrar"], [["tekrar", "yeniden"], ["okunabilir", "düzenli"]]),
            ("metni küçülten fonksiyon nasıl yazılır?", ["lower"], [["def"], ["lower"]]),
            ("boş liste kontrol eden fonksiyon yaz", ["len"], [["def"], ["len", "not"]]),
            ("ortalama hesaplayan fonksiyon nasıl olur?", ["sum"], [["sum"], ["len"], ["return"]]),
            ("fonksiyon argümanı nedir?", ["parametre"], [["argüman", "parametre"]]),
        ],
        "lists": [
            ("listeye eleman nasıl eklenir?", ["append"], [["append"], ["liste", "eleman"]]),
            ("listeden eleman nasıl silinir?", ["remove"], [["remove"], ["liste"]]),
            ("listedeki çift sayıları filtrele", ["for", "if"], [["for"], ["if"], ["% 2", "çift"]]),
            ("liste nasıl sıralanır?", ["sorted"], [["sorted", "sort"]]),
            ("listenin son elemanı nasıl alınır?", ["-1"], [["-1"], ["liste"]]),
            ("list comprehension nedir?", ["for"], [["for"], ["liste", "comprehension"]]),
            ("len liste için ne verir?", ["len"], [["len"], ["eleman", "uzunluk"]]),
            ("liste index hatası neden olur?", ["index"], [["index"], ["sınır", "uzunluk"]]),
        ],
        "dictionaries": [
            ("sözlükten değer nasıl okunur?", ["anahtar"], [["anahtar", "key"], ["değer"]]),
            ("dict.get ne işe yarar?", ["get"], [["get"], ["varsayılan", "hata"]]),
            ("sözlüğe yeni anahtar nasıl eklenir?", ["="], [["anahtar"], ["="]]),
            ("sözlükte kelime sayacı nasıl tutulur?", ["get"], [["get"], ["sayaç", "count"]]),
            ("items() ne işe yarar?", ["items"], [["items"], ["anahtar", "değer"]]),
            ("KeyError nasıl önlenir?", ["get"], [["get", "in"], ["KeyError", "anahtar"]]),
            ("sözlük anahtarları nasıl gezilir?", ["keys"], [["keys", "for"]]),
            ("sözlük değerleri nasıl gezilir?", ["values"], [["values", "for"]]),
        ],
        "strings": [
            ("strip ne işe yarar?", ["strip"], [["strip"], ["boşluk", "temiz"]]),
            ("split ile metin nasıl bölünür?", ["split"], [["split"], ["kelime", "liste"]]),
            ("join ne işe yarar?", ["join"], [["join"], ["birleştir"]]),
            ("lower ve upper ne yapar?", ["lower"], [["lower"], ["upper"]]),
            ("metin içinde kelime nasıl aranır?", ["in"], [["in"], ["metin", "kelime"]]),
            ("metnin uzunluğu nasıl bulunur?", ["len"], [["len"], ["uzunluk"]]),
            ("f-string ile değişken nasıl yazılır?", ["f-string"], [["f-string", "f\""]]),
            ("Türkçe metin temizlerken neye dikkat edilir?", ["Türkçe"], [["Türkçe"], ["karakter", "bozmamak"]]),
        ],
        "loops": [
            ("for döngüsü ne işe yarar?", ["for"], [["for"], ["tekrar", "eleman"]]),
            ("range ile 1'den 5'e kadar yazdır", ["range"], [["range"], ["for"]]),
            ("while döngüsü ne zaman kullanılır?", ["while"], [["while"], ["koşul"]]),
            ("enumerate ne işe yarar?", ["enumerate"], [["enumerate"], ["index", "sıra"]]),
            ("break ne yapar?", ["break"], [["break"], ["döngü", "çık"]]),
            ("continue ne yapar?", ["continue"], [["continue"], ["atla"]]),
            ("sonsuz döngü nasıl önlenir?", ["koşul"], [["koşul"], ["değiş", "güncelle"]]),
            ("listedeki toplam döngüyle nasıl bulunur?", ["total"], [["for"], ["total", "toplam"]]),
        ],
        "conditionals": [
            ("if else ne işe yarar?", ["if"], [["if"], ["else"], ["koşul"]]),
            ("elif ne zaman kullanılır?", ["elif"], [["elif"], ["birden fazla", "koşul"]]),
            ("çift sayı kontrolü nasıl yapılır?", ["% 2"], [["% 2"], ["if"]]),
            ("boş liste kontrolü nasıl yapılır?", ["if"], [["if"], ["liste", "boş"]]),
            ("in operatörü koşulda nasıl kullanılır?", ["in"], [["in"], ["if"]]),
            ("koşul okunabilirliği neden önemlidir?", ["okunabilir"], [["okunabilir"], ["hata"]]),
            ("not operatörü ne yapar?", ["not"], [["not"], ["ters"]]),
            ("and ve or farkı nedir?", ["and"], [["and"], ["or"]]),
        ],
        "files": [
            ("pathlib ile dosya nasıl okunur?", ["Path"], [["Path"], ["read_text"]]),
            ("pathlib ile dosya nasıl yazılır?", ["write_text"], [["Path"], ["write_text"]]),
            ("encoding neden belirtilir?", ["utf-8"], [["utf-8"], ["Türkçe"]]),
            ("dosya yoksa nasıl kontrol edilir?", ["exists"], [["exists"], ["Path"]]),
            ("splitlines ne işe yarar?", ["splitlines"], [["splitlines"], ["satır"]]),
            ("dosya yolunu güvenli nasıl birleştirirsin?", ["Path"], [["Path"], ["/"]]),
            ("FileNotFoundError nasıl yakalanır?", ["FileNotFoundError"], [["try"], ["FileNotFoundError"]]),
            ("dosya karakter sayısı nasıl bulunur?", ["len"], [["read_text"], ["len"]]),
        ],
        "json": [
            ("json.loads ne yapar?", ["loads"], [["json.loads", "loads"], ["sözlük", "dict"]]),
            ("json.dumps ne yapar?", ["dumps"], [["json.dumps", "dumps"], ["metin"]]),
            ("ensure_ascii=False neden kullanılır?", ["ensure_ascii"], [["ensure_ascii"], ["Türkçe"]]),
            ("JSONDecodeError nasıl yakalanır?", ["JSONDecodeError"], [["try"], ["JSONDecodeError"]]),
            ("JSON dosyası nasıl yazılır?", ["json.dumps"], [["json.dumps"], ["write_text"]]),
            ("JSON dosyası nasıl okunur?", ["json.loads"], [["read_text"], ["json.loads"]]),
            ("indent=2 ne işe yarar?", ["indent"], [["indent"], ["okunabilir"]]),
            ("JSON ile dict farkı nedir?", ["JSON"], [["JSON"], ["sözlük", "dict"]]),
        ],
        "errors": [
            ("TypeError neden olur?", ["tip"], [["TypeError"], ["tip", "uyumsuz"]]),
            ("ValueError neden olur?", ["değer"], [["ValueError"], ["değer", "format"]]),
            ("IndentationError nasıl çözülür?", ["girinti"], [["IndentationError"], ["girinti"]]),
            ("FileNotFoundError neden olur?", ["dosya"], [["FileNotFoundError"], ["dosya", "yol"]]),
            ("try except ne işe yarar?", ["try"], [["try"], ["except"], ["hata"]]),
            ("finally ne zaman çalışır?", ["finally"], [["finally"], ["her durumda"]]),
            ("NameError nasıl çözülür?", ["tanımla"], [["NameError"], ["tanımla", "değişken"]]),
            ("hata mesajından emin değilsen ne yaparsın?", ["uydurmam"], [["uydurmam", "emin değil"], ["hata mesajı"]]),
        ],
        "classes": [
            ("__init__ ne işe yarar?", ["__init__"], [["__init__"], ["başlangıç"]]),
            ("basit sınıf nasıl yazılır?", ["class"], [["class"], ["__init__"]]),
            ("self neyi ifade eder?", ["self"], [["self"], ["nesne"]]),
            ("metot nedir?", ["metot"], [["metot", "method"], ["sınıf"]]),
            ("__str__ ne işe yarar?", ["__str__"], [["__str__"], ["metin"]]),
            ("sınıf ne zaman kullanılır?", ["davranış"], [["veri"], ["davranış"]]),
            ("nesne nasıl oluşturulur?", ["class"], [["class"], ["nesne"]]),
            ("Rectangle alan metodu nasıl yazılır?", ["area"], [["class"], ["area"]]),
        ],
        "algorithms": [
            ("en büyük sayı nasıl bulunur?", ["max"], [["max"], ["liste"]]),
            ("kelime frekansı nasıl hesaplanır?", ["Counter"], [["Counter"], ["split"]]),
            ("lineer arama nedir?", ["sırayla"], [["sırayla"], ["arama"]]),
            ("toplam döngüyle nasıl hesaplanır?", ["total"], [["for"], ["total"]]),
            ("sorted ile sıralama nasıl yapılır?", ["sorted"], [["sorted"]]),
            ("algoritma açıklarken neye dikkat edilir?", ["adım"], [["adım"], ["basit"]]),
            ("O(n) basitçe ne demektir?", ["girdi"], [["girdi"], ["artar"]]),
            ("liste filtreleme algoritması nasıl yazılır?", ["if"], [["for"], ["if"]]),
        ],
        "pytorch": [
            ("torch tensor nasıl oluşturulur?", ["torch.tensor"], [["torch.tensor"], ["tensor"]]),
            ("CUDA kullanılabilir mi nasıl kontrol edilir?", ["torch.cuda.is_available"], [["torch.cuda.is_available"], ["cuda"]]),
            ("tensor shape ne gösterir?", ["shape"], [["shape"], ["boyut"]]),
            ("tensor toplama nasıl yapılır?", ["+"], [["torch.tensor"], ["+"]]),
            ("PyTorch device seçimi nasıl yapılır?", ["device"], [["cuda"], ["cpu"], ["device"]]),
            ("zeros ile tensor nasıl oluşturulur?", ["zeros"], [["torch.zeros", "zeros"]]),
            ("PyTorch bilmediğin konuda ne yaparsın?", ["emin değil"], [["emin değil", "uydurmam"], ["kontrol"]]),
            ("tensor ile liste farkı nedir?", ["tensor"], [["tensor"], ["liste"]]),
        ],
        "tokenizer": [
            ("tokenizer nedir?", ["Tokenizer"], [["Tokenizer", "tokenizer"], ["metin", "id"]]),
            ("ByteLevelBPETokenizer nasıl yüklenir?", ["ByteLevelBPETokenizer"], [["ByteLevelBPETokenizer"], ["vocab", "merges"]]),
            ("encode ne işe yarar?", ["encode"], [["encode"], ["ids", "id"]]),
            ("decode ne işe yarar?", ["decode"], [["decode"], ["metin"]]),
            ("tokenizer mismatch neden sorun olur?", ["vocab"], [["vocab"], ["checkpoint", "uyum"]]),
            ("vocab.json ne içerir?", ["vocab"], [["vocab"], ["id"]]),
            ("merges.txt ne işe yarar?", ["merges"], [["merges"], ["BPE"]]),
            ("Türkçe tokenizer için ne önemlidir?", ["Türkçe"], [["Türkçe"], ["karakter"]]),
        ],
        "git": [
            ("git status ne işe yarar?", ["status"], [["status"], ["değişiklik"]]),
            ("git add ne yapar?", ["add"], [["add"], ["staging"]]),
            ("git commit ne yapar?", ["commit"], [["commit"], ["kayıt"]]),
            ("git push ne yapar?", ["push"], [["push"], ["uzak"]]),
            ("commit mesajı nasıl olmalı?", ["açık"], [["açık"], ["kısa"]]),
            ("git branch neden kontrol edilir?", ["branch"], [["branch"], ["dal"]]),
            ("yıkıcı git komutlarından önce ne yapılır?", ["onay"], [["onay"], ["status"]]),
            ("git diff ne gösterir?", ["diff"], [["diff"], ["değişiklik"]]),
        ],
        "project_workflow": [
            ("DarkMind veri pipeline sırası nedir?", ["clean_text"], [["clean_text"], ["build_dataset"], ["quality"]]),
            ("corpus build etmeden önce ne yapılır?", ["temiz"], [["temiz"], ["validate", "kontrol"]]),
            ("eval sonucu neden ölçülür?", ["pass rate"], [["pass rate", "başarı"], ["ölç"]]),
            ("checkpoint ne zaman promote edilir?", ["threshold"], [["threshold", "eşik"], ["eval"]]),
            ("model eğitmeden önce hangi kontrol yapılır?", ["dataset"], [["dataset"], ["kalite"]]),
            ("DarkMind bilmediği soruda ne demeli?", ["bilmiyorum"], [["bilmiyorum", "emin değil"], ["uydurmam"]]),
            ("README komutları neden güncel olmalı?", ["komut"], [["komut"], ["tekrarlanabilir"]]),
            ("deney dokümanı neden tutulur?", ["deney"], [["deney"], ["sonuç", "izleme"]]),
        ],
    }

    for category, prompts in rows.items():
        for prompt, keywords, groups in prompts:
            add(items, category, prompt, keywords, groups)

    return items


def main() -> None:
    items = build_eval_items()

    if len(items) < 120:
        raise ValueError(f"Expected at least 120 eval items, got {len(items)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for row in items:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 70)
    print(f"Generated eval items: {len(items):,}")
    print(f"Output path: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
