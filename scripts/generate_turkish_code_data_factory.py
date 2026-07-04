from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "python_examples"
    / "turkish_code_factory_v01.txt"
)

PROMPT_VARIANTS = [
    "{task}.",
    "Kısa bir örnekle {task}.",
    "Yeni başlayan biri için {task}.",
    "Türkçe açıklamayla {task}.",
    "DarkMind eğitim verisi için {task}.",
    "Hata yapmadan basit şekilde {task}.",
    "Kodun okunabilir olmasına dikkat ederek {task}.",
    "Bunu küçük bir Python örneğiyle göster: {task}.",
    "Bilmiyorsan ne yaparsın? Yine de güvenli şekilde {task}.",
]

DIFFICULTY_BY_BUCKET = {
    "beginner_python": "easy",
    "functions": "easy",
    "lists": "easy",
    "dictionaries": "easy",
    "strings": "easy",
    "loops": "easy",
    "conditionals": "easy",
    "file_operations": "medium",
    "json_operations": "medium",
    "error_handling": "medium",
    "classes": "medium",
    "algorithms": "medium",
    "debugging": "medium",
    "project_workflow": "easy",
    "pytorch_basics": "medium",
    "tokenizer_basics": "medium",
    "cli_tools": "medium",
    "git_basics": "easy",
}


def code_answer(code: str, explanation: str) -> str:
    return f"```python\n{code.strip()}\n```\n\n{explanation.strip()}"


def text_answer(text: str) -> str:
    return text.strip()


def concept(task: str, answer: str) -> dict:
    return {
        "task": task.strip(),
        "answer": answer.strip(),
    }


def build_bucket_concepts() -> dict[str, list[dict]]:
    return {
        "beginner_python": [
            concept(
                "bir değişken tanımla ve ekrana yazdır",
                code_answer(
                    """
name = "DarkMind"
print(name)
""",
                    "Değişken, bir değeri isimle saklamanı sağlar. print() değeri ekrana yazar.",
                ),
            ),
            concept(
                "iki sayıyı topla ve sonucu göster",
                code_answer(
                    """
a = 12
b = 8
total = a + b
print(total)
""",
                    "Toplama için + operatörü kullanılır. Sonucu ayrı bir değişkende tutmak kodu okunabilir yapar.",
                ),
            ),
            concept(
                "f-string ile kısa bir mesaj oluştur",
                code_answer(
                    """
model_name = "DarkMind"
step = 1000
message = f"{model_name} {step} adım eğitildi."
print(message)
""",
                    "f-string, değişkenleri metnin içine temiz biçimde yerleştirir.",
                ),
            ),
            concept(
                "type() ile bir değerin tipini kontrol et",
                code_answer(
                    """
value = 42
print(type(value))
""",
                    "type() fonksiyonu bir değerin hangi Python tipinde olduğunu gösterir.",
                ),
            ),
            concept(
                "emin olmadığın bir Python sorusunda nasıl cevap vermen gerektiğini açıkla",
                text_answer(
                    "Kesin bilmiyorsam uydurmamalıyım. Önce sınırlı olduğumu söylemeli, sonra güvenli ve basit bir örnekle doğrulanabilir açıklama yapmalıyım.",
                ),
            ),
        ],
        "functions": [
            concept(
                "iki sayıyı toplayan fonksiyon yaz",
                code_answer(
                    """
def add_numbers(a, b):
    return a + b
""",
                    "Fonksiyon, tekrar kullanılabilir kod parçasıdır. return sonucu çağıran yere döndürür.",
                ),
            ),
            concept(
                "bir metni küçük harfe çeviren fonksiyon yaz",
                code_answer(
                    """
def normalize_text(text):
    return text.lower()
""",
                    "lower() metni küçük harfe çevirir. Bu, basit metin normalizasyonunda kullanılır.",
                ),
            ),
            concept(
                "listedeki sayıların ortalamasını hesaplayan fonksiyon yaz",
                code_answer(
                    """
def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
""",
                    "Boş liste kontrolü yapılırsa sıfıra bölme hatası önlenir.",
                ),
            ),
            concept(
                "varsayılan parametre kullanan selamlama fonksiyonu yaz",
                code_answer(
                    """
def greet(name="DarkMind"):
    return f"Merhaba {name}"
""",
                    "Varsayılan parametre, değer verilmediğinde kullanılacak güvenli bir seçenek sağlar.",
                ),
            ),
            concept(
                "fonksiyonun ne zaman kullanılacağını Türkçe açıkla",
                text_answer(
                    "Aynı işlemi birden fazla yerde yapıyorsan fonksiyon kullanmak iyi olur. Böylece kod daha okunabilir, test edilebilir ve tekrar kullanılabilir hale gelir.",
                ),
            ),
        ],
        "lists": [
            concept(
                "listeye append ile eleman ekle",
                code_answer(
                    """
numbers = [1, 2, 3]
numbers.append(4)
print(numbers)
""",
                    "append() listenin sonuna yeni bir eleman ekler.",
                ),
            ),
            concept(
                "listeden remove ile eleman sil",
                code_answer(
                    """
fruits = ["elma", "armut", "muz"]
fruits.remove("armut")
print(fruits)
""",
                    "remove() verilen değeri listeden siler. Değer yoksa hata oluşabilir.",
                ),
            ),
            concept(
                "listedeki çift sayıları list comprehension ile filtrele",
                code_answer(
                    """
numbers = [1, 2, 3, 4, 5, 6]
evens = [number for number in numbers if number % 2 == 0]
print(evens)
""",
                    "Koşullu list comprehension, kısa filtreleme işlemleri için okunabilir bir çözümdür.",
                ),
            ),
            concept(
                "listeyi sorted ile sırala",
                code_answer(
                    """
scores = [80, 95, 70]
sorted_scores = sorted(scores)
print(sorted_scores)
""",
                    "sorted() yeni bir sıralı liste döndürür; orijinal listeyi değiştirmez.",
                ),
            ),
            concept(
                "liste indeks hatasına karşı nasıl dikkatli olunacağını açıkla",
                text_answer(
                    "Bir indekse erişmeden önce listenin uzunluğunu kontrol etmek gerekir. Emin değilsen len(items) ile sınırı görmeli ve olmayan indeksi okumamalısın.",
                ),
            ),
        ],
        "dictionaries": [
            concept(
                "basit bir sözlük oluştur ve değer oku",
                code_answer(
                    """
person = {"name": "Ayşe", "city": "İstanbul"}
print(person["name"])
""",
                    "Sözlükler anahtar-değer yapısıyla veri saklar.",
                ),
            ),
            concept(
                "get ile güvenli sözlük okuma yap",
                code_answer(
                    """
settings = {"theme": "dark"}
language = settings.get("language", "tr")
print(language)
""",
                    "get() anahtar yoksa hata vermek yerine varsayılan değer döndürür.",
                ),
            ),
            concept(
                "sözlükte kelime sayacı oluştur",
                code_answer(
                    """
words = ["veri", "model", "veri"]
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
print(counts)
""",
                    "Bu kalıp, metin işleme ve frekans sayımı için çok kullanışlıdır.",
                ),
            ),
            concept(
                "sözlük anahtar ve değerlerini döngüyle yazdır",
                code_answer(
                    """
scores = {"Ali": 80, "Ayşe": 95}
for name, score in scores.items():
    print(name, score)
""",
                    "items() hem anahtarı hem değeri birlikte döndürür.",
                ),
            ),
            concept(
                "KeyError hatasını nasıl önleyeceğini açıkla",
                text_answer(
                    "Anahtarın varlığından emin değilsen doğrudan dict[key] kullanmak yerine key in dict kontrolü veya get() kullanmalısın.",
                ),
            ),
        ],
        "strings": [
            concept(
                "metni strip ile temizle",
                code_answer(
                    """
text = "  merhaba  "
clean_text = text.strip()
print(clean_text)
""",
                    "strip() baştaki ve sondaki boşlukları temizler.",
                ),
            ),
            concept(
                "metni kelimelere ayır",
                code_answer(
                    """
sentence = "temiz veri önemlidir"
words = sentence.split()
print(words)
""",
                    "split() varsayılan olarak boşluklardan böler.",
                ),
            ),
            concept(
                "kelimeleri tek metin haline getir",
                code_answer(
                    """
words = ["temiz", "veri", "önemlidir"]
sentence = " ".join(words)
print(sentence)
""",
                    "join() liste içindeki metinleri seçilen ayırıcıyla birleştirir.",
                ),
            ),
            concept(
                "metinde belirli kelime var mı kontrol et",
                code_answer(
                    """
text = "Python ile veri işliyorum"
if "Python" in text:
    print("Kelime bulundu")
""",
                    "in operatörü metin içinde arama yapmak için basit ve okunaklıdır.",
                ),
            ),
            concept(
                "metin normalizasyonunu neden dikkatli yapmak gerektiğini açıkla",
                text_answer(
                    "Metni temizlerken Türkçe karakterleri bozmamak gerekir. Gereksiz boşlukları azaltmak faydalıdır ama anlamlı karakterleri silmek veri kalitesini düşürür.",
                ),
            ),
        ],
        "loops": [
            concept(
                "for döngüsüyle listedeki elemanları yazdır",
                code_answer(
                    """
names = ["Ayşe", "Mehmet", "Zeynep"]
for name in names:
    print(name)
""",
                    "for döngüsü, koleksiyon içindeki elemanları sırayla işler.",
                ),
            ),
            concept(
                "range ile 1'den 5'e kadar say",
                code_answer(
                    """
for number in range(1, 6):
    print(number)
""",
                    "range(1, 6), 1 dahil 6 hariç olacak şekilde sayılar üretir.",
                ),
            ),
            concept(
                "while döngüsüyle sayaç artır",
                code_answer(
                    """
counter = 1
while counter <= 5:
    print(counter)
    counter += 1
""",
                    "while döngüsünde koşul yanlış olana kadar tekrar devam eder.",
                ),
            ),
            concept(
                "enumerate ile sıra numarası kullan",
                code_answer(
                    """
tasks = ["oku", "yaz", "test et"]
for index, task in enumerate(tasks, start=1):
    print(index, task)
""",
                    "enumerate(), elemanla birlikte sıra numarasını da verir.",
                ),
            ),
            concept(
                "sonsuz döngü riskini açıkla",
                text_answer(
                    "while kullanırken koşulun bir noktada değiştiğinden emin olmalısın. Aksi halde program sonsuz döngüde kalabilir.",
                ),
            ),
        ],
        "conditionals": [
            concept(
                "if else ile çift sayı kontrolü yap",
                code_answer(
                    """
number = 10
if number % 2 == 0:
    print("Çift")
else:
    print("Tek")
""",
                    "Koşul doğruysa if bloğu, yanlışsa else bloğu çalışır.",
                ),
            ),
            concept(
                "elif ile not aralığı kontrol et",
                code_answer(
                    """
grade = 72
if grade >= 85:
    print("Çok iyi")
elif grade >= 50:
    print("Geçti")
else:
    print("Kaldı")
""",
                    "elif birden fazla koşulu sırayla kontrol etmek için kullanılır.",
                ),
            ),
            concept(
                "boş liste kontrolü yap",
                code_answer(
                    """
items = []
if items:
    print("Liste dolu")
else:
    print("Liste boş")
""",
                    "Boş liste Python'da False gibi değerlendirilir.",
                ),
            ),
            concept(
                "metin uzunluğuna göre karar ver",
                code_answer(
                    """
text = "DarkMind"
if len(text) >= 5:
    print("Yeterince uzun")
else:
    print("Çok kısa")
""",
                    "len() metnin karakter sayısını verir.",
                ),
            ),
            concept(
                "koşul yazarken neden açık olmak gerektiğini anlat",
                text_answer(
                    "Koşullar okunabilir olmalıdır. Karmaşık koşulları küçük parçalara ayırmak hata yapma ihtimalini azaltır.",
                ),
            ),
        ],
        "file_operations": [
            concept(
                "pathlib ile metin dosyası oku",
                code_answer(
                    """
from pathlib import Path

path = Path("notlar.txt")
text = path.read_text(encoding="utf-8")
print(text)
""",
                    "pathlib, dosya yollarını platformdan bağımsız şekilde yönetmeye yardım eder.",
                ),
            ),
            concept(
                "pathlib ile metin dosyası yaz",
                code_answer(
                    """
from pathlib import Path

path = Path("rapor.txt")
path.write_text("Merhaba DarkMind", encoding="utf-8")
""",
                    "write_text() küçük metin dosyaları yazmak için pratiktir.",
                ),
            ),
            concept(
                "dosya varsa okuyan güvenli örnek yaz",
                code_answer(
                    """
from pathlib import Path

path = Path("notlar.txt")
if path.exists():
    print(path.read_text(encoding="utf-8"))
else:
    print("Dosya bulunamadı")
""",
                    "exists() kontrolü FileNotFoundError riskini azaltır.",
                ),
            ),
            concept(
                "dosya satırlarını liste olarak oku",
                code_answer(
                    """
from pathlib import Path

path = Path("notlar.txt")
lines = path.read_text(encoding="utf-8").splitlines()
print(lines)
""",
                    "splitlines() metni satır satır ayırır.",
                ),
            ),
            concept(
                "dosya işlemlerinde neden encoding belirtmek gerektiğini açıkla",
                text_answer(
                    "Türkçe karakterlerin bozulmaması için dosya okuma ve yazmada encoding=\"utf-8\" belirtmek güvenli bir alışkanlıktır.",
                ),
            ),
        ],
        "json_operations": [
            concept(
                "json.loads ile metni sözlüğe çevir",
                code_answer(
                    """
import json

raw_text = '{"name": "DarkMind", "language": "tr"}'
data = json.loads(raw_text)
print(data["name"])
""",
                    "json.loads() JSON metnini Python nesnesine çevirir.",
                ),
            ),
            concept(
                "json.dumps ile sözlüğü JSON metnine çevir",
                code_answer(
                    """
import json

data = {"name": "DarkMind", "language": "tr"}
raw_text = json.dumps(data, ensure_ascii=False)
print(raw_text)
""",
                    "ensure_ascii=False Türkçe karakterlerin okunabilir kalmasını sağlar.",
                ),
            ),
            concept(
                "JSON dosyasını pathlib ile yaz",
                code_answer(
                    """
from pathlib import Path
import json

data = {"status": "ok", "count": 3}
text = json.dumps(data, ensure_ascii=False, indent=2)
Path("report.json").write_text(text, encoding="utf-8")
""",
                    "indent=2 JSON çıktısını daha okunabilir yapar.",
                ),
            ),
            concept(
                "JSONDecodeError durumunu güvenli yakala",
                code_answer(
                    """
import json

raw_text = "{bad json"
try:
    data = json.loads(raw_text)
except json.JSONDecodeError:
    data = {}
print(data)
""",
                    "Bozuk JSON gelebileceği durumlarda try/except kullanmak daha güvenlidir.",
                ),
            ),
            concept(
                "JSON ile Python sözlüğü arasındaki farkı açıkla",
                text_answer(
                    "JSON bir metin veri formatıdır. Python sözlüğü ise program içinde kullanılan veri yapısıdır. json.loads ve json.dumps bu iki biçim arasında dönüşüm yapar.",
                ),
            ),
        ],
        "error_handling": [
            concept(
                "ValueError için try except örneği yaz",
                code_answer(
                    """
text = "42"
try:
    number = int(text)
    print(number)
except ValueError:
    print("Sayıya çevrilemedi")
""",
                    "Kullanıcı girdisi her zaman beklenen formatta olmayabilir.",
                ),
            ),
            concept(
                "FileNotFoundError yakalayan örnek yaz",
                code_answer(
                    """
from pathlib import Path

try:
    text = Path("notlar.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    text = ""
print(text)
""",
                    "Dosya yoksa programın çökmesini önlemek için hata yakalanabilir.",
                ),
            ),
            concept(
                "TypeError riskini tip kontrolüyle azalt",
                code_answer(
                    """
value = "10"
if isinstance(value, int):
    print(value + 5)
else:
    print("Değer int değil")
""",
                    "Tip kontrolü, uyumsuz tiplerle işlem yapmayı önler.",
                ),
            ),
            concept(
                "finally bloğu kullanılan kısa örnek yaz",
                code_answer(
                    """
try:
    number = int("5")
    print(number)
except ValueError:
    print("Hata oluştu")
finally:
    print("İşlem bitti")
""",
                    "finally bloğu hata olsa da olmasa da çalışır.",
                ),
            ),
            concept(
                "hata mesajını uydurmadan açıklama yaklaşımını anlat",
                text_answer(
                    "Hata türünden emin değilsen kesin konuşmamalısın. Hatanın tam mesajını istemeli, sonra olası nedeni ve güvenli kontrol adımlarını açıklamalısın.",
                ),
            ),
        ],
        "classes": [
            concept(
                "__init__ kullanan basit sınıf yaz",
                code_answer(
                    """
class User:
    def __init__(self, name):
        self.name = name

user = User("Ayşe")
print(user.name)
""",
                    "__init__, nesne oluşturulurken başlangıç değerlerini ayarlar.",
                ),
            ),
            concept(
                "metot içeren küçük sınıf yaz",
                code_answer(
                    """
class Greeter:
    def greet(self, name):
        return f"Merhaba {name}"

greeter = Greeter()
print(greeter.greet("Ali"))
""",
                    "Metot, sınıf içinde tanımlanan fonksiyondur.",
                ),
            ),
            concept(
                "alan hesaplayan Rectangle sınıfı yaz",
                code_answer(
                    """
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

rectangle = Rectangle(4, 5)
print(rectangle.area())
""",
                    "Sınıf, ilişkili veri ve davranışı birlikte tutar.",
                ),
            ),
            concept(
                "__str__ metodu kullanan sınıf yaz",
                code_answer(
                    """
class Note:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

note = Note("Temiz veri")
print(note)
""",
                    "__str__, nesnenin okunabilir metin temsilini döndürür.",
                ),
            ),
            concept(
                "sınıf kullanmanın ne zaman mantıklı olduğunu açıkla",
                text_answer(
                    "Bir verinin birden fazla özelliği ve o veriyle ilişkili davranışları varsa sınıf kullanmak mantıklıdır. Çok küçük işlemler için fonksiyon yeterli olabilir.",
                ),
            ),
        ],
        "algorithms": [
            concept(
                "listedeki en büyük sayıyı bul",
                code_answer(
                    """
numbers = [8, 3, 11, 5]
biggest = max(numbers)
print(biggest)
""",
                    "Basit durumlarda yerleşik max() fonksiyonu en temiz çözümdür.",
                ),
            ),
            concept(
                "kelime frekansı hesapla",
                code_answer(
                    """
from collections import Counter

text = "temiz veri temiz model"
counts = Counter(text.split())
print(counts)
""",
                    "Counter, tekrar eden öğeleri saymak için kullanışlıdır.",
                ),
            ),
            concept(
                "lineer arama örneği yaz",
                code_answer(
                    """
items = ["a", "b", "c"]
target = "b"
found = False
for item in items:
    if item == target:
        found = True
        break
print(found)
""",
                    "Lineer arama, elemanları sırayla kontrol eder.",
                ),
            ),
            concept(
                "listedeki sayıların toplamını döngüyle bul",
                code_answer(
                    """
numbers = [4, 8, 15]
total = 0
for number in numbers:
    total += number
print(total)
""",
                    "Bu örnek algoritmik düşünmenin temelidir: başlangıç değeri, döngü ve güncelleme.",
                ),
            ),
            concept(
                "algoritma açıklarken karmaşıklığı nasıl sade anlatacağını açıkla",
                text_answer(
                    "Önce algoritmanın ne yaptığını, sonra hangi adımları izlediğini söylemelisin. Karmaşıklığı anlatırken küçük veri ve büyük veri farkını basit örnekle açıklamak yeterlidir.",
                ),
            ),
        ],
        "debugging": [
            concept(
                "IndentationError neden olur açıkla",
                text_answer(
                    "IndentationError genellikle if, for, while, def veya class satırından sonra beklenen girinti yazılmadığında oluşur. Aynı bloktaki satırlar aynı girinti seviyesinde olmalıdır.",
                ),
            ),
            concept(
                "NameError hatasını küçük örnekle göster",
                code_answer(
                    """
name = "DarkMind"
print(name)
""",
                    "NameError almamak için değişkeni kullanmadan önce tanımlamalısın.",
                ),
            ),
            concept(
                "print ile basit debug yap",
                code_answer(
                    """
numbers = [1, 2, 3]
total = sum(numbers)
print("total:", total)
""",
                    "Küçük projelerde print ile ara değerleri görmek hızlı bir debug yöntemidir.",
                ),
            ),
            concept(
                "hata mesajında satır numarasını nasıl kullanacağını anlat",
                text_answer(
                    "Traceback içindeki dosya adı ve satır numarası, hatanın nerede oluştuğunu gösterir. Önce o satırı, sonra o satırda kullanılan değişkenleri kontrol etmelisin.",
                ),
            ),
            concept(
                "yanlış cevaptan emin değilsen nasıl düzeltme isteyeceğini açıkla",
                text_answer(
                    "Emin değilsem kesin çözüm gibi konuşmam. Hata mesajını, ilgili kod parçasını ve beklenen davranışı istemek daha doğru olur.",
                ),
            ),
        ],
        "project_workflow": [
            concept(
                "DarkMind veri pipeline adımlarını kısaca açıkla",
                text_answer(
                    "Önce raw veri üretilir veya toplanır. Sonra clean_text ile temizlenir, corpus build edilir, kalite kontrol yapılır, tokenizer eğitilir ve model eğitimi ayrı komutla başlatılır.",
                ),
            ),
            concept(
                "model eğitmeden önce neden dataset kalite kontrolü yapılır açıkla",
                text_answer(
                    "Kalitesiz veya tekrar eden veri küçük modeli ezbere yöneltebilir. Eğitimden önce karakter sayısı, tekrar eden satırlar ve dosya dağılımı kontrol edilmelidir.",
                ),
            ),
            concept(
                "eval sonucuna göre nasıl karar verileceğini anlat",
                text_answer(
                    "Yeni checkpoint ancak eval pass rate ve kategori sonuçları eski koşudan daha iyiyse güçlü aday sayılmalıdır. Tek bir iyi cevap genel kalite kanıtı değildir.",
                ),
            ),
            concept(
                "checkpoint dosyalarına neden dikkatli davranılması gerektiğini açıkla",
                text_answer(
                    "Checkpointler büyük dosyalardır ve deney kayıtları için önemlidir. Otomatik scriptler checkpoint silmemeli veya üzerine yazma riskini açıkça kontrol etmelidir.",
                ),
            ),
            concept(
                "bilmediğin proje komutunda nasıl davranacağını açıkla",
                text_answer(
                    "Komutu uydurmak yerine README veya scripts klasöründeki mevcut araçlara bakmak gerekir. Emin değilsem güvenli varsayımı belirtip doğrulama adımı önermeliyim.",
                ),
            ),
        ],
        "pytorch_basics": [
            concept(
                "torch tensor oluştur",
                code_answer(
                    """
import torch

values = torch.tensor([1, 2, 3])
print(values)
""",
                    "Tensor, PyTorch'un temel veri yapısıdır.",
                ),
            ),
            concept(
                "CUDA kullanılabilir mi kontrol et",
                code_answer(
                    """
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
""",
                    "Bu kontrol, modelin GPU üzerinde çalışıp çalışamayacağını anlamak için kullanılır.",
                ),
            ),
            concept(
                "tensor boyutunu yazdır",
                code_answer(
                    """
import torch

matrix = torch.zeros((2, 3))
print(matrix.shape)
""",
                    "shape, tensor boyutlarını gösterir.",
                ),
            ),
            concept(
                "basit tensor toplama örneği yaz",
                code_answer(
                    """
import torch

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
print(a + b)
""",
                    "Aynı şekle sahip tensorler eleman bazında toplanabilir.",
                ),
            ),
            concept(
                "PyTorch bilmediğin konuda nasıl cevap vermelisin açıkla",
                text_answer(
                    "PyTorch ayrıntısından emin değilsem kesin API iddiası vermemeliyim. Basit kavramı açıklayıp resmi dokümana veya küçük test koduna yönlendirmek daha güvenlidir.",
                ),
            ),
        ],
        "tokenizer_basics": [
            concept(
                "tokenizer kavramını Türkçe açıkla",
                text_answer(
                    "Tokenizer, metni modelin anlayacağı sayısal parçalara dönüştüren bileşendir. Küçük dil modellerinde tokenizer ve checkpoint uyumu çok önemlidir.",
                ),
            ),
            concept(
                "ByteLevelBPETokenizer yükleme örneği yaz",
                code_answer(
                    """
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/darkmind-tokenizer/vocab.json",
    "tokenizer/darkmind-tokenizer/merges.txt",
)
""",
                    "Tokenizer dosyaları model eğitimi ve üretim sırasında aynı olmalıdır.",
                ),
            ),
            concept(
                "metni id listesine çeviren örnek yaz",
                code_answer(
                    """
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
encoded = tokenizer.encode("Merhaba DarkMind")
ids = encoded.ids
print(ids)
""",
                    "encode() metni sayısal id listesine dönüştürür.",
                ),
            ),
            concept(
                "id listesini metne çeviren örnek yaz",
                code_answer(
                    """
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
text = tokenizer.decode([1, 2, 3])
print(text)
""",
                    "decode() sayısal id listesinden metin üretir. Gerçek çıktı vocab içeriğine bağlıdır.",
                ),
            ),
            concept(
                "tokenizer mismatch hatasını açıkla",
                text_answer(
                    "Model farklı tokenizer ile eğitildiyse vocab boyutu veya id anlamları uyuşmayabilir. Bu durumda doğru tokenizer klasörü kullanılmalı veya model yeniden eğitilmelidir.",
                ),
            ),
        ],
        "cli_tools": [
            concept(
                "argparse ile basit CLI argümanı oku",
                code_answer(
                    """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="DarkMind")
args = parser.parse_args()
print(args.name)
""",
                    "argparse, komut satırı argümanlarını düzenli okumayı sağlar.",
                ),
            ),
            concept(
                "pathlib kullanan CLI dosya yolu örneği yaz",
                code_answer(
                    """
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="data/example.txt")
args = parser.parse_args()
path = Path(args.path)
print(path)
""",
                    "Path kullanmak Windows ve diğer sistemlerde yol yönetimini kolaylaştırır.",
                ),
            ),
            concept(
                "PowerShell'de Python script çalıştırmayı açıkla",
                text_answer(
                    "PowerShell'de proje kök dizinindeyken python scripts/script_adi.py şeklinde çalıştırabilirsin. Sanal ortam kullanıyorsan önce .\\.venv\\Scripts\\Activate.ps1 komutu ile aktif edebilirsin.",
                ),
            ),
            concept(
                "CLI aracında --help neden önemlidir açıkla",
                text_answer(
                    "--help çıktısı aracın hangi argümanları aldığını gösterir. Bu, yanlış komut çalıştırma riskini azaltır.",
                ),
            ),
            concept(
                "CLI scriptinde güvenli varsayılan değer kullanmayı göster",
                code_answer(
                    """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=10)
args = parser.parse_args()
print(args.limit)
""",
                    "Güvenli varsayılan değerler aracı daha kolay ve öngörülebilir yapar.",
                ),
            ),
        ],
        "git_basics": [
            concept(
                "git status komutunun ne yaptığını açıkla",
                text_answer(
                    "git status çalışma ağacındaki değiştirilmiş, staged ve untracked dosyaları gösterir. Commit atmadan önce ilk kontrol komutudur.",
                ),
            ),
            concept(
                "git add ve git commit farkını açıkla",
                text_answer(
                    "git add dosyaları staging alanına alır. git commit ise staged değişiklikleri kalıcı bir kayıt haline getirir.",
                ),
            ),
            concept(
                "git push ne zaman kullanılır açıkla",
                text_answer(
                    "git push yerel commitleri uzak depoya göndermek için kullanılır. Önce doğru branchte olduğunu ve commit içeriğini kontrol etmek gerekir.",
                ),
            ),
            concept(
                "commit mesajı yazarken nelere dikkat edilir anlat",
                text_answer(
                    "Commit mesajı kısa, açık ve yapılan değişikliği anlatan bir cümle olmalıdır. Örneğin: add Turkish code data factory.",
                ),
            ),
            concept(
                "git komutundan emin değilsen nasıl davranacağını açıkla",
                text_answer(
                    "Emin değilsen önce git status ile durumu görmeli ve yıkıcı komutlardan kaçınmalısın. Değişiklikleri silen komutlar açık onay olmadan çalıştırılmamalıdır.",
                ),
            ),
        ],
    }


def render_example(bucket: str, difficulty: str, prompt: str, answer: str) -> str:
    return (
        f"# bucket: {bucket}\n"
        f"# difficulty: {difficulty}\n\n"
        f"Kullanıcı: {prompt.strip()}\n"
        f"Asistan:\n\n"
        f"{answer.strip()}"
    )


def build_examples() -> list[dict]:
    examples = []
    bucket_concepts = build_bucket_concepts()

    for bucket, concepts in bucket_concepts.items():
        difficulty = DIFFICULTY_BY_BUCKET[bucket]

        for concept_item in concepts:
            for template in PROMPT_VARIANTS:
                prompt = template.format(task=concept_item["task"])
                examples.append({
                    "bucket": bucket,
                    "difficulty": difficulty,
                    "prompt": prompt,
                    "answer": concept_item["answer"],
                })

    return examples


def main() -> None:
    examples = build_examples()

    if len(examples) < 800:
        raise ValueError(f"Expected at least 800 examples, got {len(examples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n\n".join(
        render_example(
            example["bucket"],
            example["difficulty"],
            example["prompt"],
            example["answer"],
        )
        for example in examples
    )
    OUTPUT_PATH.write_text(output_text + "\n", encoding="utf-8")

    counts = Counter(example["bucket"] for example in examples)

    print("=" * 70)
    print(f"Generated examples: {len(examples):,}")
    print(f"Output path: {OUTPUT_PATH}")
    print("Examples by bucket:")

    for bucket, count in sorted(counts.items()):
        print(f"- {bucket}: {count:,}")

    print("=" * 70)


if __name__ == "__main__":
    main()
