from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "python_examples"
    / "python_examples_v01.txt"
)


def add_example(examples: list[tuple[str, str]], prompt: str, code: str) -> None:
    examples.append((prompt.strip(), code.strip()))


def build_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []

    variable_examples = [
        (
            "Python'da bir metin değişkeni tanımla.",
            """
name = "DarkMind"
print(name)
""",
        ),
        (
            "Python'da bir tam sayı değişkeni tanımla.",
            """
age = 3
print(age)
""",
        ),
        (
            "Python'da ondalıklı sayı değişkeni tanımla.",
            """
temperature = 0.8
print(temperature)
""",
        ),
        (
            "Python'da boolean değişkeni tanımla.",
            """
is_active = True
print(is_active)
""",
        ),
        (
            "Python'da iki değişkenin değerini ekrana yazdır.",
            """
city = "İstanbul"
score = 42
print(city, score)
""",
        ),
        (
            "Python'da değişkenleri kullanarak kısa bir cümle oluştur.",
            """
model_name = "DarkMind"
version = "v01"
message = f"{model_name} veri seti {version} ile büyüyor."
print(message)
""",
        ),
        (
            "Python'da iki sayıyı topla ve sonucu değişkende sakla.",
            """
a = 12
b = 8
total = a + b
print(total)
""",
        ),
        (
            "Python'da kullanıcı adını bir değişkende sakla.",
            """
username = "tarik"
print(f"Merhaba {username}")
""",
        ),
    ]

    function_examples = [
        (
            "Python'da iki sayıyı toplayan bir fonksiyon yaz.",
            """
def add_numbers(a, b):
    return a + b
""",
        ),
        (
            "Python'da iki sayıyı çarpan bir fonksiyon yaz.",
            """
def multiply_numbers(a, b):
    return a * b
""",
        ),
        (
            "Python'da bir sayının karesini döndüren fonksiyon yaz.",
            """
def square(number):
    return number * number
""",
        ),
        (
            "Python'da metni büyük harfe çeviren fonksiyon yaz.",
            """
def make_upper(text):
    return text.upper()
""",
        ),
        (
            "Python'da listedeki sayıların toplamını döndüren fonksiyon yaz.",
            """
def sum_list(numbers):
    return sum(numbers)
""",
        ),
        (
            "Python'da bir isme selam veren fonksiyon yaz.",
            """
def greet(name):
    return f"Merhaba {name}"
""",
        ),
        (
            "Python'da ortalama hesaplayan bir fonksiyon yaz.",
            """
def average(numbers):
    return sum(numbers) / len(numbers)
""",
        ),
        (
            "Python'da boş liste kontrolü yapan fonksiyon yaz.",
            """
def is_empty(items):
    return len(items) == 0
""",
        ),
    ]

    if_else_examples = [
        (
            "Python'da bir sayının pozitif olup olmadığını kontrol et.",
            """
number = 7
if number > 0:
    print("Pozitif")
else:
    print("Pozitif değil")
""",
        ),
        (
            "Python'da bir sayının çift mi tek mi olduğunu yazdır.",
            """
number = 10
if number % 2 == 0:
    print("Çift")
else:
    print("Tek")
""",
        ),
        (
            "Python'da yaşa göre yetişkin kontrolü yap.",
            """
age = 20
if age >= 18:
    print("Yetişkin")
else:
    print("Çocuk")
""",
        ),
        (
            "Python'da not değerine göre geçti veya kaldı yazdır.",
            """
grade = 72
if grade >= 50:
    print("Geçti")
else:
    print("Kaldı")
""",
        ),
        (
            "Python'da boş metin kontrolü yap.",
            """
text = "Merhaba"
if text:
    print("Metin dolu")
else:
    print("Metin boş")
""",
        ),
        (
            "Python'da listenin boş olup olmadığını kontrol et.",
            """
items = [1, 2, 3]
if items:
    print("Liste dolu")
else:
    print("Liste boş")
""",
        ),
        (
            "Python'da sıcaklığa göre kısa mesaj yazdır.",
            """
temperature = 28
if temperature >= 25:
    print("Hava sıcak")
else:
    print("Hava serin")
""",
        ),
        (
            "Python'da şifre uzunluğu kontrolü yap.",
            """
password = "gizli123"
if len(password) >= 8:
    print("Şifre uzunluğu yeterli")
else:
    print("Şifre çok kısa")
""",
        ),
    ]

    for_loop_examples = [
        (
            "Python'da listedeki sayıları for döngüsüyle yazdır.",
            """
numbers = [1, 2, 3, 4]
for number in numbers:
    print(number)
""",
        ),
        (
            "Python'da 1'den 5'e kadar sayıları yazdır.",
            """
for number in range(1, 6):
    print(number)
""",
        ),
        (
            "Python'da listedeki isimlere selam ver.",
            """
names = ["Ayşe", "Mehmet", "Zeynep"]
for name in names:
    print(f"Merhaba {name}")
""",
        ),
        (
            "Python'da listedeki sayıların toplamını for döngüsüyle bul.",
            """
numbers = [4, 8, 15]
total = 0
for number in numbers:
    total += number
print(total)
""",
        ),
        (
            "Python'da listedeki çift sayıları yazdır.",
            """
numbers = [1, 2, 3, 4, 5, 6]
for number in numbers:
    if number % 2 == 0:
        print(number)
""",
        ),
        (
            "Python'da metindeki karakterleri tek tek yazdır.",
            """
text = "kod"
for character in text:
    print(character)
""",
        ),
        (
            "Python'da enumerate ile liste elemanlarını numaralandır.",
            """
tasks = ["oku", "yaz", "test et"]
for index, task in enumerate(tasks, start=1):
    print(index, task)
""",
        ),
        (
            "Python'da sözlük anahtar ve değerlerini döngüyle yazdır.",
            """
scores = {"Ali": 80, "Ayşe": 95}
for name, score in scores.items():
    print(name, score)
""",
        ),
    ]

    while_loop_examples = [
        (
            "Python'da while ile 1'den 5'e kadar say.",
            """
counter = 1
while counter <= 5:
    print(counter)
    counter += 1
""",
        ),
        (
            "Python'da while ile geri sayım yap.",
            """
counter = 5
while counter > 0:
    print(counter)
    counter -= 1
""",
        ),
        (
            "Python'da while ile toplam hesapla.",
            """
number = 1
total = 0
while number <= 5:
    total += number
    number += 1
print(total)
""",
        ),
        (
            "Python'da while döngüsünden break ile çık.",
            """
number = 0
while True:
    number += 1
    if number == 3:
        break
print(number)
""",
        ),
        (
            "Python'da while ile liste elemanlarını sırayla yazdır.",
            """
items = ["a", "b", "c"]
index = 0
while index < len(items):
    print(items[index])
    index += 1
""",
        ),
        (
            "Python'da while ile koşul sağlanana kadar artırma yap.",
            """
value = 2
while value < 20:
    value *= 2
print(value)
""",
        ),
    ]

    list_examples = [
        (
            "Python'da bir liste oluştur ve ilk elemanı yazdır.",
            """
fruits = ["elma", "armut", "muz"]
print(fruits[0])
""",
        ),
        (
            "Python'da listeye yeni eleman ekle.",
            """
fruits = ["elma", "armut"]
fruits.append("muz")
print(fruits)
""",
        ),
        (
            "Python'da listeden eleman çıkar.",
            """
fruits = ["elma", "armut", "muz"]
fruits.remove("armut")
print(fruits)
""",
        ),
        (
            "Python'da listedeki eleman sayısını bul.",
            """
numbers = [3, 5, 8, 13]
print(len(numbers))
""",
        ),
        (
            "Python'da listenin son elemanını yazdır.",
            """
items = ["ilk", "orta", "son"]
print(items[-1])
""",
        ),
        (
            "Python'da listeyi ters sırada yazdır.",
            """
numbers = [1, 2, 3, 4]
print(list(reversed(numbers)))
""",
        ),
        (
            "Python'da iki listeyi birleştir.",
            """
first = [1, 2]
second = [3, 4]
combined = first + second
print(combined)
""",
        ),
        (
            "Python'da listenin belirli bir parçasını al.",
            """
numbers = [10, 20, 30, 40, 50]
print(numbers[1:4])
""",
        ),
    ]

    dictionary_examples = [
        (
            "Python'da basit bir sözlük oluştur.",
            """
person = {"name": "Ayşe", "age": 25}
print(person)
""",
        ),
        (
            "Python'da sözlükten değer oku.",
            """
person = {"name": "Ayşe", "age": 25}
print(person["name"])
""",
        ),
        (
            "Python'da sözlüğe yeni anahtar ekle.",
            """
person = {"name": "Ayşe"}
person["city"] = "İstanbul"
print(person)
""",
        ),
        (
            "Python'da get ile güvenli sözlük okuma yap.",
            """
settings = {"theme": "dark"}
language = settings.get("language", "tr")
print(language)
""",
        ),
        (
            "Python'da sözlük anahtarlarını yazdır.",
            """
scores = {"Ali": 80, "Ayşe": 95}
for name in scores.keys():
    print(name)
""",
        ),
        (
            "Python'da sözlük değerlerini yazdır.",
            """
scores = {"Ali": 80, "Ayşe": 95}
for score in scores.values():
    print(score)
""",
        ),
        (
            "Python'da sözlükte anahtar var mı kontrol et.",
            """
person = {"name": "Ayşe"}
if "name" in person:
    print("name anahtarı var")
""",
        ),
    ]

    set_examples = [
        (
            "Python'da tekrarları kaldırmak için set kullan.",
            """
numbers = [1, 2, 2, 3, 3, 3]
unique_numbers = set(numbers)
print(unique_numbers)
""",
        ),
        (
            "Python'da iki kümenin kesişimini bul.",
            """
first = {"python", "ai", "data"}
second = {"python", "web"}
common = first & second
print(common)
""",
        ),
        (
            "Python'da iki kümenin birleşimini bul.",
            """
first = {"a", "b"}
second = {"b", "c"}
print(first | second)
""",
        ),
        (
            "Python'da sete eleman ekle.",
            """
tags = {"python", "kod"}
tags.add("veri")
print(tags)
""",
        ),
        (
            "Python'da sette eleman var mı kontrol et.",
            """
tags = {"python", "kod"}
if "python" in tags:
    print("Bulundu")
""",
        ),
    ]

    tuple_examples = [
        (
            "Python'da tuple oluştur ve elemanlarını yazdır.",
            """
point = (10, 20)
print(point[0], point[1])
""",
        ),
        (
            "Python'da tuple unpacking örneği yaz.",
            """
name, score = ("Ayşe", 95)
print(name, score)
""",
        ),
        (
            "Python'da fonksiyondan iki değer döndür.",
            """
def min_max(numbers):
    return min(numbers), max(numbers)

smallest, biggest = min_max([3, 9, 1])
print(smallest, biggest)
""",
        ),
        (
            "Python'da tuple listesini sırala.",
            """
students = [("Ali", 80), ("Ayşe", 95), ("Can", 70)]
students.sort(key=lambda item: item[1])
print(students)
""",
        ),
        (
            "Python'da değişmeyen koordinat bilgisi için tuple kullan.",
            """
location = (41.01, 28.97)
latitude, longitude = location
print(latitude, longitude)
""",
        ),
    ]

    string_examples = [
        (
            "Python'da metnin uzunluğunu bul.",
            """
text = "Merhaba"
print(len(text))
""",
        ),
        (
            "Python'da metni küçük harfe çevir.",
            """
text = "DARKMIND"
print(text.lower())
""",
        ),
        (
            "Python'da metni büyük harfe çevir.",
            """
text = "darkmind"
print(text.upper())
""",
        ),
        (
            "Python'da metindeki boşlukları temizle.",
            """
text = "  merhaba  "
print(text.strip())
""",
        ),
        (
            "Python'da metin içinde kelime ara.",
            """
text = "Python öğreniyorum"
if "Python" in text:
    print("Kelime bulundu")
""",
        ),
        (
            "Python'da metni kelimelere ayır.",
            """
text = "temiz veri önemlidir"
words = text.split()
print(words)
""",
        ),
        (
            "Python'da kelimeleri tek metinde birleştir.",
            """
words = ["temiz", "veri", "önemlidir"]
sentence = " ".join(words)
print(sentence)
""",
        ),
        (
            "Python'da f-string ile değerleri metne yerleştir.",
            """
name = "DarkMind"
step = 1000
print(f"{name} {step} adım eğitildi.")
""",
        ),
    ]

    comprehension_examples = [
        (
            "Python'da list comprehension ile sayıların karesini al.",
            """
numbers = [1, 2, 3, 4]
squares = [number * number for number in numbers]
print(squares)
""",
        ),
        (
            "Python'da list comprehension ile çift sayıları seç.",
            """
numbers = [1, 2, 3, 4, 5, 6]
evens = [number for number in numbers if number % 2 == 0]
print(evens)
""",
        ),
        (
            "Python'da metin listesini küçük harfe çevir.",
            """
words = ["PYTHON", "VERİ", "MODEL"]
lower_words = [word.lower() for word in words]
print(lower_words)
""",
        ),
        (
            "Python'da list comprehension ile uzun kelimeleri filtrele.",
            """
words = ["ai", "python", "veri", "transformer"]
long_words = [word for word in words if len(word) > 4]
print(long_words)
""",
        ),
        (
            "Python'da range ile liste comprehension kullan.",
            """
numbers = [number for number in range(1, 6)]
print(numbers)
""",
        ),
        (
            "Python'da listedeki boş metinleri çıkar.",
            """
texts = ["merhaba", "", "veri", " "]
clean_texts = [text.strip() for text in texts if text.strip()]
print(clean_texts)
""",
        ),
        (
            "Python'da sayı listesinden pozitifleri seç.",
            """
numbers = [-2, 0, 4, 7]
positive_numbers = [number for number in numbers if number > 0]
print(positive_numbers)
""",
        ),
        (
            "Python'da kelime uzunluklarını liste olarak üret.",
            """
words = ["python", "kod", "veri"]
lengths = [len(word) for word in words]
print(lengths)
""",
        ),
    ]

    file_examples = [
        (
            "Python'da pathlib ile bir metin dosyası oku.",
            """
from pathlib import Path

path = Path("notlar.txt")
content = path.read_text(encoding="utf-8")
print(content)
""",
        ),
        (
            "Python'da pathlib ile bir metin dosyası yaz.",
            """
from pathlib import Path

path = Path("notlar.txt")
path.write_text("Merhaba Python", encoding="utf-8")
""",
        ),
        (
            "Python'da dosya varsa içeriğini oku.",
            """
from pathlib import Path

path = Path("notlar.txt")
if path.exists():
    print(path.read_text(encoding="utf-8"))
""",
        ),
        (
            "Python'da satırları liste olarak oku.",
            """
from pathlib import Path

path = Path("notlar.txt")
lines = path.read_text(encoding="utf-8").splitlines()
print(lines)
""",
        ),
        (
            "Python'da listeyi satır satır dosyaya yaz.",
            """
from pathlib import Path

lines = ["birinci satır", "ikinci satır"]
Path("rapor.txt").write_text("\\n".join(lines), encoding="utf-8")
""",
        ),
        (
            "Python'da dosya yolunu Path ile oluştur.",
            """
from pathlib import Path

folder = Path("data")
path = folder / "ornek.txt"
print(path)
""",
        ),
        (
            "Python'da dosyanın karakter sayısını bul.",
            """
from pathlib import Path

path = Path("notlar.txt")
text = path.read_text(encoding="utf-8")
print(len(text))
""",
        ),
        (
            "Python'da dosya içeriğini güvenli şekilde küçük harfe çevir.",
            """
from pathlib import Path

path = Path("notlar.txt")
if path.exists():
    text = path.read_text(encoding="utf-8")
    print(text.lower())
""",
        ),
    ]

    try_examples = [
        (
            "Python'da ValueError için try except örneği yaz.",
            """
value = "42"
try:
    number = int(value)
    print(number)
except ValueError:
    print("Sayıya çevrilemedi")
""",
        ),
        (
            "Python'da dosya okurken FileNotFoundError yakala.",
            """
from pathlib import Path

try:
    text = Path("notlar.txt").read_text(encoding="utf-8")
    print(text)
except FileNotFoundError:
    print("Dosya bulunamadı")
""",
        ),
        (
            "Python'da sıfıra bölme hatasını yakala.",
            """
try:
    result = 10 / 0
    print(result)
except ZeroDivisionError:
    print("Sıfıra bölme yapılamaz")
""",
        ),
        (
            "Python'da sözlük anahtarı yoksa KeyError yakala.",
            """
person = {"name": "Ayşe"}
try:
    print(person["age"])
except KeyError:
    print("age anahtarı yok")
""",
        ),
        (
            "Python'da liste index hatasını yakala.",
            """
items = ["a", "b"]
try:
    print(items[5])
except IndexError:
    print("Geçersiz index")
""",
        ),
        (
            "Python'da try except else örneği yaz.",
            """
text = "12"
try:
    number = int(text)
except ValueError:
    print("Hata oluştu")
else:
    print(number)
""",
        ),
        (
            "Python'da finally bloğu kullanılan örnek yaz.",
            """
try:
    number = int("5")
    print(number)
except ValueError:
    print("Hata")
finally:
    print("İşlem bitti")
""",
        ),
        (
            "Python'da tip kontrolüyle TypeError riskini azalt.",
            """
value = "10"
if isinstance(value, int):
    print(value + 5)
else:
    print("Değer int değil")
""",
        ),
    ]

    class_examples = [
        (
            "Python'da basit bir sınıf tanımla.",
            """
class User:
    def __init__(self, name):
        self.name = name

user = User("Ayşe")
print(user.name)
""",
        ),
        (
            "Python'da metot içeren basit sınıf yaz.",
            """
class Greeter:
    def greet(self, name):
        return f"Merhaba {name}"

greeter = Greeter()
print(greeter.greet("Ali"))
""",
        ),
        (
            "Python'da sayaç sınıfı yaz.",
            """
class Counter:
    def __init__(self):
        self.value = 0

    def increase(self):
        self.value += 1

counter = Counter()
counter.increase()
print(counter.value)
""",
        ),
        (
            "Python'da ürün fiyatı tutan sınıf yaz.",
            """
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

product = Product("Kitap", 120)
print(product.price)
""",
        ),
        (
            "Python'da dikdörtgen alanı hesaplayan sınıf yaz.",
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
        ),
        (
            "Python'da __str__ metodu kullanan sınıf yaz.",
            """
class Note:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

note = Note("Temiz veri önemlidir")
print(note)
""",
        ),
        (
            "Python'da basit model bilgisi tutan sınıf yaz.",
            """
class ModelInfo:
    def __init__(self, name, parameter_count):
        self.name = name
        self.parameter_count = parameter_count

info = ModelInfo("DarkMind", 30000000)
print(info.name)
""",
        ),
    ]

    math_examples = [
        (
            "Python'da mutlak değer hesapla.",
            """
number = -12
print(abs(number))
""",
        ),
        (
            "Python'da yuvarlama yap.",
            """
value = 3.14159
print(round(value, 2))
""",
        ),
        (
            "Python'da listedeki en küçük sayıyı bul.",
            """
numbers = [8, 3, 11]
print(min(numbers))
""",
        ),
        (
            "Python'da listedeki en büyük sayıyı bul.",
            """
numbers = [8, 3, 11]
print(max(numbers))
""",
        ),
        (
            "Python'da toplam ve ortalama hesapla.",
            """
numbers = [10, 20, 30]
total = sum(numbers)
average = total / len(numbers)
print(average)
""",
        ),
        (
            "Python'da üs alma işlemi yap.",
            """
base = 2
power = 5
print(base ** power)
""",
        ),
        (
            "Python'da bölme sonucunu iki ondalığa yuvarla.",
            """
result = 10 / 3
print(round(result, 2))
""",
        ),
        (
            "Python'da kalan bulma işlemi yap.",
            """
number = 17
remainder = number % 5
print(remainder)
""",
        ),
    ]

    sorting_examples = [
        (
            "Python'da sayı listesini küçükten büyüğe sırala.",
            """
numbers = [5, 1, 9, 3]
sorted_numbers = sorted(numbers)
print(sorted_numbers)
""",
        ),
        (
            "Python'da sayı listesini büyükten küçüğe sırala.",
            """
numbers = [5, 1, 9, 3]
sorted_numbers = sorted(numbers, reverse=True)
print(sorted_numbers)
""",
        ),
        (
            "Python'da kelimeleri alfabetik sırala.",
            """
words = ["veri", "model", "python"]
print(sorted(words))
""",
        ),
        (
            "Python'da kelimeleri uzunluğa göre sırala.",
            """
words = ["ai", "python", "transformer"]
print(sorted(words, key=len))
""",
        ),
        (
            "Python'da sözlük listesini puana göre sırala.",
            """
students = [
    {"name": "Ali", "score": 80},
    {"name": "Ayşe", "score": 95},
]
students = sorted(students, key=lambda item: item["score"])
print(students)
""",
        ),
        (
            "Python'da tuple listesini ikinci elemana göre sırala.",
            """
pairs = [("a", 3), ("b", 1), ("c", 2)]
print(sorted(pairs, key=lambda item: item[1]))
""",
        ),
        (
            "Python'da listeyi yerinde sırala.",
            """
numbers = [4, 2, 7]
numbers.sort()
print(numbers)
""",
        ),
        (
            "Python'da büyük küçük harf duyarsız sıralama yap.",
            """
words = ["Python", "ai", "Data"]
print(sorted(words, key=str.lower))
""",
        ),
    ]

    filtering_examples = [
        (
            "Python'da listedeki çift sayıları filtrele.",
            """
numbers = [1, 2, 3, 4, 5, 6]
evens = [number for number in numbers if number % 2 == 0]
print(evens)
""",
        ),
        (
            "Python'da kısa kelimeleri filtrele.",
            """
words = ["ai", "python", "kod", "transformer"]
long_words = [word for word in words if len(word) >= 5]
print(long_words)
""",
        ),
        (
            "Python'da boş metinleri listeden çıkar.",
            """
texts = ["merhaba", "", " ", "veri"]
cleaned = [text for text in texts if text.strip()]
print(cleaned)
""",
        ),
        (
            "Python'da belirli puanın üstündeki öğrencileri seç.",
            """
students = {"Ali": 80, "Ayşe": 95, "Can": 60}
passed = {name: score for name, score in students.items() if score >= 70}
print(passed)
""",
        ),
        (
            "Python'da pozitif sayıları filtrele.",
            """
numbers = [-3, 0, 2, 5]
positive = [number for number in numbers if number > 0]
print(positive)
""",
        ),
        (
            "Python'da belirli harfle başlayan kelimeleri seç.",
            """
words = ["model", "metin", "veri", "makine"]
selected = [word for word in words if word.startswith("m")]
print(selected)
""",
        ),
        (
            "Python'da tek sayıları filter fonksiyonuyla seç.",
            """
numbers = [1, 2, 3, 4, 5]
odds = list(filter(lambda number: number % 2 == 1, numbers))
print(odds)
""",
        ),
        (
            "Python'da None değerlerini listeden çıkar.",
            """
items = [1, None, 2, None, 3]
clean_items = [item for item in items if item is not None]
print(clean_items)
""",
        ),
    ]

    counting_examples = [
        (
            "Python'da listedeki elemanları Counter ile say.",
            """
from collections import Counter

words = ["veri", "model", "veri"]
counts = Counter(words)
print(counts)
""",
        ),
        (
            "Python'da metindeki harf sayısını hesapla.",
            """
text = "banana"
count = text.count("a")
print(count)
""",
        ),
        (
            "Python'da listedeki çift sayı adedini bul.",
            """
numbers = [1, 2, 3, 4, 6]
count = sum(1 for number in numbers if number % 2 == 0)
print(count)
""",
        ),
        (
            "Python'da kelime frekansı hesapla.",
            """
from collections import Counter

text = "temiz veri temiz model"
counts = Counter(text.split())
print(counts)
""",
        ),
        (
            "Python'da sözlükle basit sayaç tut.",
            """
words = ["a", "b", "a"]
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
print(counts)
""",
        ),
        (
            "Python'da listedeki uzun kelime sayısını bul.",
            """
words = ["ai", "python", "transformer"]
count = sum(1 for word in words if len(word) > 4)
print(count)
""",
        ),
        (
            "Python'da toplam satır sayısını hesapla.",
            """
text = "birinci\\nikinci\\nüçüncü"
line_count = len(text.splitlines())
print(line_count)
""",
        ),
    ]

    cli_examples = [
        (
            "Python'da kullanıcıdan isim alıp selam ver.",
            """
name = input("Adın nedir? ")
print(f"Merhaba {name}")
""",
        ),
        (
            "Python'da kullanıcıdan iki sayı alıp topla.",
            """
first = int(input("İlk sayı: "))
second = int(input("İkinci sayı: "))
print(first + second)
""",
        ),
        (
            "Python'da kullanıcıdan yaş alıp yetişkin kontrolü yap.",
            """
age = int(input("Yaş: "))
if age >= 18:
    print("Yetişkin")
else:
    print("Çocuk")
""",
        ),
        (
            "Python'da kullanıcıdan metin alıp uzunluğunu yazdır.",
            """
text = input("Metin: ")
print(len(text))
""",
        ),
        (
            "Python'da kullanıcıdan sayı alıp karesini yazdır.",
            """
number = int(input("Sayı: "))
print(number * number)
""",
        ),
        (
            "Python'da kullanıcıdan virgüllü liste alıp parçalara ayır.",
            """
raw_text = input("Kelimeleri virgülle yaz: ")
items = [item.strip() for item in raw_text.split(",") if item.strip()]
print(items)
""",
        ),
    ]

    all_groups = [
        variable_examples,
        function_examples,
        if_else_examples,
        for_loop_examples,
        while_loop_examples,
        list_examples,
        dictionary_examples,
        set_examples,
        tuple_examples,
        string_examples,
        comprehension_examples,
        file_examples,
        try_examples,
        class_examples,
        math_examples,
        sorting_examples,
        filtering_examples,
        counting_examples,
        cli_examples,
    ]

    for group in all_groups:
        for prompt, code in group:
            add_example(examples, prompt, code)

    return examples


def render_example(prompt: str, code: str) -> str:
    return f"Kullanıcı: {prompt}\nAsistan:\n\n```python\n{code}\n```"


def main() -> None:
    examples = build_examples()

    if len(examples) < 120:
        raise ValueError(f"Expected at least 120 examples, got {len(examples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n\n".join(
        render_example(prompt, code)
        for prompt, code in examples
    )
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print(f"Generated examples: {len(examples)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
