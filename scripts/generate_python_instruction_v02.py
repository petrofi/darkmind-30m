from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "python_examples"
    / "python_instruction_v02.txt"
)


PROMPT_VARIANTS = [
    "{task}",
    "{task} Kısa bir örnek göster.",
    "{task} Basit bir Python örneği yaz.",
    "{task} Yeni başlayan biri için açıkla.",
    "{task} Kodla göster.",
    "{task} Küçük ve çalışır bir örnek ver.",
    "{task} Türkçe açıklama ve kodla anlat.",
    "{task} Sadece kısa bir çözüm yaz.",
]


EXAMPLE_SPECS = [
    {
        "task": "Python'da değişken nasıl tanımlanır?",
        "answer": "Python'da değişken tanımlamak için bir isme değer atanır.",
        "code": """
name = "DarkMind"
age = 1
print(name, age)
""",
    },
    {
        "task": "Python'da string işlemleri nasıl yapılır?",
        "answer": "String üzerinde lower, upper, strip ve split gibi metotlar kullanılabilir.",
        "code": """
text = "  Merhaba Python  "
cleaned = text.strip().lower()
print(cleaned)
""",
    },
    {
        "task": "Python'da sayılarla temel matematik işlemleri nasıl yapılır?",
        "answer": "Sayılar üzerinde toplama, çıkarma, çarpma ve bölme işlemleri yapılabilir.",
        "code": """
a = 10
b = 3
print(a + b)
print(a - b)
print(a * b)
print(a / b)
""",
    },
    {
        "task": "Python'da fonksiyon nasıl yazılır?",
        "answer": "Fonksiyonlar def anahtar kelimesiyle tanımlanır.",
        "code": """
def greet(name):
    return f"Merhaba {name}"

print(greet("Ayşe"))
""",
    },
    {
        "task": "Python'da return değeri olan fonksiyon nasıl yazılır?",
        "answer": "return, fonksiyonun hesapladığı değeri dışarı döndürür.",
        "code": """
def add_numbers(a, b):
    return a + b

result = add_numbers(4, 6)
print(result)
""",
    },
    {
        "task": "Python'da if else nasıl kullanılır?",
        "answer": "if/else, koşula göre farklı kod bloklarının çalışmasını sağlar.",
        "code": """
score = 72
if score >= 50:
    print("Geçti")
else:
    print("Kaldı")
""",
    },
    {
        "task": "Python'da for döngüsü nasıl kullanılır?",
        "answer": "for döngüsü, bir koleksiyonun elemanlarını sırayla gezmek için kullanılır.",
        "code": """
names = ["Ali", "Ayşe", "Can"]
for name in names:
    print(name)
""",
    },
    {
        "task": "Python'da while döngüsü nasıl kullanılır?",
        "answer": "while döngüsü, koşul doğru olduğu sürece çalışır.",
        "code": """
counter = 1
while counter <= 5:
    print(counter)
    counter += 1
""",
    },
    {
        "task": "Python'da listeye eleman nasıl eklenir?",
        "answer": "Python'da listeye eleman eklemek için append metodu kullanılır.",
        "code": """
numbers = [1, 2, 3]
numbers.append(4)
print(numbers)
""",
    },
    {
        "task": "Python'da listeden eleman nasıl silinir?",
        "answer": "remove, listedeki belirli bir değeri silmek için kullanılır.",
        "code": """
fruits = ["elma", "armut", "muz"]
fruits.remove("armut")
print(fruits)
""",
    },
    {
        "task": "Python'da liste filtreleme nasıl yapılır?",
        "answer": "Liste filtrelemek için koşullu list comprehension kullanılabilir.",
        "code": """
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = [number for number in numbers if number % 2 == 0]
print(even_numbers)
""",
    },
    {
        "task": "Python'da list comprehension nasıl kullanılır?",
        "answer": "List comprehension, yeni liste üretmek için kısa ve okunabilir bir yoldur.",
        "code": """
numbers = [1, 2, 3, 4]
squares = [number * number for number in numbers]
print(squares)
""",
    },
    {
        "task": "Python'da sözlükten güvenli değer nasıl okunur?",
        "answer": "Sözlükten güvenli okumak için get metodu kullanılabilir.",
        "code": """
person = {"name": "Ali"}
age = person.get("age", 0)
print(age)
""",
    },
    {
        "task": "Python'da sözlüğe değer nasıl eklenir?",
        "answer": "Sözlükte yeni anahtar oluşturup değer atayabilirsin.",
        "code": """
person = {"name": "Ayşe"}
person["city"] = "İstanbul"
print(person)
""",
    },
    {
        "task": "Python'da sözlük ile kelime frekansı nasıl sayılır?",
        "answer": "Sözlük, kelime sayacı gibi kullanılabilir.",
        "code": """
words = ["veri", "model", "veri"]
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
print(counts)
""",
    },
    {
        "task": "Python'da set ile tekrarlar nasıl kaldırılır?",
        "answer": "set, tekrar eden değerleri tekilleştirmek için kullanılabilir.",
        "code": """
numbers = [1, 2, 2, 3, 3]
unique_numbers = set(numbers)
print(unique_numbers)
""",
    },
    {
        "task": "Python'da tuple nasıl kullanılır?",
        "answer": "Tuple, değişmemesi beklenen küçük veri grupları için kullanışlıdır.",
        "code": """
point = (10, 20)
x, y = point
print(x, y)
""",
    },
    {
        "task": "Python'da liste nasıl sıralanır?",
        "answer": "sorted, listeyi yeni bir sıralı liste olarak döndürür.",
        "code": """
items = [3, 1, 4, 2]
sorted_items = sorted(items)
print(sorted_items)
""",
    },
    {
        "task": "Python'da lambda temel olarak nasıl kullanılır?",
        "answer": "lambda, küçük anonim fonksiyonlar yazmak için kullanılabilir.",
        "code": """
double = lambda number: number * 2
print(double(5))
""",
    },
    {
        "task": "Python'da pathlib ile dosya nasıl okunur?",
        "answer": "Path.read_text, metin dosyasını okumak için kullanılabilir.",
        "code": """
from pathlib import Path

text = Path("notlar.txt").read_text(encoding="utf-8")
print(text)
""",
    },
    {
        "task": "Python'da pathlib ile dosya nasıl yazılır?",
        "answer": "Path.write_text, metin dosyasına yazmak için kullanılabilir.",
        "code": """
from pathlib import Path

Path("sonuc.txt").write_text("Merhaba Python", encoding="utf-8")
""",
    },
    {
        "task": "Python'da JSON metni nasıl okunur?",
        "answer": "json.loads, JSON metnini Python sözlüğüne dönüştürür.",
        "code": """
import json

raw_text = '{"name": "DarkMind", "version": 2}'
data = json.loads(raw_text)
print(data["name"])
""",
    },
    {
        "task": "Python'da JSON nasıl yazılır?",
        "answer": "json.dumps, Python verisini JSON metnine dönüştürür.",
        "code": """
import json

data = {"name": "DarkMind", "language": "tr"}
json_text = json.dumps(data, ensure_ascii=False)
print(json_text)
""",
    },
    {
        "task": "Python'da try except nasıl kullanılır?",
        "answer": "try/except, hata oluşabilecek kodu güvenli şekilde ele almak için kullanılır.",
        "code": """
try:
    number = int("42")
    print(number)
except ValueError as error:
    print(f"Dönüşüm hatası: {error}")
""",
    },
    {
        "task": "Python'da özel exception nasıl tanımlanır?",
        "answer": "Kendi hata türünü Exception sınıfından türeterek tanımlayabilirsin.",
        "code": """
class InvalidScoreError(Exception):
    pass

def check_score(score):
    if score < 0:
        raise InvalidScoreError("Skor negatif olamaz")
    return score
""",
    },
    {
        "task": "Python'da basit class nasıl yazılır?",
        "answer": "Class, ilgili veri ve davranışları bir arada tutar.",
        "code": """
class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Merhaba {self.name}"

user = User("Ali")
print(user.greet())
""",
    },
    {
        "task": "Python'da dataclass nasıl kullanılır?",
        "answer": "dataclass, veri taşıyan basit sınıfları daha kısa yazmayı sağlar.",
        "code": """
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

product = Product("Kitap", 120.0)
print(product.name)
""",
    },
    {
        "task": "Python'da input ile kullanıcıdan veri nasıl alınır?",
        "answer": "input fonksiyonu terminalden metin okumak için kullanılır.",
        "code": """
name = input("Adın nedir? ")
print(f"Merhaba {name}")
""",
    },
    {
        "task": "Python'da basit hesap makinesi nasıl yazılır?",
        "answer": "Basit hesap makinesi için sayıları ve işlem türünü kontrol edebilirsin.",
        "code": """
a = 10
b = 5
operation = "+"

if operation == "+":
    print(a + b)
elif operation == "-":
    print(a - b)
""",
    },
    {
        "task": "Python'da bir metnin palindrom olup olmadığını kontrol eden fonksiyon yaz.",
        "answer": "Palindrom kontrolünde metni sadeleştirip ters haliyle karşılaştırabilirsin.",
        "code": """
def is_palindrome(text):
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

print(is_palindrome("kazak"))
""",
    },
    {
        "task": "Python'da asal sayı kontrolü nasıl yapılır?",
        "answer": "Asal sayı kontrolünde 2'den kareköke kadar bölen aranabilir.",
        "code": """
def is_prime(number):
    if number < 2:
        return False
    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False
    return True

print(is_prime(17))
""",
    },
    {
        "task": "Python'da fibonacci dizisi nasıl üretilir?",
        "answer": "Fibonacci dizisinde her sayı kendinden önceki iki sayının toplamıdır.",
        "code": """
def fibonacci(count):
    numbers = []
    a, b = 0, 1
    for _ in range(count):
        numbers.append(a)
        a, b = b, a + b
    return numbers

print(fibonacci(7))
""",
    },
    {
        "task": "Python'da faktöriyel nasıl hesaplanır?",
        "answer": "Faktöriyel, 1'den n'e kadar sayıların çarpımıdır.",
        "code": """
def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result *= value
    return result

print(factorial(5))
""",
    },
    {
        "task": "Python'da metindeki kelime sayısı nasıl bulunur?",
        "answer": "split ile metni kelimelere ayırıp len ile sayabilirsin.",
        "code": """
text = "temiz veri güçlü model"
word_count = len(text.split())
print(word_count)
""",
    },
    {
        "task": "Python'da satır sayısı nasıl bulunur?",
        "answer": "splitlines, metni satırlarına ayırır.",
        "code": """
text = "birinci satır\\nikinci satır\\nüçüncü satır"
line_count = len(text.splitlines())
print(line_count)
""",
    },
    {
        "task": "Python'da CSV benzeri metin nasıl parçalanır?",
        "answer": "Dış bağımlılık olmadan virgüle göre split yapılabilir.",
        "code": """
line = "Ali,80,İstanbul"
name, score, city = [part.strip() for part in line.split(",")]
print(name, score, city)
""",
    },
    {
        "task": "Python'da basit assertion testi nasıl yazılır?",
        "answer": "assert, beklenen değerin doğru olduğunu kontrol etmek için kullanılabilir.",
        "code": """
def add_numbers(a, b):
    return a + b

# Küçük bir kontrol testi
assert add_numbers(2, 3) == 5
assert add_numbers(-1, 1) == 0
""",
    },
    {
        "task": "Python'da enumerate ne işe yarar?",
        "answer": "enumerate, döngüde hem sıra numarasını hem de elemanı verir.",
        "code": """
tasks = ["oku", "yaz", "test et"]
for index, task in enumerate(tasks, start=1):
    print(index, task)
""",
    },
]


def render_example(task: str, answer: str, code: str, variant_index: int) -> str:
    prompt = PROMPT_VARIANTS[variant_index].format(task=task)
    code = code.strip()

    if variant_index in {4, 7}:
        response = f"Asistan:\n\n```python\n{code}\n```"
    else:
        response = f"Asistan: {answer}\n\n```python\n{code}\n```"

    return f"Kullanıcı: {prompt}\n{response}"


def build_examples() -> list[str]:
    examples = []

    for spec in EXAMPLE_SPECS:
        for variant_index in range(len(PROMPT_VARIANTS)):
            examples.append(
                render_example(
                    task=spec["task"],
                    answer=spec["answer"],
                    code=spec["code"],
                    variant_index=variant_index,
                )
            )

    return examples


def main() -> None:
    examples = build_examples()

    if len(examples) < 300:
        raise ValueError(f"Expected at least 300 examples, got {len(examples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n\n".join(examples), encoding="utf-8")

    print(f"Generated examples: {len(examples)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
