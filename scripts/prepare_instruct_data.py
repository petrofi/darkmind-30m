from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
import random
import re
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

DEFAULT_OUT = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_v0_3.jsonl"
DEFAULT_SEED = ROOT_DIR / "data" / "instruct" / "darkmind_instruct_seed.jsonl"
IDENTITY_RESPONSE = (
    "Ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen küçük "
    "bir Türkçe dil modeli denemesiyim."
)
SOURCE_DATASETS = {
    "turkish_alpaca": "TFLai/Turkish-Alpaca",
    "openorca_tr": "malhajar/OpenOrca-tr",
    "alpaca_gpt4_tr": "malhajar/alpaca-gpt4-tr",
    "aya": "CohereLabs/aya_dataset",
    "turkish_instructions": "merve/turkish_instructions",
    "atasoglu": "atasoglu/instruction-turkish",
    "turkish_distilled": "afkfatih/turkish-distilled-5K",
    "diyalog": "alibayram/diyalog-dataset",
}
DEFAULT_SOURCES = ",".join(SOURCE_DATASETS)
TARGET_MIX = {
    "identity": 0.15,
    "programming": 0.25,
    "ai": 0.20,
    "general": 0.15,
    "emotional_support": 0.10,
    "career": 0.10,
    "explanation": 0.05,
}
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_WORDS = {
    "ve", "bir", "bu", "için", "nedir", "nasıl", "olarak", "ile", "çok",
    "daha", "kısa", "açıkla", "örnek", "yardım", "model", "veri", "kod",
}
BAD_IDENTITY_PHRASES = [
    "ben chatgpt",
    "chatgpt olarak",
    "openai tarafından geliştirildim",
    "openai tarafindan gelistirildim",
    "openai geliştirdi",
    "openai gelistirdi",
    "as an ai language model",
]
SECRET_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\s().-]*){9,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret|token|password|passwd|bearer)\b", re.IGNORECASE),
    re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,})\b"),
]
DISALLOWED_TOPIC_WORDS = [
    "porn", "porno", "seks", "çıplak", "erotik", "bomba yap", "silah yap",
    "uyuşturucu", "nefret", "ırkçı", "öldür", "intihar yöntemi",
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "evet"}


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_turkish(prompt: str, response: str) -> bool:
    text = f" {prompt} {response} ".lower()

    if any(char in text for char in TURKISH_CHARS):
        return True

    words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower()))
    return len(words & TURKISH_WORDS) >= 3


def too_many_special_chars(text: str) -> bool:
    if not text:
        return True

    special = sum(1 for char in text if not char.isalnum() and not char.isspace() and char not in ".,:;!?-_'\"()[]/%")
    return special / max(len(text), 1) > 0.18


def has_blocked_content(prompt: str, response: str) -> bool:
    text = f"{prompt}\n{response}".lower()

    if any(phrase in text for phrase in BAD_IDENTITY_PHRASES):
        return True

    if any(word in text for word in DISALLOWED_TOPIC_WORDS):
        return True

    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def categorize(prompt: str, response: str) -> str:
    text = f" {prompt} {response} ".lower()

    if any(word in text for word in ["kimsin", "chatgpt", "openai", "darkmind", "kim geliştirdi", "kendini tanıt", "hangi model"]):
        return "identity"
    if any(word in text for word in ["python", "kod", "api", "backend", "frontend", "docker", "git", "sql", "json", "terminal", "programlama"]):
        return "programming"
    if any(word in text for word in ["yapay zeka", "model", "token", "llm", "transformer", "fine-tuning", "öğrenmesi", "loss", "veri seti"]):
        return "ai"
    if any(word in text for word in ["kötü hissediyorum", "motivasyon", "stres", "moral", "kaygı", "yalnız", "zorlanıyorum", "yorgun"]):
        return "emotional_support"
    if any(word in text for word in ["kariyer", "öğren", "çalışma plan", "staj", "mülakat", "cv", "portfolyo", "study"]):
        return "career"
    if any(word in text for word in ["kısaca", "basitçe", "örnek", "özet", "maddelerle", "açıkla"]):
        return "explanation"
    if any(word in text for word in ["türkiye", "istanbul", "üniversite", "bilim", "internet", "tarih", "coğrafya"]):
        return "general"

    return "other"


def is_good_example(
    prompt: str,
    response: str,
    min_response_chars: int,
    max_response_chars: int,
    max_prompt_chars: int,
) -> bool:
    if not prompt or not response:
        return False
    if len(prompt) > max_prompt_chars:
        return False
    if len(response) < min_response_chars or len(response) > max_response_chars:
        return False
    if not looks_turkish(prompt, response):
        return False
    if has_blocked_content(prompt, response):
        return False
    if too_many_special_chars(prompt) or too_many_special_chars(response):
        return False
    if len(response.split()) < 3:
        return False
    return True


def example_hash(prompt: str, response: str) -> str:
    payload = f"{prompt.strip().lower()}\n{response.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_example(
    prompt: str,
    response: str,
    source: str,
    category: str | None = None,
) -> dict:
    prompt = clean_text(prompt)
    response = clean_text(response)
    return {
        "prompt": prompt,
        "response": response,
        "source": source,
        "category": category or categorize(prompt, response),
    }


def add_if_valid(
    bucket: dict[str, list[dict]],
    seen: set[str],
    example: dict,
    args: argparse.Namespace,
) -> bool:
    prompt = example["prompt"]
    response = example["response"]

    if not is_good_example(
        prompt=prompt,
        response=response,
        min_response_chars=args.min_response_chars,
        max_response_chars=args.max_response_chars,
        max_prompt_chars=args.max_prompt_chars,
    ):
        return False

    key = example_hash(prompt, response)

    if key in seen:
        return False

    seen.add(key)
    bucket[example["category"]].append(example)
    return True


def load_seed_examples(path: Path) -> list[dict]:
    if not path.exists():
        return []

    examples = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if prompt and response:
                examples.append(make_example(prompt, response, "local_seed"))

    return examples


def synthetic_identity_examples(limit: int = 320) -> list[dict]:
    prompts = [
        "Sen kimsin?", "Kimsin?", "Merhaba, sen kimsin?", "Selam, sen kimsin?",
        "Kendini tanıtır mısın?", "Bana kendinden bahset.", "Adın ne?",
        "Hangi modelsin?", "DarkMind nedir?", "DarkMind'i açıkla.",
        "Seni kim geliştirdi?", "Kim tarafından geliştirildin?",
        "Tarık Yasin Sağlıcak kimdir bu projede?", "Bu modelin geliştiricisi kim?",
        "ChatGPT misin?", "Sen ChatGPT misin?", "ChatGPT gibi misin?",
        "OpenAI tarafından mı geliştirildin?", "Seni OpenAI mı geliştirdi?",
        "OpenAI modeli misin?", "Gemini misin?", "Claude musun?",
        "Hazır büyük bir model misin?", "Üretim için hazır mısın?",
        "Her şeyi biliyor musun?", "Kesin doğru cevap verir misin?",
        "Sınırların var mı?", "Yanlış cevap verebilir misin?",
        "Türkçe model misin?", "Amacın ne?", "Ne için geliştirildin?",
        "Bana nasıl yardımcı olabilirsin?", "Kendini tek cümleyle tanıt.",
        "Kimlik cevabın ne olmalı?", "Kendini doğru tanıt.",
        "Bu bir öğrenme projesi mi?", "Büyük modellerle aynı mısın?",
        "ChatGPT seviyesinde misin?", "Kişisel bilgilerimi biliyor musun?",
    ]
    prefixes = ["", "Lütfen cevapla: ", "Kısa cevap ver: ", "Açıkça söyle: "]
    examples = []

    for prefix in prefixes:
        for prompt in prompts:
            actual_prompt = f"{prefix}{prompt}".strip()
            lower = actual_prompt.lower()

            if any(word in lower for word in ["chatgpt", "openai", "gemini", "claude"]):
                response = f"Hayır. {IDENTITY_RESPONSE}"
            elif any(word in lower for word in ["her şeyi", "kesin", "sınır", "yanlış"]):
                response = (
                    "Sınırlı bir modelim ve yanlış cevap verebilirim. "
                    f"{IDENTITY_RESPONSE}"
                )
            elif "kişisel" in lower:
                response = (
                    "Hayır. Bu sohbet dışında kişisel bilgilerini bilmem. "
                    f"{IDENTITY_RESPONSE}"
                )
            else:
                response = IDENTITY_RESPONSE

            examples.append(make_example(actual_prompt, response, "local_identity", "identity"))

    numbered = []
    index = 1

    while len(numbered) < limit:
        base = examples[(index - 1) % len(examples)]
        prompt = base["prompt"]
        response = base["response"]

        if index > len(examples):
            prompt = prompt.replace("?", f" ({index})?") if "?" in prompt else f"{prompt} ({index})"

        numbered.append(make_example(prompt, response, "local_identity", "identity"))
        index += 1

    return numbered[:limit]


def local_fallback_examples() -> list[dict]:
    examples = []
    programming = {
        "Python": "Python, okunabilir sözdizimine sahip genel amaçlı bir programlama dilidir.",
        "API": "API, iki yazılımın belirli kurallar üzerinden iletişim kurmasını sağlayan arayüzdür.",
        "backend": "Backend, uygulamanın sunucu tarafında çalışan iş mantığı ve veri işlemleridir.",
        "frontend": "Frontend, kullanıcının gördüğü ve etkileşim kurduğu arayüz bölümüdür.",
        "Docker": "Docker, uygulamaları bağımlılıklarıyla birlikte konteyner içinde çalıştırmaya yarar.",
        "Git": "Git, kod değişikliklerini takip eden bir sürüm kontrol sistemidir.",
        "SQL": "SQL, ilişkisel veritabanlarında veri sorgulamak ve yönetmek için kullanılan dildir.",
        "JSON": "JSON, verileri anahtar ve değer yapısıyla saklayan hafif bir metin formatıdır.",
        "fonksiyon": "Fonksiyon, belirli bir işi yapan ve tekrar çağrılabilen kod bloğudur.",
        "döngü": "Döngü, bir işlemi belirli koşullarla tekrar çalıştıran programlama yapısıdır.",
        "değişken": "Değişken, program içinde bir değeri saklamak için kullanılan isimdir.",
        "hata ayıklama": "Hata ayıklama, koddaki sorunları bulma ve düzeltme sürecidir.",
        "test": "Test, kodun beklenen şekilde çalışıp çalışmadığını kontrol eder.",
        "refactor": "Refactor, kodun davranışını değiştirmeden yapısını iyileştirmedir.",
        "terminal": "Terminal, komut yazarak programlarla etkileşim kurulan metin tabanlı arayüzdür.",
    }
    ai = {
        "yapay zeka": "Yapay zeka, makinelerin verilerden örüntüler öğrenerek tahmin veya üretim yapmasını sağlayan yöntemlerin genel adıdır.",
        "dil modeli": "Dil modeli, verilen metnin devamını tahmin etmeyi öğrenen yapay zeka modelidir.",
        "token": "Token, modelin metni işlerken kullandığı küçük metin parçasıdır.",
        "tokenizer": "Tokenizer, metni modelin anlayacağı token kimliklerine dönüştürür.",
        "Transformer": "Transformer, dikkat mekanizması kullanan modern bir sinir ağı mimarisidir.",
        "fine-tuning": "Fine-tuning, önceden eğitilmiş modeli özel veriyle belirli davranışlara uyarlamaktır.",
        "overfitting": "Overfitting, modelin eğitim verisini ezberleyip yeni örneklerde zayıf kalmasıdır.",
        "checkpoint": "Checkpoint, modelin belirli bir andaki ağırlıklarını ve eğitim bilgilerini saklayan dosyadır.",
        "loss": "Loss, model tahmininin hedefe ne kadar uzak olduğunu gösteren hata ölçüsüdür.",
        "validation loss": "Validation loss, modelin eğitimde görmediği verideki hata değeridir.",
    }
    general = {
        "Türkiye": "Türkiye, Avrupa ve Asya arasında yer alan, başkenti Ankara olan bir ülkedir.",
        "İstanbul": "İstanbul, tarihi ve kültürel önemi yüksek büyük bir Türkiye şehridir.",
        "üniversite": "Üniversite, yükseköğretim ve araştırma yapılan eğitim kurumudur.",
        "bilim": "Bilim, gözlem ve akıl yürütmeyle olayları anlamaya çalışan sistemli bilgidir.",
        "internet": "İnternet, dünya genelindeki bilgisayarları birbirine bağlayan büyük iletişim ağıdır.",
        "teknoloji": "Teknoloji, işleri kolaylaştırmak için geliştirilen araç ve yöntemlerin bütünüdür.",
        "kütüphane": "Kütüphane, kitap ve bilgi kaynaklarına erişim sağlayan öğrenme alanıdır.",
        "iletişim": "İletişim, duygu, düşünce veya bilginin kişiler arasında aktarılmasıdır.",
    }
    support = [
        ("Bugün kendimi kötü hissediyorum.", "Bunu yaşaman zor olabilir. Küçük bir mola vermek ve güvendiğin biriyle konuşmak iyi gelebilir."),
        ("Motivasyonum düştü.", "Motivasyon bazen düşer. Bugün atabileceğin tek küçük adıma odaklanabilirsin."),
        ("Çalışmaya devam etmekte zorlanıyorum.", "Kısa bir ara verip sonra on dakikalık küçük bir görev seçmek daha yönetilebilir olabilir."),
        ("Çok stresliyim.", "Önce nefesini yavaşlatmayı dene. Sonra en acil tek işi seçmek stresi azaltabilir."),
        ("Hata yapmaktan korkuyorum.", "Hata yapmak öğrenmenin doğal parçasıdır. Küçük denemelerle riski azaltıp ilerleyebilirsin."),
    ]
    career = [
        ("Yazılıma nasıl başlamalıyım?", "Temel Python, algoritma mantığı ve küçük projelerle başlamak iyi olur."),
        ("Python öğrenmek için plan öner.", "Önce değişken, koşul, döngü ve fonksiyonları öğren. Sonra küçük projeler yap."),
        ("Portfolyo nasıl hazırlanır?", "Az sayıda ama çalışan projeler seç. Her proje için amaç ve kurulum bilgisini yaz."),
        ("Teknik mülakata nasıl hazırlanırım?", "Temel kavramları tekrar et, küçük kod soruları çöz ve projelerini açıklamaya hazırlan."),
        ("Çalışma planı nasıl kurulur?", "Haftalık hedef belirle, günlük küçük görevler yaz ve sonunda kısa değerlendirme yap."),
    ]
    styles = [
        ("Kısa cevap ver.", "Tamam. Kısa ve net cevap vermeye çalışırım."),
        ("Basitçe anlat.", "Elbette. Konuyu sade kelimelerle ve gereksiz ayrıntıya girmeden anlatırım."),
        ("Örnek vererek anlat.", "Önce kısa açıklama yapar, sonra küçük bir örnekle konuyu somutlaştırırım."),
        ("Maddelerle yaz.", "Bilgiyi kısa ve düzenli maddeler halinde yazabilirim."),
        ("Güvenli sınırlarla cevap ver.", "Riskli alanlarda kesin iddia yerine genel ve güvenli bilgiyle sınırlı kalırım."),
    ]

    for name, response in programming.items():
        for prompt in [f"{name} nedir?", f"{name} ne işe yarar?", f"{name} konusunu kısaca açıkla."]:
            examples.append(make_example(prompt, response, "local_fallback", "programming"))

    for name, response in ai.items():
        for prompt in [f"{name} nedir?", f"{name} kavramını açıkla.", f"{name} neden önemlidir?"]:
            examples.append(make_example(prompt, response, "local_fallback", "ai"))

    for name, response in general.items():
        for prompt in [f"{name} nedir?", f"{name} hakkında kısa bilgi ver."]:
            examples.append(make_example(prompt, response, "local_fallback", "general"))

    for prompt, response in support:
        for suffix in ["", " Ne yapabilirim?", " Bana kısa destek ver."]:
            examples.append(make_example(f"{prompt}{suffix}", response, "local_fallback", "emotional_support"))

    for prompt, response in career:
        for suffix in ["", " Kısa öneri ver.", " Nereden başlamalıyım?"]:
            examples.append(make_example(f"{prompt}{suffix}", response, "local_fallback", "career"))

    for prompt, response in styles:
        for suffix in ["", " Lütfen.", " Bu şekilde cevapla."]:
            examples.append(make_example(f"{prompt}{suffix}", response, "local_fallback", "explanation"))

    return examples


def extract_messages_pairs(messages: object) -> list[tuple[str, str]]:
    if not isinstance(messages, list):
        return []

    pairs = []
    pending_user = None

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", message.get("from", ""))).lower()
        content = clean_text(
            message.get("content", message.get("value", message.get("text", "")))
        )

        if not content:
            continue

        if role in {"user", "human", "instruction"}:
            pending_user = content
        elif role in {"assistant", "gpt", "bot"} and pending_user:
            pairs.append((pending_user, content))
            pending_user = None

    return pairs


def normalize_row(row: dict) -> list[tuple[str, str]]:
    for key in ["messages", "conversations"]:
        pairs = extract_messages_pairs(row.get(key))
        if pairs:
            return pairs

    prompt = ""
    response = ""

    for key in ["instruction", "prompt", "question", "query", "input", "inputs"]:
        if row.get(key):
            prompt = clean_text(row.get(key))
            break

    instruction = clean_text(row.get("instruction", ""))
    input_text = clean_text(row.get("input", ""))

    if instruction and input_text and input_text != instruction:
        prompt = f"{instruction}\n{input_text}"

    for key in ["output", "response", "completion", "answer", "target", "targets"]:
        if row.get(key):
            response = clean_text(row.get(key))
            break

    if prompt and response:
        return [(prompt, response)]

    return []


def iter_dataset_rows(dataset_name: str, source_name: str, scan_limit: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is not installed") from exc

    last_error = None

    for split in ["train", "validation", "test"]:
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            for index, row in enumerate(dataset):
                if index >= scan_limit:
                    break
                if isinstance(row, dict):
                    yield row
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    for split in ["train[:2000]", "validation[:1000]", "test[:1000]"]:
        try:
            dataset = load_dataset(dataset_name, split=split)
            for index, row in enumerate(dataset):
                if index >= scan_limit:
                    break
                if isinstance(row, dict):
                    yield row
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"{source_name} failed to load: {last_error}")


def collect_external_examples(
    args: argparse.Namespace,
    bucket: dict[str, list[dict]],
    seen: set[str],
) -> tuple[Counter, list[dict]]:
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    accepted_by_source = Counter()
    skipped = []

    if args.max_examples <= 1000:
        return accepted_by_source, [
            {
                "source": source,
                "reason": "skipped during <=1000 smoke run to avoid slow external dataset downloads",
            }
            for source in sources
        ]

    scan_limit = max(1000, min(8000, args.max_examples * 4))

    for source in sources:
        dataset_name = SOURCE_DATASETS.get(source)

        if not dataset_name:
            skipped.append({"source": source, "reason": "unknown source"})
            continue

        before = sum(len(items) for items in bucket.values())

        try:
            for row in iter_dataset_rows(dataset_name, source, scan_limit):
                for prompt, response in normalize_row(row):
                    category = categorize(prompt, response)
                    example = make_example(prompt, response, source, category)

                    if add_if_valid(bucket, seen, example, args):
                        accepted_by_source[source] += 1

                if sum(accepted_by_source.values()) >= args.max_examples:
                    break
        except Exception as exc:  # noqa: BLE001
            skipped.append({"source": source, "reason": str(exc)[:500]})
            continue

        after = sum(len(items) for items in bucket.values())

        if after == before:
            skipped.append({"source": source, "reason": "no usable clean examples found"})

    return accepted_by_source, skipped


def add_local_examples(args: argparse.Namespace, bucket: dict[str, list[dict]], seen: set[str]) -> Counter:
    counts = Counter()

    for example in synthetic_identity_examples(320):
        if add_if_valid(bucket, seen, example, args):
            counts[example["source"]] += 1

    if args.include_seed:
        for example in load_seed_examples(DEFAULT_SEED):
            if add_if_valid(bucket, seen, example, args):
                counts[example["source"]] += 1

    fallback = local_fallback_examples()
    fallback_round = 0

    while sum(len(items) for items in bucket.values()) < min(args.max_examples, 1200):
        added_this_round = 0
        fallback_round += 1

        for example in fallback:
            if fallback_round == 1:
                candidate = example
            else:
                candidate = make_example(
                    prompt=f"{example['prompt']} Kısa ve net cevapla. ({fallback_round})",
                    response=example["response"],
                    source="local_fallback",
                    category=example["category"],
                )

            if add_if_valid(bucket, seen, candidate, args):
                counts[candidate["source"]] += 1
                added_this_round += 1

            if sum(len(items) for items in bucket.values()) >= args.max_examples:
                break

        if added_this_round == 0 or fallback_round >= 10:
            break

    return counts


def select_examples(bucket: dict[str, list[dict]], max_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    for items in bucket.values():
        rng.shuffle(items)

    selected = []
    selected_ids = set()

    identity_items = bucket.get("identity", [])
    for example in identity_items[: min(len(identity_items), 320, max_examples)]:
        selected.append(example)
        selected_ids.add(id(example))

    remaining_slots = max_examples - len(selected)

    for category, ratio in TARGET_MIX.items():
        if remaining_slots <= 0:
            break

        target = int(max_examples * ratio)
        current = sum(1 for example in selected if example["category"] == category)
        needed = max(0, target - current)

        for example in bucket.get(category, []):
            if needed <= 0 or remaining_slots <= 0:
                break
            if id(example) in selected_ids:
                continue
            selected.append(example)
            selected_ids.add(id(example))
            needed -= 1
            remaining_slots -= 1

    all_remaining = [
        example
        for items in bucket.values()
        for example in items
        if id(example) not in selected_ids
    ]
    rng.shuffle(all_remaining)

    for example in all_remaining:
        if len(selected) >= max_examples:
            break
        selected.append(example)
        selected_ids.add(id(example))

    return selected[:max_examples]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def write_meta(path: Path, examples: list[dict], skipped: list[dict]) -> None:
    meta_path = path.with_suffix(".meta.json")
    payload = {
        "output": str(path),
        "example_count": len(examples),
        "sources": Counter(example["source"] for example in examples),
        "categories": Counter(example["category"] for example in examples),
        "skipped_sources": skipped,
    }
    serializable = {
        **payload,
        "sources": dict(payload["sources"]),
        "categories": dict(payload["categories"]),
    }
    meta_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a controlled Turkish instruction dataset for DarkMind."
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT.relative_to(ROOT_DIR)))
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--sources", type=str, default=DEFAULT_SOURCES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_response_chars", type=int, default=20)
    parser.add_argument("--max_response_chars", type=int, default=800)
    parser.add_argument("--max_prompt_chars", type=int, default=500)
    parser.add_argument("--include_seed", type=parse_bool, default=True)
    args = parser.parse_args()

    if args.max_examples < 1:
        raise ValueError("--max_examples must be at least 1")

    random.seed(args.seed)
    out_path = resolve_path(args.out)
    bucket: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    local_counts = add_local_examples(args, bucket, seen)
    external_counts, skipped = collect_external_examples(args, bucket, seen)
    selected = select_examples(bucket, args.max_examples, args.seed)
    write_jsonl(out_path, selected)
    write_meta(out_path, selected, skipped)

    print("=" * 70)
    print("DarkMind instruction dataset prepared")
    print("=" * 70)
    print(f"Output: {out_path}")
    print(f"Examples: {len(selected):,}")
    print(f"Local accepted: {sum(local_counts.values()):,}")
    print(f"External accepted before mixing: {sum(external_counts.values()):,}")
    print("Sources:")
    for source, count in Counter(example["source"] for example in selected).most_common():
        print(f"  {source}: {count:,}")
    print("Categories:")
    for category, count in Counter(example["category"] for example in selected).most_common():
        print(f"  {category}: {count:,}")
    print("Skipped sources:")
    for item in skipped:
        print(f"  {item['source']}: {item['reason']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
