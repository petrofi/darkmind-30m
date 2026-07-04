from __future__ import annotations

import json
import re
import sys
from pathlib import Path


OUTPUT_PATH = Path("data/instruct/ai_instruct_seed.jsonl")
META_PATH = Path("data/instruct/ai_instruct_seed.meta.json")
SOURCE = "local_ai_seed"
CATEGORY = "ai"
TARGET_ROWS = 200

BLOCKED_IDENTITY_RE = re.compile(
    r"\b(chatgpt|openai|as an ai language model|i am chatgpt|ben chatgpt)\b",
    re.IGNORECASE,
)

TOPICS = [
    {
        "name": "yapay zeka",
        "short": "Yapay zeka, bilgisayarların verilerden örüntü öğrenerek tahmin, sınıflandırma veya metin üretme gibi görevleri yapmasını sağlar.",
        "pitfall": "Tek başına sihirli bir çözüm değildir; veri kalitesi, görev tanımı ve değerlendirme yöntemi sonucu doğrudan etkiler.",
        "example": "Bir destek aracında yapay zeka, gelen soruyu sınıflandırıp uygun yanıt taslağı önerebilir.",
    },
    {
        "name": "dil modeli",
        "short": "Dil modeli, verilen metnin devamında hangi tokenların gelebileceğini öğrenen bir model türüdür.",
        "pitfall": "Küçük dil modelleri akıcı görünebilir ama bilgi, muhakeme ve yönerge takibinde kolayca hata yapabilir.",
        "example": "Bir Türkçe dil modeli, 'Python nedir?' sorusuna kısa bir açıklama üretmek için eğitilebilir.",
    },
    {
        "name": "token",
        "short": "Token, metnin model tarafından işlenen küçük parçasıdır; bazen kelime, bazen kelime parçası olabilir.",
        "pitfall": "Token sayısı arttıkça eğitim ve çıkarım maliyeti de artar.",
        "example": "Türkçe ekler nedeniyle bir kelime birkaç tokene bölünebilir.",
    },
    {
        "name": "tokenizer",
        "short": "Tokenizer, ham metni modelin anlayacağı token kimliklerine çeviren bileşendir.",
        "pitfall": "Kötü tokenizer seçimi Türkçe karakterleri ve ekleri verimsiz temsil edebilir.",
        "example": "Byte-level BPE tokenizer, bilinmeyen karakterlerde daha dayanıklı olabilir.",
    },
    {
        "name": "pretraining",
        "short": "Pretraining, modelin büyük metinlerden genel dil örüntülerini öğrendiği ilk eğitim aşamasıdır.",
        "pitfall": "Pretraining tek başına sohbet davranışı öğretmez; model daha çok metin devam ettirmeyi öğrenir.",
        "example": "Türkçe web metinleriyle pretraining, temel akıcılığı artırabilir.",
    },
    {
        "name": "instruction tuning",
        "short": "Instruction tuning, modele kullanıcı isteğine cevap verme biçimini örneklerle öğreten denetimli ince ayardır.",
        "pitfall": "Küçük ve tekrarlı veri, modelin ezberci veya kalıplaşmış cevap vermesine yol açabilir.",
        "example": "JSONL satırlarında prompt ve response alanlarıyla kısa yönerge-cevap örnekleri tutulabilir.",
    },
    {
        "name": "validation loss",
        "short": "Validation loss, modelin eğitimde görmediği doğrulama örneklerinde ne kadar hata yaptığını gösterir.",
        "pitfall": "Çok düşük validation loss her zaman iyi genelleme anlamına gelmez; veri sızıntısı veya benzer şablonlar olabilir.",
        "example": "Train loss düşerken validation loss yükseliyorsa overfitting başlamış olabilir.",
    },
    {
        "name": "checkpoint",
        "short": "Checkpoint, model ağırlıklarının ve bazen eğitim durumunun kaydedilmiş halidir.",
        "pitfall": "Yanlış checkpoint üstüne eğitim yapmak önceki deney sonuçlarını karıştırabilir.",
        "example": "Eğitimden sonra en iyi validation loss veren checkpoint ayrı bir dosyada saklanabilir.",
    },
    {
        "name": "overfitting",
        "short": "Overfitting, modelin eğitim verisini fazla ezberleyip yeni örneklerde zayıf kalmasıdır.",
        "pitfall": "Az ve tekrarlı instruction verisi overfitting riskini artırır.",
        "example": "Aynı identity cevabının çok fazla tekrar etmesi modelin ilgisiz sorulara da kimlik cevabı vermesine neden olabilir.",
    },
    {
        "name": "transformer",
        "short": "Transformer, attention mekanizmasıyla tokenlar arasındaki ilişkileri öğrenen model mimarisidir.",
        "pitfall": "Parametre sayısı küçükse transformer mimarisi tek başına güçlü muhakeme garantisi vermez.",
        "example": "GPT tipi modeller transformer bloklarından oluşur.",
    },
    {
        "name": "PyTorch",
        "short": "PyTorch, tensör işlemleri ve otomatik türev desteğiyle model eğitmek için kullanılan bir kütüphanedir.",
        "pitfall": "Tensor boyutlarını yanlış ayarlamak eğitim sırasında shape hatalarına neden olabilir.",
        "example": "Bir batch tensörü genellikle batch_size ve sequence_length boyutlarını içerir.",
    },
    {
        "name": "CUDA",
        "short": "CUDA, uygun NVIDIA GPU üzerinde model eğitimini ve çıkarımı hızlandırmak için kullanılır.",
        "pitfall": "GPU belleği yetmezse batch size veya block size küçültülmelidir.",
        "example": "PyTorch'ta model ve tensorlar aynı cuda cihazına taşınmalıdır.",
    },
    {
        "name": "JSONL",
        "short": "JSONL, her satırda ayrı bir JSON nesnesi tutan sade bir veri formatıdır.",
        "pitfall": "Bozuk bir satır tüm veri hazırlama adımını durdurabilir; satır satır doğrulama yapılmalıdır.",
        "example": "Instruction verisi için her satırda prompt, response, source ve category alanları tutulabilir.",
    },
    {
        "name": "dataset cleaning",
        "short": "Dataset cleaning, boş, tekrar eden, özel bilgi içeren veya görevle ilgisiz örnekleri ayıklama sürecidir.",
        "pitfall": "Gürültülü veri küçük modellerde yanlış davranışı hızlıca güçlendirebilir.",
        "example": "URL, e-posta, telefon ve kimlik iddiası içeren satırlar eğitimden çıkarılabilir.",
    },
    {
        "name": "batch size",
        "short": "Batch size, eğitimde aynı anda işlenen örnek sayısını ifade eder.",
        "pitfall": "Batch size büyüdükçe bellek ihtiyacı artar; çok küçük olursa loss daha dalgalı olabilir.",
        "example": "Küçük GPU belleğinde batch size 4 gibi düşük değerler tercih edilebilir.",
    },
    {
        "name": "learning rate",
        "short": "Learning rate, model ağırlıklarının her adımda ne kadar güncelleneceğini belirler.",
        "pitfall": "Çok yüksek learning rate loss patlamasına, çok düşük learning rate yavaş öğrenmeye yol açabilir.",
        "example": "Fine-tuning sırasında küçük bir learning rate daha kontrollü olabilir.",
    },
    {
        "name": "eval set",
        "short": "Eval set, modelin eğitim dışında sabit sorularla nasıl davrandığını izlemek için kullanılır.",
        "pitfall": "Eval soruları eğitim verisine sızarsa sonuçlar olduğundan iyi görünür.",
        "example": "Kimlik, yazılım ve destek soruları ayrı kategorilerle değerlendirilebilir.",
    },
]

PROMPT_TEMPLATES = [
    "{topic} nedir?",
    "{topic} kavramını kısa açıkla.",
    "{topic} neden önemlidir?",
    "{topic} için küçük bir örnek ver.",
    "{topic} kullanırken yaygın bir risk nedir?",
    "Bir junior geliştiriciye {topic} nasıl anlatılır?",
    "{topic} ile ilgili dikkat edilmesi gereken nokta nedir?",
    "{topic} DarkMind eğitim hattında ne işe yarar?",
    "{topic} hakkında iki cümlelik açıklama yaz.",
    "{topic} öğrenirken ilk bakılacak şey nedir?",
    "{topic} için pratik bir kontrol öner.",
    "{topic} küçük modellerde neden önemlidir?",
]


def response_for(topic: dict[str, str], template_index: int) -> str:
    name = topic["name"]
    if template_index == 0:
        return topic["short"]
    if template_index == 1:
        return f"{topic['short']} Kısa cevaplarda amaç, kavramı sade tutup gereksiz ayrıntıya girmemektir."
    if template_index == 2:
        return f"{name} önemlidir çünkü eğitim kalitesi, model davranışı veya ölçüm güvenilirliği üzerinde doğrudan etkisi vardır."
    if template_index == 3:
        return topic["example"]
    if template_index == 4:
        return topic["pitfall"]
    if template_index == 5:
        return f"Bir junior geliştiriciye {name}, önce küçük bir örnekle ve sonra tek bir riskle anlatılmalıdır. Özet: {topic['short']}"
    if template_index == 6:
        return f"{name} için dikkat edilecek nokta şudur: {topic['pitfall']}"
    if template_index == 7:
        return f"DarkMind hattında {name}, veri hazırlama, eğitim veya değerlendirme kararlarını daha kontrollü hale getirmek için kullanılır."
    if template_index == 8:
        return f"{topic['short']} {topic['pitfall']}"
    if template_index == 9:
        return f"{name} öğrenirken önce tanımı, sonra küçük bir örneği incelemek iyi olur. Sonraki adım olarak tek bir pratik kullanım yazılabilir."
    if template_index == 10:
        return f"Pratik kontrol: {name} içeren örnekte giriş, beklenen çıktı ve olası hata kaynağını ayrı ayrı not et."
    return f"Küçük modellerde {name} önemlidir çünkü az veri ve düşük kapasite hatalı örüntüleri daha hızlı öğrenebilir."


def build_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for topic in TOPICS:
        for template_index, prompt_template in enumerate(PROMPT_TEMPLATES):
            prompt = prompt_template.format(topic=topic["name"])
            response = response_for(topic, template_index)
            key = (prompt.casefold(), response.casefold())
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "source": SOURCE,
                    "category": CATEGORY,
                }
            )
            if len(examples) >= TARGET_ROWS:
                return examples
    return examples


def validate_examples(examples: list[dict[str, str]]) -> dict[str, int]:
    duplicate_count = 0
    blocked_identity_count = 0
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = (example["prompt"].casefold(), example["response"].casefold())
        if key in seen:
            duplicate_count += 1
        seen.add(key)
        if BLOCKED_IDENTITY_RE.search(f"{example['prompt']}\n{example['response']}"):
            blocked_identity_count += 1
    return {
        "duplicate_count": duplicate_count,
        "blocked_identity_phrase_count": blocked_identity_count,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    validation = validate_examples(examples)
    if validation["duplicate_count"] or validation["blocked_identity_phrase_count"]:
        raise ValueError(f"AI seed validation failed: {validation}")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    metadata = {
        "source": SOURCE,
        "category": CATEGORY,
        "accepted_rows": len(examples),
        "topics": [topic["name"] for topic in TOPICS],
        "completed": 150 <= len(examples) <= 250,
        "first_10_samples": examples[:10],
        **validation,
    }
    with META_PATH.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
