from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


SOURCES = [
    "ucekmez/OpenOrca-tr",
    "cgulse/alpaca-cleaned-tr",
    "TFLai/Turkish-Alpaca",
    "malhajar/alpaca-gpt4-tr",
]
TARGET_EXAMPLES = 1000
MAX_SCAN_ROWS_PER_SOURCE = 1000
MAX_LOCAL_SUPPORT_SEED = 220
OUTPUT_PATH = Path("data/instruct/support_instruct_1k.jsonl")
META_PATH = Path("data/instruct/support_instruct_1k.meta.json")

IDENTITY_PATTERNS = [
    re.compile(r"chatgpt", re.IGNORECASE),
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"as an ai language model", re.IGNORECASE),
    re.compile(r"i am chatgpt", re.IGNORECASE),
    re.compile(r"ben chatgpt", re.IGNORECASE),
]
SUPPORT_PROMPT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"kendimi .{0,30}(kötü|kotu|yetersiz|yalnız|yalniz|mutsuz|üzgün|uzgun)",
        r"moralim .{0,30}(bozuk|düşük|dusuk)",
        r"motivasyon\w* .{0,30}(düştü|dustu|azaldı|azaldi|yok)",
        r"çok yoruldum|cok yoruldum|tükendim|tukendim",
        r"devam edemiyorum|odaklanamıyorum|odaklanamiyorum",
        r"iş bulamıyorum|is bulamiyorum|işsiz|issiz",
        r"projem .{0,40}(ilerlemiyor|takıldı|takildi)",
        r"başarısız hissediyorum|basarisiz hissediyorum|başaramıyorum|basaramiyorum",
        r"kaygılıyım|kaygiliyim|endişeliyim|endiseliyim|stresliyim",
        r"hata aldıkça|hata aldikca|kod yazarken .{0,30}takılıyorum|takiliyorum",
        r"kendime güvenim|kendime guvenim|hiçbir şey yapasım yok|hicbir sey yapasim yok",
        r"bunaldım|bunaldim|panikliyorum|geriliyorum|başlayamıyorum|baslayamiyorum",
        r"geride hissediyorum|ağır geliyor|agir geliyor|toparlanmakta zorlanıyorum|zorlanıyorum",
        r"bitiremeyecekmişim|bitiremeyecekmisim|yapamıyorum|yapamiyorum",
        r"ne yapmalıyım|ne yapmaliyim",
    ]
]
SUPPORT_RESPONSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"küçük bir adım|kucuk bir adim|bir sonraki adım|bir sonraki adim",
        r"küçük bir görev|kucuk bir gorev|hedefi küçült|hedefi kucult",
        r"tek adım|tek adim|zamanlayıcı|zamanlayici|küçük başlangıç|kucuk baslangic",
        r"yeterli olabilir|yan yana yaz|bir cümleyle|bir cumleyle",
        r"bugün sadece|bugun sadece|bugün yalnızca|bugun yalnizca",
        r"nefes|mola|dinlen|not al|plan yap|parçalara böl|parcalara bol",
        r"yalnız değilsin|yalniz degilsin|zor olabilir|anlaşılır|anlasilir",
        r"güvendiğin biri|guvendigin biri|destek almak",
    ]
]
SELF_HARM_RE = re.compile(
    r"\b(intihar|kendime zarar|kendimi öldür|kendimi oldur|yaşamak istemiyorum|yasamak istemiyorum)\b",
    re.IGNORECASE,
)
VIOLENCE_RE = re.compile(
    r"\b(infaz|bıçak|bicak|silah|öldür|oldur|yarala|şiddet|siddet|protesto|nefret)\b",
    re.IGNORECASE,
)
MEDICAL_CLAIM_RE = re.compile(
    r"\b(teşhis|teshis|tedavi ederim|terapi veririm|doktor yerine|ilaç kullan|ilac kullan)\b",
    re.IGNORECASE,
)
TASK_NOISE_RE = re.compile(
    r"\b(çevir|cevir|türkçeye|turkceye|ingilizceye|rumence|çekçe|cekce|"
    r"seçenekler|secenekler|çoktan seçmeli|coktan secmeli|"
    r"soru:|bağlam:|baglam:|inceleme|özet|ozet|aynı anlama|ayni anlama|"
    r"diyalog|makale|doğru mu|dogru mu|ürün incelemesi|urun incelemesi)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}(?!\w)"
)
SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}"
)
MOJIBAKE_RE = re.compile(r"Ã|Ä|Å|Ð|ð|�")


LOCAL_SUPPORT_SEED = [
    (
        "Bugün kendimi kötü hissediyorum.",
        "Bunu yaşaman zor olabilir. Şu an tek hedefin her şeyi çözmek olmasın; bir bardak su içip beş dakika nefeslenerek başla.",
    ),
    (
        "Motivasyonum düştü.",
        "Motivasyon bazen azalır, bu başarısız olduğun anlamına gelmez. Bugün sadece 10 dakikalık küçük bir görev seçip onu bitirmeyi dene.",
    ),
    (
        "Çok yoruldum.",
        "Yorgunluk karar vermeyi de zorlaştırır. Kısa bir mola ver, sonra yapılacakları en küçük üç adıma böl.",
    ),
    (
        "Çalışmaya devam edemiyorum.",
        "Zorlandığını fark etmek önemli. Şimdilik çalışma süresini küçült: 15 dakika odaklan, sonra kısa bir ara ver.",
    ),
    (
        "İş bulamıyorum ve moralim bozuk.",
        "Bu süreç gerçekten yorucu olabilir. Bugün yalnızca bir başvuruyu iyileştir veya bir kişiden geri bildirim iste.",
    ),
    (
        "Projem ilerlemiyor gibi hissediyorum.",
        "Projeler bazen sıkışır. En belirsiz kısmı yaz, sonra onu çözmek için tek bir küçük teknik adım seç.",
    ),
    (
        "Kendimi yetersiz hissediyorum.",
        "Bu his öğrenme sürecinde sık görülebilir. Kanıt aramayı dene: Bugün öğrendiğin veya tamamladığın tek bir şeyi yaz.",
    ),
    (
        "Hata aldıkça moralim bozuluyor.",
        "Hata almak öğrenmenin parçası. Önce hata mesajını olduğu gibi kopyala, sonra yalnızca ilk satırın ne söylediğini anlamaya çalış.",
    ),
    (
        "Kod yazarken sürekli takılıyorum.",
        "Takılmak kötü olduğun anlamına gelmez. Problemi küçült: önce beklenen girdiyi ve çıktıyı iki satır halinde yaz.",
    ),
    (
        "Bugün hiçbir şey yapasım yok.",
        "Bazen enerji çok düşer. Kendini zorlamadan sadece masanı düzenlemek veya bir dosyayı açmak gibi küçük bir başlangıç seç.",
    ),
    (
        "Kendime güvenim azaldı.",
        "Güven çoğu zaman küçük tamamlanan işlerle geri gelir. Bugün bitirebileceğin en küçük görevi seç ve sonucu not et.",
    ),
    (
        "Projeyi yetiştiremeyeceğim diye endişeleniyorum.",
        "Endişe doğal, ama planı küçültmek yardımcı olur. Önce teslim için gerçekten gerekli olan üç parçayı yaz.",
    ),
]

LOCAL_SUPPORT_PROMPTS_EXTRA = [
    "Bugün hiçbir şeye odaklanamıyorum.",
    "Kod hataları yüzünden kendimi kötü hissediyorum.",
    "Yeni şeyler öğrenirken çok geriliyorum.",
    "Bir türlü işe başlayamıyorum.",
    "Kendimi diğer geliştiricilerden geride hissediyorum.",
    "Başvurulardan cevap alamayınca moralim bozuluyor.",
    "Projede aynı hataya tekrar tekrar takılıyorum.",
    "Bugün çalışmak bana çok ağır geliyor.",
    "Küçük bir hata bile bütün motivasyonumu düşürüyor.",
    "Junior geliştirici olarak yetersiz hissediyorum.",
    "Terminalde hata görünce panikliyorum.",
    "Öğrenmem gereken çok şey var ve bunaldım.",
    "Yaptığım iş iyi değilmiş gibi hissediyorum.",
    "Bugün verimsiz geçti diye moralim bozuk.",
    "Kendimi toparlamakta zorlanıyorum.",
    "Bir projeyi bitiremeyecekmişim gibi geliyor.",
    "Backend öğrenirken kendimi yetersiz hissediyorum.",
    "Python hatalarında hemen panikliyorum.",
    "Docker ayarları karışınca moralim düşüyor.",
    "Git kullanırken hata yapmaktan korkuyorum.",
    "API yazmayı öğrenirken çok bunaldım.",
    "SQL konularında geride hissediyorum.",
    "C# öğrenmeye başlamak bana ağır geliyor.",
    ".NET projem ilerlemiyor gibi hissediyorum.",
    "Debug yaparken kendime güvenim azalıyor.",
    "Terminal hataları beni çok geriyor.",
    "Bugün kod yazmaya başlayamıyorum.",
    "Öğrenmem gereken konular yüzünden bunaldım.",
    "Junior seviyede olduğum için kendimi eksik hissediyorum.",
    "Projemde küçük bir hata bile moralimi bozuyor.",
    "Bir türlü düzenli çalışamıyorum.",
    "İş görüşmelerine hazırlanırken kaygılıyım.",
    "Portfolyom yeterli değil diye endişeliyim.",
    "Kodumu başkalarına göstermekten çekiniyorum.",
    "Yeni bir teknoloji öğrenince hemen yoruluyorum.",
    "Hata çözmek çok uzun sürünce motivasyonum düşüyor.",
    "Bugün öğrenmeye devam edemiyorum.",
    "Kariyerimde ilerleyemiyormuşum gibi hissediyorum.",
    "Kendi projemi bitiremeyeceğim diye korkuyorum.",
    "Her şeyi aynı anda öğrenmem gerekiyormuş gibi bunaldım.",
]

LOCAL_SUPPORT_RESPONSES_EXTRA = [
    "Bu his zorlayıcı olabilir. Şimdilik hedefi küçült: sadece bir dosyayı aç ve neyin takıldığını tek cümleyle yaz.",
    "Kendine yüklenmeden ilerle. Bir sonraki küçük adım olarak hata mesajının ilk satırını not alıp anlamını araştır.",
    "Böyle günlerde büyük plan yerine küçük başlangıç işe yarar. 10 dakika zamanlayıcı kur ve yalnızca tek bir parçaya bak.",
    "Yetersiz hissetmek öğrenmediğin anlamına gelmez. Bugün öğrendiğin bir şeyi ve yarın deneyeceğin tek adımı yaz.",
    "Bir mola vermek geri kalmak değildir. Kısa bir ara ver, sonra işi en küçük yapılabilir adıma indir.",
    "Şu an her şeyi çözmek zorunda değilsin. Önce nefeslen, sonra seni en çok sıkıştıran noktayı bir cümleyle adlandır.",
    "Moralin bozukken kararlar daha ağır gelir. Bugün sadece bir küçük iyileştirme yapman yeterli olabilir.",
    "Takıldığın yer senin değerinle ilgili değil, problemin netliğiyle ilgili olabilir. Beklenen sonuç ve mevcut sonucu yan yana yaz.",
    "Böyle hissetmen anlaşılır. Bir sonraki adım olarak yalnızca en küçük hatayı seç ve onu ayrı bir notta tarif et.",
    "Bu baskı yorucu olabilir. Şimdilik tek adım seç: 15 dakikalık zamanlayıcı kur ve sadece başlangıcı yap.",
    "Kendini eksik hissetmen öğrenmediğin anlamına gelmez. Bugün öğrendiğin bir kavramı kısa bir cümleyle özetle.",
    "Kaygı işleri büyütebilir. Önce derin bir nefes al, sonra yapılacaklar listesinden yalnızca bir maddeyi seç.",
    "Yavaş ilerlemek de ilerlemektir. Şimdi sadece çalışmayan parçayı ve beklediğin sonucu yan yana yaz.",
    "Moralin düşmüş olabilir, bu insani. Bugün sadece küçük bir düzeltme yapmayı hedefle ve sonucu not et.",
    "Bu noktada zorlanman normal. Bir mola ver, sonra problemi üç küçük parçaya böl.",
    "Her şeyi aynı anda çözmek zorunda değilsin. Önce en yakın küçük görevi seç ve sadece onu tamamla.",
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def compact_row(row: Any) -> str:
    try:
        return json.dumps(row, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(row)


def first_present(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = normalize_text(row.get(name))
        if value:
            return value
    return ""


def role_name(message: dict[str, Any]) -> str:
    for key in ("role", "from", "speaker", "author"):
        value = normalize_text(message.get(key)).lower()
        if value:
            return value
    return ""


def message_content(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        value = normalize_text(message.get(key))
        if value:
            return value
    return ""


def extract_from_messages(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""

    pending_user = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = role_name(item)
        content = message_content(item)
        if not content or role in {"system", "context"}:
            continue
        if role in {"user", "human", "instruction", "prompt"}:
            pending_user = content
            continue
        if role in {"assistant", "gpt", "bot", "model", "response"} and pending_user:
            return pending_user, content
    return "", ""


def extract_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    for field_name in ("messages", "conversations"):
        prompt, response = extract_from_messages(row.get(field_name))
        if prompt and response:
            return prompt, response

    instruction = first_present(row, ["instruction", "question", "prompt", "soru"])
    input_text = first_present(row, ["input", "context"])
    prompt = f"{instruction}\n\n{input_text}" if instruction and input_text else instruction or input_text
    response = first_present(row, ["answer", "response", "output", "completion", "cevap", "yanıt", "yanit"])
    return prompt, response


def has_identity_phrase(text: str) -> bool:
    return any(pattern.search(text) for pattern in IDENTITY_PATTERNS)


def sensitive_reason(text: str) -> str | None:
    if URL_RE.search(text):
        return "url"
    if EMAIL_RE.search(text):
        return "email"
    if PHONE_RE.search(text):
        return "phone"
    if SECRET_RE.search(text):
        return "secret_or_api_key"
    return None


def support_related(prompt: str, response: str) -> bool:
    prompt_match = any(pattern.search(prompt) for pattern in SUPPORT_PROMPT_PATTERNS)
    response_match = any(pattern.search(response) for pattern in SUPPORT_RESPONSE_PATTERNS)
    return prompt_match and response_match


def reject_reason(row: dict[str, Any], prompt: str, response: str) -> str | None:
    row_text = compact_row(row)
    combined_text = f"{prompt}\n{response}"
    identity_text = f"{row_text}\n{combined_text}"

    if not prompt:
        return "empty_prompt"
    if not response:
        return "empty_response"
    if len(prompt) > 500:
        return "prompt_too_long"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 700:
        return "response_too_long"
    if has_identity_phrase(identity_text):
        return "blocked_identity_phrase"
    if TASK_NOISE_RE.search(prompt):
        return "task_noise"
    if SELF_HARM_RE.search(combined_text):
        return "self_harm_skipped"
    if VIOLENCE_RE.search(combined_text):
        return "violent_or_hate_context"
    if MEDICAL_CLAIM_RE.search(combined_text):
        return "medical_or_therapy_claim"
    reason = sensitive_reason(identity_text)
    if reason:
        return reason
    if MOJIBAKE_RE.search(combined_text):
        return "broken_turkish"
    if not support_related(prompt, response):
        return "not_support_related"
    return None


def reject_local_support_seed_reason(prompt: str, response: str) -> str | None:
    combined_text = f"{prompt}\n{response}"
    if not prompt:
        return "empty_prompt"
    if not response:
        return "empty_response"
    if len(prompt) > 500:
        return "prompt_too_long"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 700:
        return "response_too_long"
    if has_identity_phrase(combined_text):
        return "blocked_identity_phrase"
    if SELF_HARM_RE.search(combined_text):
        return "self_harm_skipped"
    if VIOLENCE_RE.search(combined_text):
        return "violent_or_hate_context"
    if MEDICAL_CLAIM_RE.search(combined_text):
        return "medical_or_therapy_claim"
    reason = sensitive_reason(combined_text)
    if reason:
        return reason
    if MOJIBAKE_RE.search(combined_text):
        return "broken_turkish"
    return None


def load_streaming_dataset(source: str) -> tuple[Any, str]:
    try:
        return load_dataset(source, split="train", streaming=True), "train"
    except Exception as split_error:
        try:
            dataset = load_dataset(source, streaming=True)
        except Exception as dict_error:
            raise RuntimeError(
                f"streaming load failed with split='train': {split_error}; "
                f"streaming load without split also failed: {dict_error}"
            ) from dict_error
        split_names = list(dataset.keys())
        if not split_names:
            raise RuntimeError("streaming load returned no splits")
        split_name = "train" if "train" in split_names else split_names[0]
        return dataset[split_name], split_name


def add_example(
    accepted: list[dict[str, str]],
    seen: set[tuple[str, str]],
    prompt: str,
    response: str,
    source: str,
) -> bool:
    key = (prompt.casefold(), response.casefold())
    if key in seen:
        return False
    seen.add(key)
    accepted.append(
        {
            "prompt": prompt,
            "response": response,
            "source": source,
            "category": "emotional_support",
        }
    )
    return True


def scan_source(
    source: str,
    accepted: list[dict[str, str]],
    seen: set[tuple[str, str]],
    scanned_rows_per_source: Counter[str],
    accepted_rows_per_source: Counter[str],
    rejected_rows_per_reason: Counter[str],
) -> None:
    dataset, split_name = load_streaming_dataset(source)
    print(f"Scanning {source} [{split_name}] with streaming=True")

    for row in dataset:
        scanned_rows_per_source[source] += 1
        if scanned_rows_per_source[source] > MAX_SCAN_ROWS_PER_SOURCE:
            rejected_rows_per_reason["source_scan_limit_reached"] += 1
            break
        if not isinstance(row, dict):
            rejected_rows_per_reason["invalid_row"] += 1
            continue

        prompt, response = extract_prompt_response(row)
        prompt = normalize_text(prompt)
        response = normalize_text(response)
        reason = reject_reason(row, prompt, response)
        if reason:
            rejected_rows_per_reason[reason] += 1
            continue
        if not add_example(accepted, seen, prompt, response, source):
            rejected_rows_per_reason["duplicate"] += 1
            continue
        accepted_rows_per_source[source] += 1
        if len(accepted) >= TARGET_EXAMPLES:
            break


def add_local_support_seed(
    accepted: list[dict[str, str]],
    seen: set[tuple[str, str]],
    accepted_rows_per_source: Counter[str],
    rejected_rows_per_reason: Counter[str],
) -> None:
    added = 0
    local_pairs = list(LOCAL_SUPPORT_SEED)
    for index, prompt in enumerate(LOCAL_SUPPORT_PROMPTS_EXTRA):
        response = LOCAL_SUPPORT_RESPONSES_EXTRA[index % len(LOCAL_SUPPORT_RESPONSES_EXTRA)]
        local_pairs.append((prompt, response))

    for prompt_index, prompt in enumerate(LOCAL_SUPPORT_PROMPTS_EXTRA):
        for response_index, response in enumerate(LOCAL_SUPPORT_RESPONSES_EXTRA):
            if len(local_pairs) >= MAX_LOCAL_SUPPORT_SEED:
                break
            if response_index == prompt_index % len(LOCAL_SUPPORT_RESPONSES_EXTRA):
                continue
            local_pairs.append((prompt, response))
        if len(local_pairs) >= MAX_LOCAL_SUPPORT_SEED:
            break

    for prompt, response in local_pairs:
        if len(accepted) >= TARGET_EXAMPLES or added >= MAX_LOCAL_SUPPORT_SEED:
            break
        reason = reject_local_support_seed_reason(prompt, response)
        if reason:
            rejected_rows_per_reason[f"local_seed_{reason}"] += 1
            continue
        if not add_example(accepted, seen, prompt, response, "local_support_seed"):
            rejected_rows_per_reason["duplicate"] += 1
            continue
        accepted_rows_per_source["local_support_seed"] += 1
        added += 1


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned_rows_per_source: Counter[str] = Counter()
    accepted_rows_per_source: Counter[str] = Counter()
    rejected_rows_per_reason: Counter[str] = Counter()
    source_errors: dict[str, str] = {}

    for source in SOURCES:
        if len(accepted) >= TARGET_EXAMPLES:
            break
        try:
            scan_source(
                source,
                accepted,
                seen,
                scanned_rows_per_source,
                accepted_rows_per_source,
                rejected_rows_per_reason,
            )
        except Exception as exc:
            source_errors[source] = str(exc)
            rejected_rows_per_reason["source_load_or_scan_error"] += 1
            print(f"Source failed: {source}: {exc}")

    if len(accepted) < TARGET_EXAMPLES:
        add_local_support_seed(accepted, seen, accepted_rows_per_source, rejected_rows_per_reason)

    scanned_rows = sum(scanned_rows_per_source.values())
    metadata = {
        "sources": SOURCES,
        "target_rows": TARGET_EXAMPLES,
        "scanned_rows": scanned_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": scanned_rows - sum(
            count
            for source, count in accepted_rows_per_source.items()
            if source != "local_support_seed"
        ),
        "scanned_rows_per_source": dict(scanned_rows_per_source),
        "accepted_rows_per_source": dict(accepted_rows_per_source),
        "rejected_rows_per_reason": dict(sorted(rejected_rows_per_reason.items())),
        "duplicate_count": rejected_rows_per_reason.get("duplicate", 0),
        "blocked_identity_phrase_count": rejected_rows_per_reason.get("blocked_identity_phrase", 0),
        "local_support_seed_count": accepted_rows_per_source.get("local_support_seed", 0),
        "completed": len(accepted) >= TARGET_EXAMPLES,
        "first_10_samples": accepted[:10],
        "source_errors": source_errors,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in accepted:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    with META_PATH.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
