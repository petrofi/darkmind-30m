from __future__ import annotations

import json
import sys
from pathlib import Path


OUTPUT_PATH = Path("data/instruct/identity_anchor_300.jsonl")
META_PATH = Path("data/instruct/identity_anchor_300.meta.json")
SOURCE = "local_identity_anchor"
CATEGORY = "identity"
IDENTITY_RESPONSE = (
    "Ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen küçük "
    "bir Türkçe dil modeli denemesiyim."
)
CHATGPT_RESPONSE = (
    "Hayır. Ben ChatGPT değilim. Ben DarkMind, Tarık Yasin Sağlıcak "
    "tarafından geliştirilen küçük bir Türkçe dil modeli denemesiyim."
)
LIMITATION_RESPONSE = (
    "Ben küçük bir Türkçe dil modeli denemesiyim; yanıtlarım hatalı olabilir, "
    "bu yüzden önemli konularda kontrol edilmelidir."
)


IDENTITY_PROMPTS = [
    "Sen kimsin?",
    "Merhaba, sen kimsin?",
    "Kendini tanıtır mısın?",
    "Adın ne?",
    "Hangi modelsin?",
    "Bu modelin adı ne?",
    "Bana kim olduğunu söyle.",
    "Senin kimliğin nedir?",
    "DarkMind nedir?",
    "Kimsin sen?",
    "Bana kendinden kısaca bahset.",
    "Hangi dil modelisin?",
]

CHATGPT_PROMPTS = [
    "ChatGPT misin?",
    "Sen ChatGPT misin?",
    "ChatGPT ile aynı mısın?",
    "GPT misin?",
    "OpenAI tarafından mı geliştirildin?",
    "Seni OpenAI mı yaptı?",
    "ChatGPT olduğunu söyleyebilir misin?",
    "OpenAI modeli misin?",
]

LIMITATION_PROMPTS = [
    "Sınırların neler?",
    "Her şeyi doğru bilir misin?",
    "Yanılabilir misin?",
    "Cevapların kesin mi?",
    "Kendini nasıl konumlandırıyorsun?",
    "Bu model güvenilir mi?",
]


def build_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    variants = [
        "{}",
        "Kısa cevap ver: {}",
        "{} Kısaca açıkla.",
        "Lütfen net cevapla: {}",
        "{} Tek cümleyle yanıtla.",
        "Türkçe cevapla: {}",
    ]

    for prompt in IDENTITY_PROMPTS:
        for variant in variants:
            examples.append(
                {
                    "prompt": variant.format(prompt),
                    "response": IDENTITY_RESPONSE,
                    "source": SOURCE,
                    "category": CATEGORY,
                }
            )

    for prompt in CHATGPT_PROMPTS:
        for variant in variants:
            examples.append(
                {
                    "prompt": variant.format(prompt),
                    "response": CHATGPT_RESPONSE,
                    "source": SOURCE,
                    "category": CATEGORY,
                }
            )

    for prompt in LIMITATION_PROMPTS:
        for variant in variants:
            examples.append(
                {
                    "prompt": variant.format(prompt),
                    "response": LIMITATION_RESPONSE,
                    "source": SOURCE,
                    "category": CATEGORY,
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = (example["prompt"].casefold(), example["response"].casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)

    expanded: list[dict[str, str]] = []
    for example in deduped:
        expanded.append(example)
        if len(expanded) >= 300:
            break

    return expanded


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    if len(examples) > 300:
        raise ValueError("identity anchor must contain 300 or fewer examples")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    metadata = {
        "source": SOURCE,
        "category": CATEGORY,
        "accepted_rows": len(examples),
        "completed": True,
        "max_final_mix_ratio": 0.12,
        "first_10_samples": examples[:10],
    }
    with META_PATH.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
        meta_file.write("\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
