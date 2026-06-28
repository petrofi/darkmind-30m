from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "raw_collected"
    / "qa_pairs"
    / "generated_qa_variants_v01.txt"
)


TOPICS = [
    {
        "answer": "Ben DarkMind. Türkçe odaklı küçük bir dil modeli geliştirme projesiyim.",
        "questions": [
            "sen kimsin",
            "Sen kimsin?",
            "kendini tanıtır mısın",
            "Kendini tanıtır mısın?",
            "DarkMind kimdir?",
            "sen ne modelisin",
            "Sen nasıl bir modelsin?",
        ],
    },
    {
        "answer": "DarkMind, tokenizer, Transformer mimarisi ve eğitim pipeline'ı sıfırdan kurulan küçük bir Türkçe dil modeli projesidir.",
        "questions": [
            "darkmind nedir",
            "DarkMind nedir?",
            "DarkMind ne işe yarar?",
            "darkmind projesi ne",
            "Bu proje nedir?",
            "DarkMind'i açıklar mısın?",
            "DarkMind nasıl bir proje?",
        ],
    },
    {
        "answer": "Hayır. DarkMind bitmiş veya güçlü bir genel sohbet modeli değildir; geliştirme ve öğrenme amaçlı küçük bir projedir.",
        "questions": [
            "DarkMind hazır bir model mi?",
            "darkmind hazır mı",
            "Bu model tamamlandı mı?",
            "ChatGPT gibi güçlü müsün?",
            "DarkMind üretime hazır mı?",
            "Sen bitmiş bir yapay zeka mısın?",
            "Bu model gerçek üründe kullanılabilir mi?",
        ],
    },
    {
        "answer": "Tokenizer, metni modelin anlayabileceği token adı verilen küçük parçalara dönüştüren bileşendir.",
        "questions": [
            "tokenizer nedir",
            "Tokenizer nedir?",
            "Tokenizer ne işe yarar?",
            "Token ne demek?",
            "Metin neden tokenlara ayrılır?",
            "Tokenizer neden önemli?",
            "DarkMind tokenizer kullanıyor mu?",
        ],
    },
    {
        "answer": "Transformer, attention mekanizmasıyla tokenlar arasındaki ilişkileri öğrenen modern bir sinir ağı mimarisidir.",
        "questions": [
            "transformer nedir",
            "Transformer nedir?",
            "Transformer ne işe yarar?",
            "Dil modellerinde Transformer neden kullanılır?",
            "DarkMind Transformer mı kullanıyor?",
            "Transformer mimarisi ne demek?",
            "Transformer'ı kısa anlatır mısın?",
        ],
    },
    {
        "answer": "Causal mask, modelin gelecek tokenları görmesini engeller ve sadece geçmiş bağlama göre tahmin yapmasını sağlar.",
        "questions": [
            "causal mask nedir",
            "Causal mask nedir?",
            "Causal mask neden kullanılır?",
            "Model neden geleceği görmemeli?",
            "Causal mask ne işe yarar?",
            "GPT'de maskeleme ne demek?",
            "Gelecek tokenları kapatmak neden önemli?",
        ],
    },
    {
        "answer": "Attention, modelin üretim yaparken hangi tokenlara daha fazla önem vereceğini öğrenmesini sağlayan mekanizmadır.",
        "questions": [
            "attention nedir",
            "Attention nedir?",
            "Attention ne işe yarar?",
            "Self attention ne demek?",
            "Model hangi kelimeye bakacağını nasıl öğrenir?",
            "Attention mekanizmasını açıkla",
            "Attention neden önemli?",
        ],
    },
    {
        "answer": "Overfitting, modelin eğitim verisini ezberleyip yeni örneklerde zayıf performans göstermesidir.",
        "questions": [
            "overfitting nedir",
            "Overfitting nedir?",
            "Aşırı öğrenme ne demek?",
            "Model neden ezberler?",
            "Overfitting nasıl anlaşılır?",
            "Train loss düşüp val loss yükselirse ne olur?",
            "DarkMind overfit olabilir mi?",
        ],
    },
    {
        "answer": "Loss, modelin tahmini ile doğru cevap arasındaki hatayı ölçen değerdir; eğitimde düşmesi beklenir.",
        "questions": [
            "loss nedir",
            "Loss nedir?",
            "Kayıp değeri ne demek?",
            "Training loss nedir?",
            "Validation loss nedir?",
            "Loss neden düşmeli?",
            "Loss değeri modeli nasıl etkiler?",
        ],
    },
    {
        "answer": "Checkpoint, model ağırlıklarının ve eğitim bilgilerinin kaydedildiği dosyadır; daha sonra üretim veya devam eğitimi için kullanılır.",
        "questions": [
            "checkpoint nedir",
            "Checkpoint nedir?",
            "Checkpoint ne işe yarar?",
            "Model neden checkpoint kaydeder?",
            "Checkpoint dosyası ne içerir?",
            "Eğitimden sonra modeli nasıl saklarız?",
            "Checkpoint ile tokenizer uyumlu olmalı mı?",
        ],
    },
    {
        "answer": "CUDA, NVIDIA GPU üzerinde PyTorch işlemlerini hızlandırmak için kullanılan altyapıdır.",
        "questions": [
            "cuda nedir",
            "CUDA nedir?",
            "CUDA ne işe yarar?",
            "CUDA neden önemli?",
            "torch cuda false ne demek?",
            "GPU eğitimi için CUDA gerekir mi?",
            "DarkMind CUDA kullanabilir mi?",
        ],
    },
    {
        "answer": "GPU, paralel hesaplamalarda CPU'ya göre daha hızlı olabilir; küçük denemeler CPU'da yapılabilir ama eğitim GPU'da daha verimlidir.",
        "questions": [
            "gpu ve cpu farkı nedir",
            "GPU ve CPU farkı nedir?",
            "Model eğitimi CPU'da olur mu?",
            "GPU neden daha hızlı?",
            "CPU ile eğitim yapılabilir mi?",
            "DarkMind için GPU gerekli mi?",
            "GPU yoksa ne olur?",
        ],
    },
    {
        "answer": "Data pipeline, ham veriyi toplama, temizleme, corpus oluşturma ve eğitim için hazır hale getirme sürecidir.",
        "questions": [
            "data pipeline nedir",
            "Data pipeline nedir?",
            "Veri pipeline'ı ne işe yarar?",
            "Ham veri nasıl eğitime hazırlanır?",
            "DarkMind veri pipeline'ı ne yapar?",
            "Clean text ve build corpus neden var?",
            "Dataset büyütme süreci nedir?",
        ],
    },
    {
        "answer": "Corpus, tokenizer ve model eğitimi için birleştirilmiş metin veri setidir.",
        "questions": [
            "corpus nedir",
            "Corpus nedir?",
            "corpus_v3 ne işe yarar",
            "Eğitim corpusu ne demek?",
            "Model hangi metinden öğrenir?",
            "Corpus neden temiz olmalı?",
            "Corpus nasıl oluşturulur?",
        ],
    },
    {
        "answer": "Temiz veri; tekrarları, bozuk satırları ve gereksiz gürültüyü azaltılmış, okunabilir eğitim metnidir.",
        "questions": [
            "temiz veri nedir",
            "Temiz veri nedir?",
            "Clean data neden önemli?",
            "Kirli veri modeli bozar mı?",
            "Veri temizlemek ne işe yarar?",
            "Boş satırlar neden temizlenir?",
            "Kaliteli veri neden gerekli?",
        ],
    },
    {
        "answer": "Chat demo, eğitilmiş checkpoint ile terminalde soru sorup modelden kısa cevap almak için kullanılan basit arayüzdür.",
        "questions": [
            "chat demo nedir",
            "Chat demo nedir?",
            "chat_demo.py ne yapar",
            "DarkMind ile nasıl konuşurum?",
            "Terminalden sohbet nasıl çalışır?",
            "Chat demo hangi promptu kullanır?",
            "Model cevabını nasıl test ederim?",
        ],
    },
    {
        "answer": "Bilmediğim konularda kesin konuşmamalıyım; sınırlı olduğumu söylemek daha doğru davranıştır.",
        "questions": [
            "bilmiyorsan ne yaparsın",
            "Bilmediğin soruda ne yaparsın?",
            "Her şeyi biliyor musun?",
            "Cevabından emin değilsen ne demelisin?",
            "Yanlış cevap verirsen ne olur?",
            "Sana her konuda güvenebilir miyim?",
            "Bilinmeyen sorularda nasıl davranmalısın?",
        ],
    },
]


def build_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []

    for topic in TOPICS:
        for question in topic["questions"]:
            examples.append((question, topic["answer"]))

    return examples


def render_example(question: str, answer: str) -> str:
    return f"Kullanıcı: {question}\nAsistan: {answer}"


def main() -> None:
    examples = build_examples()

    if len(examples) < 100:
        raise ValueError(f"Expected at least 100 examples, got {len(examples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n\n".join(
        render_example(question, answer)
        for question, answer in examples
    )
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print(f"Generated examples: {len(examples)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
