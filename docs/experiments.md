# DarkMind-30M Base v0.1 - 10K Corpus

- Dataset: pretrain_10k.jsonl
- Documents: 10,000
- Tokens: 32,087,591
- Params: 28,127,232
- Architecture: 8 layers / 8 heads / 512 embedding / block size 256
- Checkpoint: models/darkmind-30m-10k-step15000.pt
- Final train loss: 2.8288
- Final val loss: 2.7345

Notes:

- Turkish structure improved.
- Model is still a base LM, not an instruct/chat model.
- Repetition loops still appear.
- Next step is instruct fine-tuning.

# DarkMind-30M Instruct v0.1

- Dataset size: TBD
- Base checkpoint: models/darkmind-30m-10k-step15000.pt
- Training loss: TBD
- Validation loss: TBD
- Test prompts:
  - Merhaba, sen kimsin?
  - Python nedir?
  - Bugün kendimi kötü hissediyorum.
- Before/after examples: TBD

Notes:

- This run is intended to teach basic Turkish question-answer behavior.
- It starts from the Base v0.1 checkpoint, not from random weights.
- It is not expected to make the model ChatGPT-level.

# DarkMind-30M Instruct v0.2

- Dataset size: 500 examples
- Dataset path: data/instruct/darkmind_instruct_v0_2.jsonl
- Eval prompt set: data/eval/darkmind_eval_prompts.jsonl
- Base checkpoint: models/darkmind-30m-10k-step15000.pt
- Smoke checkpoint: models/darkmind-30m-instruct-v0.2-smoke.pt
- Params: 28,127,232
- Train examples: 450
- Validation examples: 50
- Training loss: 1.4953
- Validation loss: 1.9724

Training command:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_v0_2.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 3 --batch_size 4 --block_size 256 --max_steps 300 --lr 0.00005 --val_ratio 0.1 --eval_interval 50 --eval_batches 10 --save_path models/darkmind-30m-instruct-v0.2-smoke.pt
```

Sample outputs:

```text
Prompt: Merhaba, sen kimsin?
Answer: Ben DarkMind, Tar?k Yasin Sa?l?cak taraf?ndan geli?tirilen k???k bir T?rk?e dil modeli denemesiyim.
```

```text
Prompt: ChatGPT misin?
Answer: Hata, yaz?l?m geli?tirmede daha d?zenli ve anla??lc? olur. K?saca, bilgi, ara?t?r ?al??ma sa?lamaya yard?mc? olabilir.
```

```text
Prompt: Python nedir?
Answer: Python, Python, bilgisayarlar?n? ve ??renmesi i?in ?al??t?r.
```

Known limitations:

- Identity response improved for direct identity prompts.
- Related identity prompts such as `ChatGPT misin?` can still fail.
- Turkish special characters may appear corrupted in generated text.
- The model can still drift into base-pretraining continuations after the answer.
- This is still a small local smoke run, not a production chat model.

Eval report:

- reports/instruct_eval_outputs.md

# DarkMind-30M Instruct v0.3 Dataset Pipeline

- Smoke dataset: data/instruct/darkmind_instruct_v0_3_smoke.jsonl
- Smoke examples: 1,000
- Full target output: data/instruct/darkmind_instruct_v0_3.jsonl
- Eval prompt set: data/eval/darkmind_eval_prompts.jsonl
- Eval prompt count: 50
- Base checkpoint: models/darkmind-30m-10k-step15000.pt
- Smoke checkpoint: models/darkmind-30m-instruct-v0.3-smoke.pt
- Params: 28,127,232
- Train examples: 900
- Validation examples: 100
- Training loss: 2.5957
- Validation loss: 2.4596

Dataset sources attempted:

- TFLai/Turkish-Alpaca
- malhajar/OpenOrca-tr
- malhajar/alpaca-gpt4-tr
- CohereLabs/aya_dataset
- merve/turkish_instructions
- atasoglu/instruction-turkish
- afkfatih/turkish-distilled-5K
- alibayram/diyalog-dataset

Skipped sources in smoke run:

- All external sources were skipped because `--max_examples 1000` smoke runs avoid slow external dataset downloads.

Smoke source distribution:

- local_fallback: 574
- local_identity: 320
- local_seed: 106

Smoke category distribution:

- identity: 323
- programming: 258
- ai: 135
- general: 70
- emotional_support: 66
- career: 64
- explanation: 64
- other: 20

Training command:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_v0_3_smoke.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 2 --batch_size 4 --block_size 256 --max_steps 200 --lr 0.00003 --val_ratio 0.1 --eval_interval 50 --eval_batches 10 --save_path models/darkmind-30m-instruct-v0.3-smoke.pt
```

Recommended full v0.3 command:

```powershell
.\.venv\Scripts\python.exe scripts\train_instruct_jsonl.py --data data/instruct/darkmind_instruct_v0_3.jsonl --base_checkpoint models/darkmind-30m-10k-step15000.pt --epochs 3 --batch_size 4 --block_size 256 --max_steps 2000 --lr 0.00003 --val_ratio 0.1 --eval_interval 100 --eval_batches 20 --save_path models/darkmind-30m-instruct-v0.3.pt
```

Sample outputs:

```text
Prompt: Merhaba, sen kimsin?
Answer: Sınırlı, eğitim denemesiyim. Ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen küçük bir Türkçe dil modeli denemesiyim ve denemesiyim, Türkiye'de geliştirilen yeni sürümüdür.
```

```text
Prompt: ChatGPT misin?
Answer: Hata, Python, Python. Ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen küçük bir Türkçe dil modeli denemesiyim.
```

```text
Prompt: Python nedir?
Answer: Hata, model, yazılım ve yazılımlara sahip olan yazılımların geliştirilmesi için geliştirilen küçük bir yazılımların geliştirmesini sağlayan programlama dilidir.
```

Known limitations:

- v0.3 smoke improves identity direction and no longer says `Ben ChatGPT` in the tested prompts.
- Answers still contain irrelevant fragments such as `Hata, Python`.
- Support prompts can collapse into identity answers.
- The model still drifts into base-pretraining continuations after the answer.
- Full HF-backed data preparation was not run in the smoke test.

Eval report:

- reports/instruct_eval_outputs.md
