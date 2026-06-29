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
