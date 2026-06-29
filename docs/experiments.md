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
