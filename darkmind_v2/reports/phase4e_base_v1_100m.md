# DarkMind v2 Phase 4E Base V1 100M-horizon attempt

Classification: **CONDITIONAL PASS**

Training stopped exactly at the 75M continuation gate. The final no-wrap stop was not authorized or reached, so this is not a completed Corpus V3 pass and not a 100M result.

## Gate decision

- 50M gate: PASS; validation/eval improvement from 25M `5.827%` / `5.791%`.
- 75M gate: CONDITIONAL; validation/eval improvement from 50M `2.165%` / `2.104%`.
- Total 25M-to-75M improvement: validation `7.866%`; eval `7.773%`.
- Rebound was 0%, no catastrophic probe regression occurred, integrity passed, and generation hard failures were zero.
- The 75M gate required at least 3% validation and eval improvement from 50M. Both landed in the 1%-3% conditional-stop band.

## Progression

| Step | Tokens | LR | Train | Validation | Eval | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 3,051 | 24,993,792 | 0.000090230387 | 5.597983 | 5.526496 | 5.473539 | 238.302 |
| 4,096 | 33,554,432 | 0.000082811055 | 5.457444 | 5.378776 | 5.328465 | 206.121 |
| 5,120 | 41,943,040 | 0.000074274139 | 5.102102 | 5.276650 | 5.226948 | 186.224 |
| 6,103 | 49,995,776 | 0.000065458628 | 5.130779 | 5.204454 | 5.156582 | 173.570 |
| 7,168 | 58,720,256 | 0.000055892363 | 5.043829 | 5.147137 | 5.101765 | 164.312 |
| 8,192 | 67,108,864 | 0.000047337704 | 4.770581 | 5.113922 | 5.070560 | 159.263 |
| 9,155 | 74,997,760 | 0.000040413947 | 5.214144 | 5.091754 | 5.048094 | 155.725 |

## Exact stop

- Final completed optimizer step: `9,155`.
- Consumed tokens: `74,997,760`; sequence index: `146,480`.
- Unconsumed full-batch capacity: `23,076,864` tokens / `45,072` sequences.
- Immutable tail outside a full 16-sequence optimizer batch: `14` sequences.
- No step 9156, final segment, second epoch, SFT, Qwen generation, or upload occurred.

## Checkpoint

- Full-resume checkpoint: `C:\DarkMindRuntime\phase4e\runs\base_v1_stage3_first_corpus_pass_v2\checkpoints\step_009155_tokens_074997760`.
- Model SHA-256: `d76b253fe0feced86e0a246c949649e95540f7f4bbeef3ba2491bcc7de3174f0`.
- Resume-state SHA-256: `970f8c848487a383c63fb02e7049e28d22d0a282200c97f6c1f0376dce57efd0`.

The model remains a base model: not instruction-tuned, not a chatbot, not production-ready, and not approved for public upload.
