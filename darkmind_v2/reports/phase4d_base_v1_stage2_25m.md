# DarkMind v2 Phase 4D Base V1 Stage-2 25M

Stage-2 classification: **STRONG PASS**
100M recommendation: Prepare a separate approval request for continuation toward the 100M gate; do not start it automatically.

This classification concerns stable learning only. The model is not instruction-tuned, not a chat model, not production-ready, and not publicly released.

## Identity and scope

- Exact resume checkpoint: `C:\DarkMindRuntime\phase4c\runs\base_v1_stage1_5m_v2_confirmation\checkpoints\step_000610_tokens_004997120`.
- Frozen V2 config SHA-256: `9358b8b33a87729ef2f19cfad76acbe370ed44ada911fa21512e4085eccf52ea`.
- Stage-2 authorization SHA-256: `b7943e9480769f85e7dde1ffafe229f6d45c02574c87ae1a481edbfed59a6969`.
- Step/token range: `610` / `4,997,120` to `3051` / `24,993,792`.
- Additional work: `2,441` steps, `19,996,672` tokens, `39,056` sequences.
- Base V1 architecture, frozen tokenizer, Corpus V3, deterministic order, optimizer grouping, and V2 scheduler were not changed.

## Learning progression

| Step | Tokens | Applied LR | Train loss | Validation | Eval | Eval perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 610 | 4,997,120 | 0.000099693965 | 6.371749 | 6.356269 | 6.306323 | 548.026 |
| 1,024 | 8,388,608 | 0.000098998786 | 5.957427 | 6.102912 | 6.051275 | 424.654 |
| 1,536 | 12,582,912 | 0.000097598167 | 5.666016 | 5.896094 | 5.843950 | 345.140 |
| 1,831 | 14,999,552 | 0.000096528279 | 5.980260 | 5.802467 | 5.750193 | 314.251 |
| 2,048 | 16,777,216 | 0.000095623009 | 5.626973 | 5.739001 | 5.687355 | 295.112 |
| 2,560 | 20,971,520 | 0.000093108121 | 5.718340 | 5.621224 | 5.566874 | 261.615 |
| 3,051 | 24,993,792 | 0.000090230387 | 5.597983 | 5.526496 | 5.473539 | 238.302 |

Validation improvement from step 610: `13.054%`; eval improvement: `13.206%`.
Validation/eval rebound: `0.000%` / `0.000%`.
Best validation checkpoint: step `3051` at `C:\DarkMindRuntime\phase4d\runs\base_v1_stage2_25m_v2_retry1\checkpoints\step_003051_tokens_024993792`.

## Optimization and activations

Gradient norm p50/p95/max: `1.2812` / `1.5703` / `2.5625`.
Clipped-step fraction: `97.952%`; maximum sentinel update-to-weight ratio: `0.00116190`.
The high clipping fraction is disclosed as an optimization characteristic; losses, updates, logits, and residual diagnostics remained finite and the validation/eval curve did not rebound.

| Step | Logit std | Prediction entropy | Embedding norm mean | Final/early residual RMS |
|---:|---:|---:|---:|---:|
| 610 | 1.937820 | 6.128659 | 0.597538 | 143.241076 |
| 1,024 | 1.898944 | 6.223854 | 0.613982 | 129.519712 |
| 1,536 | 1.866494 | 6.077896 | 0.626573 | 120.354806 |
| 1,831 | 1.901790 | 6.122252 | 0.631883 | 115.360312 |
| 2,048 | 1.895763 | 6.005208 | 0.635046 | 110.222500 |
| 2,560 | 1.840910 | 5.999512 | 0.641414 | 103.850207 |
| 3,051 | 1.851104 | 5.793063 | 0.646361 | 95.388398 |

## Fixed probes

| Probe | Step 610 | Step 3051 | Improvement | Catastrophic regression |
|---|---:|---:|---:|---|
| english_prose | 6.708841 | 5.809107 | 13.411% | NO |
| english_technical | 6.916442 | 6.185848 | 10.563% | NO |
| eval | 6.692104 | 5.856340 | 12.489% | NO |
| source_python_docs_en_3_14_6 | 4.199107 | 3.511546 | 16.374% | NO |
| source_python_docs_en_3_14_6_text | 6.916442 | 6.185848 | 10.563% | NO |
| source_python_docs_tr_3_14_6 | 5.922074 | 5.041651 | 14.867% | NO |
| source_python_docs_tr_3_14_6_text | 7.060092 | 6.218752 | 11.917% | NO |
| source_tatoeba_sentences_detailed_20260704 | 6.789620 | 5.936430 | 12.566% | NO |
| source_wikimedia_enwiki_20260701 | 6.202080 | 5.173727 | 16.581% | NO |
| source_wikimedia_enwikibooks_20260701 | 5.964764 | 5.330787 | 10.629% | NO |
| source_wikimedia_enwikiversity_20260201 | 5.881983 | 5.232677 | 11.039% | NO |
| source_wikimedia_enwikiversity_20260601_articles | 6.623748 | 5.878013 | 11.259% | NO |
| source_wikimedia_trwiki_20260601_articles_p1p1500000 | 6.567373 | 5.530270 | 15.792% | NO |
| source_wikimedia_trwiki_20260701 | 6.218733 | 5.290398 | 14.928% | NO |
| source_wikimedia_trwikibooks_20260601_articles | 8.089769 | 7.676311 | 5.111% | NO |
| source_wikimedia_trwikibooks_20260701 | 6.711619 | 5.857870 | 12.720% | NO |
| source_wikimedia_trwikivoyage_20260601_articles | 6.450075 | 5.506991 | 14.621% | NO |
| training_distribution | 6.047763 | 5.242410 | 13.317% | NO |
| turkish_prose | 6.567373 | 5.530270 | 15.792% | NO |
| turkish_technical | 8.089769 | 7.676311 | 5.111% | NO |
| validation | 6.603610 | 5.701172 | 13.666% | NO |

All Turkish, English, prose, technical, and source-family probes avoided catastrophic regression. Turkish technical improved more slowly than the other primary probes and remains a disclosed weakness.

## Generation diagnostics

| Step | Greedy repetition | Greedy loops | Sampling repetition | Sampling loops |
|---:|---:|---:|---:|---:|
| 610 | 100.0% | 100.0% | 81.2% | 81.2% |
| 1,024 | 100.0% | 100.0% | 56.2% | 56.2% |
| 1,536 | 75.0% | 75.0% | 50.0% | 50.0% |
| 1,831 | 68.8% | 68.8% | 37.5% | 31.2% |
| 2,048 | 81.2% | 81.2% | 50.0% | 50.0% |
| 2,560 | 43.8% | 43.8% | 18.8% | 18.8% |
| 3,051 | 68.8% | 56.2% | 37.5% | 37.5% |

Authoritative final audit: `200` greedy and `500` fixed-seeded generations; hard failures `0`.
Final greedy repetition/loop rates: `50.0%` / `29.0%`.
Final sampling repetition/loop rates: `19.8%` / `16.8%`.
Raw generations are retained without sanitization. Generation quality remains diagnostic and does not make this checkpoint a chatbot or release candidate.

## Runtime and restart

Active/wall throughput: `16135.8` / `13426.8` tokens/s.
Peak allocated/reserved VRAM: `1,761,924,608` / `2,055,208,960` bytes.
GPU temperature min/mean/max: `53.0` / `67.1` / `70.0` C.
GPU power min/mean/max: `13.1` / `72.8` / `90.0` W.
Real process restart: `PASS`; PIDs `[20664, 26840]`.
Optimizer, scheduler, RNG, data position, and deterministic sequence order all passed continuation checks.

One initial Segment-A attempt stopped at durable step 1379 because Windows briefly denied the atomic `progress.json` replace. Its evidence is preserved unchanged. A retry-safe bounded rename policy was added, and the official retry1 run restarted from the immutable step-610 source and completed both exact segments without further I/O failure.

## Decision

**STRONG PASS**: validation and eval improved by more than 8%, rebound stayed within 2%, late milestones did not worsen, every fixed probe avoided catastrophic regression, and all integrity gates passed.

Prepare a separate approval request for continuation toward the 100M gate; do not start it automatically.

No 100M continuation, SFT, Qwen teacher generation, corpus modification, tokenizer modification, public release, or Hugging Face upload occurred.
