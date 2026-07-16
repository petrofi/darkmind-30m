# DarkMind v2 Base V1 Stage-1 5M

Classification: **FAIL**
Pipeline integrity: **PASS**
Learning-quality gate: **FAIL**
Best checkpoint: **step 128**
Final validation and eval worse than initialization: **YES**

Model: darkmind-v2-base-v1 / 118,056,960 parameters
Config SHA-256: `8e9775721b0173a92e88de15c2195428932b3aa5beec57d568674c25887c5e39`
Architecture hash: `3a2dda86293ceae23ca4e50ea47c840b7fc46021d293c862d330110851ac8305`
Corpus hash: `e75c4aa4f39cc7a3cb4fe754e2a0e85268ced300f8504a86d443540eb609e1c5`
Tokenized manifest hash: `1296caacf09d49b1c48c0fee7d5f5a523a0019e8e7e0e70132fbf68d8f023c82`
Initialization hash: `f1da070885650b70dc999f22b6ef8a438bb47fe7479020dc46ebdf68ae3d9c6b`

Optimizer steps / tokens: 610 / 4,997,120
Scheduler position: 610 of 12,207
Final next-step LR: 0.0002988150
Nominal 100M scheduler unused tail: 256 tokens
Current complete train-sequence capacity deficit to scheduler horizon: 1,917,952 tokens

| Step | Tokens | Validation loss | Eval loss | Eval perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 10.246471 | 10.243818 | 28108.240 |
| 128 | 1,048,576 | 8.388659 | 8.321871 | 4112.850 |
| 305 | 2,498,560 | 9.485073 | 9.377119 | 11814.926 |
| 458 | 3,751,936 | 10.317393 | 10.185603 | 26518.646 |
| 610 | 4,997,120 | 10.849637 | 10.699401 | 44329.285 |

Validation improvement: -5.887%
Eval improvement: -4.447%
Final train / validation gap: 1.979259
Active throughput: 18486.7 tokens/s
Complete wall throughput: 5982.1 tokens/s
Peak allocated / reserved VRAM: 1,049,673,728 / 1,897,922,560 bytes
Temperature range: [59.0, 69.0]
Power range: [48.43, 86.94]

## Calibration

Measured optimizer steps: 30
Active / full-wall throughput: 18404.6 / 12396.3 tokens/s
p50 / p95 step: 0.4434 / 0.4520 seconds
Peak allocated / reserved VRAM: 1,755,868,672 / 1,904,214,016 bytes
Optimizer state: 472,228,528 bytes

## Checkpoints

| Step | Model SHA-256 | Model bytes | Resume bytes |
|---:|---|---:|---:|
| 0 | `f1da070885650b70dc999f22b6ef8a438bb47fe7479020dc46ebdf68ae3d9c6b` | 239,801,512 | 15,630 |
| 128 | `c93564e9b3b347e57c6ccb1c01a1f805a60d56c23aef4646a4294291d4b1bee3` | 239,801,512 | 472,389,504 |
| 305 | `7c486f06d3300c0d9eb7066aa7495af341d5aff23432935ba3aa4477e3a85260` | 239,801,512 | 472,389,504 |
| 458 | `d00b1e49033ba1ee08d374a230fd17089701bab520b20b958146d17027ca4960` | 239,801,512 | 472,389,504 |
| 610 | `a7dd0ef9bf4336482b1c15bdc7117bf7b4998846ed60af761d78b38a70a793e0` | 239,801,512 | 472,389,504 |

## Generation health

| Step | Greedy repetition | Greedy loops | Sampling repetition | Sampling loops | Greedy/sample EOS |
|---:|---:|---:|---:|---:|---|
| 0 | 49/50 | 49 | 0/50 | 0 | 0.000/0.000 |
| 128 | 36/50 | 4 | 8/50 | 8 | 0.920/1.000 |
| 305 | 50/50 | 42 | 1/50 | 1 | 0.200/1.000 |
| 458 | 40/50 | 29 | 0/50 | 0 | 0.520/1.000 |
| 610 | 42/50 | 35 | 0/50 | 0 | 0.300/1.000 |

The fresh-process midpoint restart passed model, optimizer, scheduler, RNG, data-position, validation-reproducibility, and learning-rate continuity checks.
The first Segment B process stopped after uncheckpointed step 319 because OneDrive temporarily locked the repeatedly replaced progress JSON. Its metrics and traceback evidence are preserved. The immutable step-305 checkpoint was unchanged, and the single clean-process retry passed through step 610.
All raw generation outputs remain in ignored runtime manifests. Repetition, exact loops, EOS, Unicode, byte fallback, scripts, and finite logits are reported without sanitization.
Absolute Phase 3B finalist losses are not treated as comparable because the corpus, split, and production scheduler differ.
The model is not instruction-tuned, not a chatbot, not production-ready, and not publicly released.
Hugging Face upload performed: **NO**

25M continuation recommended: **NO**

DARKMIND V2 BASE V1 STAGE-1 5M FAILED AND TRAINING IS STOPPED
