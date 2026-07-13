# DarkMind v2 Phase 2C Next-Model Requirements

## Evidence-Derived Floor

- Minimum non-vocabulary parameters: 45M.
- Maximum vocabulary/embedding share: 25%; prefer 20% or lower.
- Frozen vocabulary: 24,000 with tied input/output embeddings.
- Initial context length: 1,024; extend only after throughput and data-length analysis.
- Corpus: at least 250M-500M high-quality deduplicated tokens before architecture selection; target training tokens should follow the table below.
- Turkish/English balance, factual prose, and code/technical sources must be measured separately.

## Candidate Classes

These are parameter-count anchors, not a selected winner.

| Class | d_model | Layers | Heads | Exact params | Non-vocab params | Vocab share | Est. BF16 train VRAM | Est. active tok/s | Target train tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60M | 512 | 15 | 8 | 60,099,072 | 47,811,072 | 20.45% | 2.0-3.0 GiB | 9k-13k | 1.2B+ |
| 80M | 640 | 13 | 10 | 80,022,400 | 64,662,400 | 19.19% | 2.6-3.8 GiB | 7k-10k | 1.6B+ |
| 100M | 640 | 17 | 10 | 99,716,480 | 84,356,480 | 15.40% | 3.2-4.8 GiB | 5.5k-8k | 2.0B+ |
| 120M | 768 | 15 | 12 | 125,538,048 | 107,106,048 | 14.68% | 4.0-6.0 GiB | 4.5k-7k | 2.5B+ |

## Training-System Requirements

- Preserve deterministic uint16 shards, exact resume state, no hidden wraparound, and milestone public-preview audits.
- Gradient checkpointing: optional for 60M/80M, recommended for 100M, required for the 120M class on an 8 GiB GPU unless calibration proves otherwise.
- Use BF16, fused AdamW only after deterministic equivalence testing, and activation-memory measurements at context 1,024.
- Benchmark each class with the same effective token batch before selecting by throughput, VRAM, validation scaling, and generation quality.
- Do not train any candidate until corpus licensing, model-weight licensing, and a larger balanced corpus are ready.
