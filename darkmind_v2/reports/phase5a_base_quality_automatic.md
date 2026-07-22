# Phase 5A automatic base-quality review

This is a deterministic base-continuation diagnostic. Automatic proxies are not human semantic-quality scores.

## Immutable evaluation identity

- Checkpoint: `C:\DarkMindRuntime\phase4f\runs\base_v1_first_corpus_pass_completion_v2\checkpoints\step_011972_tokens_098074624`
- Model SHA-256: `458816257836a60d804a373c17c617642c99e413c6c190d4fd1e2f73b95fd993`
- Prompt manifest SHA-256: `c82db48e4276d4a9a4d90ea1752956a55848869c55ea0e2ce590358eb39f9197`
- Unique prompts: 440
- Raw outputs: `C:\DarkMindRuntime\phase5a\evaluations\base_quality_suite_v1_v2` (outside Git)

## Aggregate decoding health

| Mode | Generations | Repetition | Exact loops | EOS | Empty | Language consistency | Switch errors | Code proxy | Punctuation proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 440 | 60.0% | 56.8% | 2.0% | 0.0% | 100.0% | 0.0% | 65.0% | 5.5% |
| seeded_sampling | 440 | 33.2% | 33.2% | 8.4% | 0.0% | 100.0% | 0.0% | 67.5% | 16.4% |

## Prompt perplexity by category

Prompt perplexity measures model fit to the controlled prompt text; it is not a semantic score for the continuation.

| Category | Prompts per mode | Greedy loss | Greedy PPL | Sampling loss | Sampling PPL |
|---|---:|---:|---:|---:|---:|
| code_structured | 40 | 4.2379 | 69.3 | 4.2379 | 69.3 |
| english_educational | 30 | 6.3885 | 595.0 | 6.3885 | 595.0 |
| english_general | 60 | 6.6245 | 753.3 | 6.6245 | 753.3 |
| english_technical | 30 | 6.4271 | 618.4 | 6.4271 | 618.4 |
| eos_completion_behavior | 20 | 6.8594 | 952.8 | 6.8594 | 952.8 |
| factual_context | 40 | 7.1422 | 1264.2 | 7.1422 | 1264.2 |
| language_switch_resistance | 20 | 6.4375 | 624.8 | 6.4375 | 624.8 |
| long_context_consistency | 20 | 6.4984 | 664.1 | 6.4984 | 664.1 |
| punctuation_paragraph | 20 | 6.7438 | 848.7 | 6.7438 | 848.7 |
| turkish_educational | 40 | 6.7578 | 860.8 | 6.7578 | 860.8 |
| turkish_general | 80 | 7.0328 | 1133.2 | 7.0328 | 1133.2 |
| turkish_technical | 40 | 7.1258 | 1243.6 | 7.1258 | 1243.6 |

## Language fit

| Mode | Language | Prompts | Mean loss | Perplexity |
|---|---|---:|---:|---:|
| greedy | en | 220 | 6.0697 | 432.5 |
| greedy | tr | 220 | 7.0480 | 1150.6 |
| seeded_sampling | en | 220 | 6.0697 | 432.5 |
| seeded_sampling | tr | 220 | 7.0480 | 1150.6 |

## Integrity, Unicode, and bounded memorization evidence

- Greedy/sampling special-token leakage: 0 / 0.
- Greedy/sampling invalid UTF-8 sequences: 0 / 0.
- Exact train/held-out continuations in the bounded audit: 0 / 0.
- Longest true-continuation match: 8 tokens; longest generated n-gram found in train shards: 16 tokens.
- Extraction risk is not claimed to be zero.

## Interpretation boundary

The complete suite still requires blinded human review. Repetition, loop, EOS, language-ID, bracket balance, punctuation, overlap, and perplexity measurements are automatic health indicators only. They do not establish factual reliability, topical coherence, or usefulness.
