# Phase 2B.2 Public Research-Preview Audit

## Decision

**FAIL - additional pretraining and release preparation are required before public release.**

Generation pipeline hard failures were zero, but output health is too unstable for a public model artifact: 89.5% of greedy outputs received repetition warnings, 65.5% contained exact repeated n-gram loops, all ten greedy code prompts failed a conservative continuation check, and manual review found no reliable Turkish, English, technical, or code continuation. In addition, no standalone model-weight distribution license has been designated.

## Checkpoint

- Selected checkpoint: `step_000256_tokens_001048576`
- Model SHA-256: `5e2fd69d4775940629926a7bf659e36beafe4c1cd544feb8d97beeae6537b097`
- Parameters: 9,369,088
- Training tokens / steps: 1,048,576 / 256
- Corpus fraction seen: approximately 8.9% of one train epoch
- Validation loss / eval loss / perplexity: 7.096028 / 7.081271 / 1,207.163

## Audit Set

The source-controlled manifest contains exactly 200 base-model continuation prompts: 60 Turkish ordinary, 30 Turkish factual, 20 Turkish technical, 50 English ordinary, 20 English factual, 10 English technical, and 10 code/structured prompts. It does not use private data or long copyrighted passages and does not claim comprehensive factual measurement.

## Mechanical Results

| Metric | Greedy | Seeded sampling |
| --- | ---: | ---: |
| Generations | 200 | 500 |
| Hard failures | 0 | 0 |
| Invalid UTF-8 byte sequences | 0 | 0 |
| U+FFFD characters | 0 | 0 |
| Mojibake outputs | 0 | 0 |
| Unexpected-script outputs | 0 | 0 |
| Mixed-script outputs | 0 | 0 |
| Special-token leakage | 0 | 0 |
| Empty outputs | 2 | 0 |
| Exact repeated n-gram loop outputs | 131 | 67 |
| Longest repeated-token run | 31 | 10 |
| Mean unique-token ratio | 0.4102 | 0.7094 |
| P50 unique-token ratio | 0.1522 | 0.7500 |
| P50 / P90 / P95 output tokens | 32 / 32 / 32 | 16 / 32 / 32 |
| EOS completion rate | 41.0% | 64.6% |

Sampling used two profiles (`0.7/0.9/40` and `0.9/0.95/50` for temperature/top-p/top-k), five fixed seeds, and a deterministic 50-prompt stratified subset.

## Quality Warnings

| Warning | Greedy | Seeded sampling |
| --- | ---: | ---: |
| repetition | 179 | 158 |
| very_short_output | 58 | 61 |
| empty_output | 2 | 0 |
| factual_unreliability | 50 | 130 |
| code_generation_failure | 10 | 12 |
| unexpected_script | 0 | 0 |
| mixed_script | 0 | 0 |
| special_token_leakage | 0 | 0 |

`factual_unreliability` is conservatively attached to factual-category outputs because this audit does not verify factual correctness. Automatic metrics do not score coherence; representative manual review found widespread incoherence even where mechanical warnings were absent.

## Hard-Gate Accounting

All generation integrity hard failures were zero: tokenizer/checkpoint/config/corpus hashes matched; token IDs stayed in range; Unicode, byte-fallback, U+FFFD, normal-piece mojibake, output mojibake, decoder, logits, safetensors, AutoModel, and AutoTokenizer checks passed. Model-card disclosures, corpus attribution references, and license status are now present.

The local package is technically valid, but `model_weight_distribution_license_not_designated` remains a disclosed release-preparation warning. Together with severe output instability, it blocks the eligibility decision.

## Required Next Work

1. Continue base pretraining under a separately approved training phase and token budget.
2. Repeat this exact deterministic audit and require a substantial reduction in repetition/loop rates plus meaningful representative continuations.
3. Select and document model-weight distribution terms after reviewing corpus-license obligations.
4. Do not begin SFT merely to conceal an undertrained base checkpoint.

Authoritative runtime manifests are under `evaluations/public_preview_v2/`. The earlier `public_preview_v1/` was preserved because it contained an EOS-as-leakage taxonomy error and is not authoritative.
