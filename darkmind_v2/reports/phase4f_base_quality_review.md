# DarkMind v2 Phase 4F Base Quality Review

First-pass classification: **PASS**

This review covers the first deterministic Corpus V3 pass only. The model is a from-scratch Base V1 model, not instruction-tuned, not a chat model, not production-ready, and not publicly uploaded.

## Exact completion identity

- Start: step 9,155, 74,997,760 tokens.
- Final no-wrap stop: step 11,972, 98,074,624 tokens, 191,552 sequences.
- The deterministic 14-sequence incomplete tail was excluded; step 11,973 and a second epoch were not run.
- Best validation checkpoint: step 11,972 at `C:\DarkMindRuntime\phase4f\runs\base_v1_first_corpus_pass_completion_v2\checkpoints\step_011972_tokens_098074624`.

## Loss progression

| Step | Tokens | Applied LR | Train | Validation | Eval | Eval perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 9,155 | 74,997,760 | 0.000040413947 | 5.214144 | 5.091754 | 5.048094 | 155.725 |
| 10,375 | 84,992,000 | 0.000033880803 | 5.285613 | 5.071706 | 5.028681 | 152.731 |
| 10,986 | 89,997,312 | 0.000031742048 | 5.099640 | 5.064458 | 5.022116 | 151.732 |
| 11,596 | 94,994,432 | 0.000030438972 | 4.927190 | 5.059540 | 5.016205 | 150.838 |
| 11,972 | 98,074,624 | 0.000030065053 | 5.113889 | 5.057351 | 5.013796 | 150.475 |

75M to final validation/eval improvement: 0.676% / 0.679%. Rebound: 0.000% / 0.000%.

## Optimization and activations

Gradient p50/p95/max: 1.9453 / 2.2188 / 3.1094. Clipping rate: 100.0%.
Clipping coefficient p50/p95/min: 0.514056 / 0.558952 / 0.321608.
Update-to-weight p50/p95/max: 0.00017760 / 0.00023117 / 0.00026623. Non-finite events: 0.
Persistent clipping remains a production-policy warning. It was not treated as an automatic failure because losses, updates, logits, and activations remained controlled under the frozen policy.

| Step | Logit std | Prediction entropy | Embedding norm | Final/early residual RMS |
|---:|---:|---:|---:|---:|
| 10,375 | 1.926770 | 5.522699 | 0.660283 | 64.084333 |
| 10,986 | 1.932041 | 5.521070 | 0.660301 | 63.875836 |
| 11,596 | 1.926926 | 5.509566 | 0.660314 | 63.318023 |
| 11,972 | 1.929437 | 5.494267 | 0.660320 | 63.208987 |

## Language and category probes

| Probe | 75M loss | Final loss | Improvement | Catastrophic regression |
|---|---:|---:|---:|---|
| english_prose | 5.500023 | 5.458065 | 0.763% | NO |
| english_technical | 5.852518 | 5.847260 | 0.090% | NO |
| eval | 5.466234 | 5.444838 | 0.391% | NO |
| source_python_docs_en_3_14_6 | 3.118254 | 3.110546 | 0.247% | NO |
| source_python_docs_en_3_14_6_text | 5.852518 | 5.847260 | 0.090% | NO |
| source_python_docs_tr_3_14_6 | 4.623624 | 4.590641 | 0.713% | NO |
| source_python_docs_tr_3_14_6_text | 5.765424 | 5.746211 | 0.333% | NO |
| source_tatoeba_sentences_detailed_20260704 | 5.620668 | 5.600035 | 0.367% | NO |
| source_wikimedia_enwiki_20260701 | 4.457373 | 4.423625 | 0.757% | NO |
| source_wikimedia_enwikibooks_20260701 | 4.854400 | 4.860835 | -0.133% | NO |
| source_wikimedia_enwikiversity_20260201 | 4.914685 | 4.899632 | 0.306% | NO |
| source_wikimedia_enwikiversity_20260601_articles | 5.541475 | 5.541436 | 0.001% | NO |
| source_wikimedia_trwiki_20260601_articles_p1p1500000 | 5.057325 | 5.000535 | 1.123% | NO |
| source_wikimedia_trwiki_20260701 | 4.772876 | 4.722937 | 1.046% | NO |
| source_wikimedia_trwikibooks_20260601_articles | 7.288141 | 7.262280 | 0.355% | NO |
| source_wikimedia_trwikibooks_20260701 | 5.543173 | 5.517723 | 0.459% | NO |
| source_wikimedia_trwikivoyage_20260601_articles | 5.134255 | 5.102798 | 0.613% | NO |
| training_distribution | 4.748456 | 4.709315 | 0.824% | NO |
| turkish_prose | 5.057325 | 5.000535 | 1.123% | NO |
| turkish_technical | 7.288141 | 7.262280 | 0.355% | NO |
| validation | 5.310241 | 5.282865 | 0.516% | NO |

Turkish and English grammatical/topical continuation, technical continuation, factual reliability, and code/structured behavior remain base-model diagnostics. Automatic probe improvements do not establish factual correctness or user-facing quality.

## Final generation audit

| Mode | Generations | Repetition | Exact loops | EOS | Mean unique-token ratio | Empty | Short |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 200 | 45.5% | 36.0% | 8.0% | 0.313 | 6 | 1 |
| sampling | 500 | 17.0% | 15.8% | 16.6% | 0.679 | 7 | 0 |

The audit retained unsanitized raw outputs outside Git. Invalid-byte, replacement-character, mojibake, script-consistency, and special-token leakage counters are recorded in runtime evidence. No chatbot-quality claim is made.

Automatic non-empty/non-repetitive continuation-health proxy counts were Turkish 28/90, English 10/80, and technical 0/30. These are not semantic-quality judgments; manual review remains required.

## Memorization and PII

Controlled prefixes: 48 train and 48 validation/eval. Exact continuation rates: 0.0% train and 0.0% held-out.
Longest exact target span: 8 tokens. Longest generated n-gram found in the full training shards: 16 tokens.
The broad regex initially flagged 5 numeric phone-like candidates and produced an initial FAIL record. That immutable record is preserved at hash `b400c8acb99acf1ced76fac5fc4b9a4fced85dd82b2c0be31d5eda9329f932b2`.
Deterministic adjudication found plausible identities {'email': 0, 'phone': 0, 'url': 0}; all five phone-like candidates were repetitive year/ISBN-like numeric false positives. Material personal-data reproductions: 0. Final hard blockers: none.
Extraction risk is not claimed to be zero. The audit is controlled and bounded; it does not prove absence of memorization outside its probes.

## Separate base-quality review

| Dimension | Assessment |
|---|---|
| Turkish grammatical continuation | Emerging structure, but still weak and repetition-prone; health proxy 28/90. |
| Turkish topical consistency | Some probe improvement, but sustained topical coherence is not established. |
| English grammatical continuation | Emerging structure remains weaker than Turkish; health proxy 10/80. |
| English topical consistency | Not established; source probes are mostly flat-to-positive. |
| Technical-text continuation | Weak; health proxy 0/30 and English technical loss improved only marginally. |
| Factual reliability | Not reliable; factual outputs retain an explicit unreliability warning. |
| Code/structured output | Weak and not suitable for use; code-generation warnings remain. |
| Repetition | High: greedy 45.5%, sampling 17.0%. |
| EOS behavior | Weak: greedy 8.0%, sampling 16.6%. |
| Memorization risk | Controlled audit passed after preserving and adjudicating the initial broad-regex failure; risk is not claimed to be zero. |
| PII risk | No plausible identity or material personal-data reproduction was observed in the bounded audit. |

## Base-quality decision

The model has learned measurable Turkish/English and technical language structure, with final classification **PASS, but not Strong PASS**. Learning slowed after 75M tokens: final validation and eval losses improved by only 0.676% and 0.679%, respectively, from 75M to the exact first-pass stop. It remains too weak and insufficiently reviewed for instruction tuning or user-facing use.
New unique Corpus V3 expansion is the preferred next investment. A second identical epoch is not justified automatically and would require a separate approval with renewed overexposure and extraction-risk controls.
A stronger open-base Instruct track should remain separate from this Base V1 evidence line. No SFT, Qwen teacher generation, second epoch, or upload occurred.
