# Phase 5B conservative capacity model

## Method

Optimistic capacity is a rights-filtered upper scenario, expected capacity is pre-dedup usable text after the most likely extraction and quality losses, and conservative capacity is the post-filter lower bound after license, language, PII, exact/near dedup, Corpus V3 overlap, benchmark, length and code-quality losses. Conditional and deferred values are planning signals only and never enter approved totals.

| Source | State | Optimistic | Expected | Conservative | Expected rejection | Confidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GOV.UK | Conditional | 35.0M | 20.0M | 12.0M | 42.9% | Low |
| MDN | Approved | 12.0M | 8.0M | 5.0M | 33.3% | Medium |
| Rust 1.90 | Approved | 2.5M | 1.5M | 1.0M | 40.0% | Medium |
| PostgreSQL 18.0 | Approved | 1.0M | 0.5M | 0.3M | 50.0% | Medium |
| Turk Kutuphaneciligi | Conditional | 6.0M | 3.0M | 1.5M | 50.0% | Low |
| NIST | Conditional | 50.0M | 25.0M | 12.0M | 50.0% | Low |
| EUR-Lex | Conditional | 50.0M | 25.0M | 12.0M | 50.0% | Low |
| DGT-Acquis | Deferred | 4.0M | 2.0M | 1.0M | 50.0% | Medium |
| DergiPark allowlist | Conditional | 60.0M | 25.0M | 10.0M | 58.3% | Low |
| Gutenberg Turkish | Conditional | 3.0M | 1.5M | 0.7M | 50.0% | Low |
| PMC OA | Conditional | 80.0M | 40.0M | 20.0M | 50.0% | Low |
| OpenStax legacy | Conditional | 25.0M | 12.0M | 6.0M | 52.0% | Low |
| Stack Exchange | Conditional | 50.0M | 20.0M | 8.0M | 60.0% | Low |
| GitHub allowlist | Conditional | 50.0M | 20.0M | 8.0M | 60.0% | Low |
| arXiv explicit CC | Conditional | 80.0M | 35.0M | 15.0M | 56.3% | Low |
| Turkiye portal | Deferred | 20.0M | 10.0M | 3.0M | 50.0% | Low |
| Python incremental | Deferred | 1.0M | 0.5M | 0.1M | 50.0% | Medium |
| Three rejected classes | Rejected | 0 | 0 | 0 | 100% | High |

## Approved capacity gate

- Approved expected pre-dedup usable capacity: **10,000,000 tokens**.
- Required approved expected capacity: **250,000,000 tokens**.
- Approved expected gap: **240,000,000 tokens**.
- Approved conservative post-filter capacity: **6,300,000 unique tokens**.
- Required approved conservative capacity: **200,000,000 unique tokens**.
- Approved conservative gap: **193,700,000 tokens**.

The approved set does not support acquisition of a 200M tranche. Even the conditional estimates remain low-confidence until exact inventories and samples exist. Capacity cannot be promoted by arithmetic alone: each source must first satisfy artifact, license, acquisition, attribution, quality and overlap gates.

## Loss accounting

Every future inventory must report separate token losses for extraction failure, language filtering, document quality, PII/secrets, license rejection, exact dedup, near dedup, Corpus V3 overlap, benchmark contamination, document length and code quality. Conservative capacity is recomputed from measured accepted unique tokens, never filled with duplicates or synthetic text.
