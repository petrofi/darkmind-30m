# DarkMind v2 Selected Tokenizer Decision

## Selection

- Selected candidate: D
- Tokenizer type: SentencePiece BPE
- Vocabulary size: 24,000
- Frozen tokenizer name: `darkmind_v2_sp_bpe24k_v1`
- Byte fallback: enabled
- Reserved special-token IDs: 0-7
- All tokenizer acceptance hard gates: PASS

## Corpus References

| Artifact | SHA-256 |
| --- | --- |
| tokenizer train split | `f1ac92acd9faf5a4ef909f400f7bfdb0b0093d96085834877cdeb0d0d5b1152f` |
| tokenizer validation split | `f5869481d09fafde637ee3c2227e8823e05cc0c1e9e45b15fd5d005267157e58` |
| tokenizer eval split | `208e05f3a08ffdddcf9fcc78b941e81b0a5a7f8be989903a6a9979df24b47a7e` |
| processed corpus manifest | `283869111766281c89fb75f7cae43c1cdcb07d4bdbd5770f6685f6da4f74a4f1` |
| attribution manifest | `b820ec56a0c173604a5a97663c1ca510c7c78800d3e559722b23ffb74eb3120f` |
| split manifest | `959f7f56ef503b991b8860722468fba8040561b249518dc312ce2ab4565344f3` |

## Candidate Comparison

| Candidate | Type | Vocabulary | Hard gates | Weighted score |
| --- | --- | ---: | --- | ---: |
| A | SentencePiece BPE | 12,000 | PASS | 20.000000 |
| B | SentencePiece BPE | 16,000 | PASS | 53.912577 |
| C | SentencePiece Unigram | 16,000 | PASS | 51.245497 |
| D | SentencePiece BPE | 24,000 | PASS | 90.000000 |

Candidate D was selected because it produced the best Turkish, English, and technical/code token efficiency and the shortest p95/p99 sequence tails while preserving zero unknown tokens, zero round-trip failures, and a clean vocabulary. Its lead exceeded the two-point tie-break window, so the smaller-vocabulary preference did not override the result.

## Parameter-Cost Decision

The 24k vocabulary is accepted because the measured compression improvement was substantial across every primary efficiency metric. The cost is nevertheless architectural, not incidental: with 384-dimensional tied embeddings the vocabulary uses 9.216M parameters, while 512-dimensional tied embeddings use 12.288M parameters.

**Tied input/output embeddings are required by default.** An untied LM head is not allowed by default, especially for a 45M-class model. At 512 dimensions an untied 24k vocabulary consumes 24.576M parameters, or about 54.61% of a 45M model.

Future model guidance:

- A tiny smoke model may use this tokenizer only for compatibility and pipeline validation.
- A real base model should target approximately the 60M class when using a 512-dimensional embedding.
- The exact total and vocabulary-related parameter counts must be calculated before any training starts.
- The copied `training_config.json` records the pre-freeze candidate run; this decision and `tokenizer_freeze_manifest.json` are the authoritative freeze records.
- The 149 KB training log is retained because it is small and preserves SentencePiece training provenance.

No model has been trained with this frozen tokenizer. No instruction tuning or SFT has started. Pilot500 will not be reused until a newly trained base model passes the base-quality gates.
