# DarkMind v2 Phase 1B Corpus Report

Result: **PASS**

## Final Corpus

- Final normalized characters: 49999936
- Turkish characters: 29999974
- English characters: 19999962
- Target: 50,000,000 total; 30,000,000 Turkish; 20,000,000 English; each within +/-1%.

## Splits

| Split | Documents | Characters |
| --- | ---: | ---: |
| train | 313296 | 45001542 |
| validation | 17966 | 2498701 |
| eval | 17597 | 2499693 |

## Source Allocations

| Source | Candidate TR | Candidate EN | Planned TR | Planned EN | Selected TR | Selected EN | Selected total | Cap | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| wikimedia_trwiki_20260601_articles_p1p1500000 | 19983785 | 0 | 19983785 | 0 | 19970079 | 0 | 19970079 | 20000000 | PASS |
| python_docs_en_3_14_6_text | 0 | 7480086 | 0 | 7480086 | 0 | 7116858 | 7116858 | 7500000 | PASS |
| python_docs_tr_3_14_6_text | 2281545 | 0 | 2281545 | 0 | 2141593 | 0 | 2141593 | 7500000 | PASS |
| tatoeba_sentences_detailed_20260704 | 6752531 | 8227532 | 6114178 | 7539472 | 6272967 | 8000554 | 14273521 | 15000000 | PASS |
| wikimedia_enwikiversity_20260601_articles | 0 | 4980442 | 0 | 4980442 | 0 | 4882550 | 4882550 | 5000000 | PASS |
| wikimedia_trwikibooks_20260601_articles | 980610 | 0 | 980610 | 0 | 980174 | 0 | 980174 | 1000000 | PASS |
| wikimedia_trwikivoyage_20260601_articles | 639882 | 0 | 639882 | 0 | 635161 | 0 | 635161 | 700000 | PASS |

## Content And Rejections

| Content type | Selected characters |
| --- | ---: |
| dialogue_and_sentence_prose | 14273521 |
| educational_articles | 4882550 |
| encyclopedic_articles | 19970079 |
| instructional_books | 980174 |
| technical_documentation | 9258451 |
| travel_guide_articles | 635161 |

- Input rejections by reason: document_length=588962, exact_duplicate=4444, language_mismatch=85602, mojibake=190, near_duplicate=1609, pii_or_url=4689, replacement_character=5, unsafe_control_character=1
- Exact duplicate removals: 4444
- Near-duplicate removals: 1609

## Hygiene Gates

| Gate | Value |
| --- | --- |
| invalid_utf8 | 0 |
| mojibake_detections | 0 |
| replacement_characters | 0 |
| language_mismatch | 0 |
| unresolved_exact_duplicates | 0 |
| unresolved_near_duplicate_clusters | 0 |
| missing_license_metadata | 0 |
| missing_attribution_metadata | 0 |
| source_cap_violations | 0 |
| deterministic_split_hashes_present | True |
| deterministic_rebuild_verification | PASS |

## Manifest Hashes

| Artifact | SHA-256 |
| --- | --- |
| attribution_manifest.jsonl | `b820ec56a0c173604a5a97663c1ca510c7c78800d3e559722b23ffb74eb3120f` |
| corpus_manifest.json | `283869111766281c89fb75f7cae43c1cdcb07d4bdbd5770f6685f6da4f74a4f1` |
| determinism_verification.json | `a0401f006f42da9fc66e5277cb61a65213e641e27bd49916165c70a9959788df` |
| rejected_documents.jsonl | `e45d88ee3e5fd16fb114ce2ea505a0b55dba8d4d87a4cbb4a925b61c2dbf42b5` |
| source_allocation.json | `0dc3e32bbb1df70f60b328b6c7ebfe05944f9b11494ae2864ecabcd0cd148521` |
| split_manifest.json | `959f7f56ef503b991b8860722468fba8040561b249518dc312ce2ab4565344f3` |
| tokenizer_eval.txt | `208e05f3a08ffdddcf9fcc78b941e81b0a5a7f8be989903a6a9979df24b47a7e` |
| tokenizer_train.txt | `f1ac92acd9faf5a4ef909f400f7bfdb0b0093d96085834877cdeb0d0d5b1152f` |
| tokenizer_validation.txt | `f5869481d09fafde637ee3c2227e8823e05cc0c1e9e45b15fd5d005267157e58` |

## Determinism

- Verification mode rendered the finalized split and manifest content a second time from the same fixed seed and stable ordering.
- Deterministic rebuild verification: PASS

## Risks

- No unresolved corpus hygiene or source-cap risks detected.

TOKENIZER CANDIDATE TRAINING IS READY FOR USER APPROVAL
