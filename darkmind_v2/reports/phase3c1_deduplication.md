# Phase 3C.1 Deduplication

Status: **PASS**

Exact duplicate removals: 330
Near-duplicate removals: 4,414
Historical paragraph-overlap removals: 25,408
Cross-source paragraph-overlap removals: 50,983
Unresolved exact duplicates: 0
Unresolved near-duplicate clusters: 0
Evaluation contamination removals: 6
Accepted evaluation contamination: 0
Rejected records retained: 182,828

| Rejection reason | Records |
|---|---:|
| control_character | 196 |
| cross_source_paragraph_overlap | 50,983 |
| empty_after_paragraph_deduplication | 1,567 |
| evaluation_contamination | 6 |
| exact_duplicate | 279 |
| excessive_punctuation | 697 |
| final_cross_source_exact_duplicate | 51 |
| markup_leakage | 5,554 |
| material_pii_email | 1,160 |
| material_pii_phone | 94 |
| material_pii_turkish_identity_number | 2 |
| mojibake | 2,111 |
| near_duplicate | 4,414 |
| phase1b_paragraph_overlap | 25,408 |
| post_dedup_excessive_punctuation | 18 |
| post_dedup_repeated_lines | 42 |
| post_dedup_wrong_or_uncertain_language | 63 |
| repeated_character | 666 |
| repeated_lines | 5,345 |
| repeated_paragraph_within_document | 58,527 |
| replacement_character | 91 |
| too_long | 394 |
| too_short | 17,339 |
| wrong_or_uncertain_language | 7,821 |
