# Phase 5A Corpus V4 candidate-source plan

## Planning boundary

This is a registry proposal, not acquisition approval. No source was downloaded in Phase 5A. Each future snapshot requires an official version pin, hash manifest, item-level license record, attribution record, PII screen, benchmark exclusion, and cross-source deduplication before admission.

Registry validation: **PASS**. The registry contains 20 candidates: 5 approved for planning, 10 conditional, 2 deferred, and 3 rejected.

| State | Sources | Expected unique capacity | Expected raw bytes |
| --- | ---: | ---: | ---: |
| approved | 5 | 52,000,000 | 15,500,000,000 |
| conditional | 10 | 286,000,000 | 884,000,000,000 |
| deferred | 2 | 22,000,000 | 20,300,000,000 |
| rejected | 3 | 0 | 0 |

Only 52M tokens are currently planning-approved without further item-level conditions. The 338M approved-plus-conditional capacity is sufficient on paper for the 200M target, but 286M of it is not admissible until its conditions are resolved. Capacity is not a legal conclusion and does not authorize bulk retrieval.

## Approved planning candidates

| Source | Role | Cap | Required handling |
| --- | --- | ---: | --- |
| GOV.UK OGL content | English institutional and educational prose | 25M | Exclude personal data, third-party rights, logos, and differently marked pages; retain OGL attribution. |
| MDN content | English technical documentation and code forms | 15M | Pin a Git commit; keep page attribution, ShareAlike scope, modification notices, and code-sample licensing separate. |
| Rust official documentation | Technical documentation and code | 4M | Pin the 1.90.0 tag; retain MIT/Apache notices and exclude separately licensed vendor material. |
| PostgreSQL 18 documentation | Technical and structured text | 3M | Pin the version; preserve the PostgreSQL license and provenance. |
| Turk Kutuphaneciligi CC BY articles | Turkish academic and educational prose | 5M | Admit only article-level CC BY 4.0 records with complete title, author, URL, and modification attribution. |

## Conditional candidates

- NIST cleared publications: admit only U.S. government-authored text after third-party exclusions.
- EUR-Lex reusable text: separate EU-owned reusable material from third-party rights and preserve source notices.
- DGT-Acquis TR/EN aligned subset: use only documents covered by the applicable Commission reuse terms; retain alignment provenance.
- DergiPark: article-level allowlist only; reject missing, NC-only, ND, or unclear commercial-reuse terms.
- Project Gutenberg Turkish works: perform work-level public-domain review and preserve Project Gutenberg terms and identifiers.
- PMC OA: accept only machine-readable CC0, CC BY, or CC BY-SA articles; exclude NC and unclear records.
- OpenStax: pin legacy CC BY editions only; do not treat newer 2026 CC BY-NC-SA material as eligible.
- Stack Exchange dump: preserve per-post attribution and date-dependent CC BY-SA version; filter personal and benchmark-like content.
- GitHub allowlist: explicit repository and file allowlist under approved permissive licenses; visibility alone is not a license.
- arXiv explicit-license subset: accept only per-paper CC0, CC BY, or CC BY-SA records; exclude default arXiv distribution licenses.

## Deferred and rejected

The Turkiye open-data document source is deferred until dataset-level text-reuse rights are established. Incremental Python documentation is deferred because Corpus V3 already contains closely related material and its marginal diversity is low.

Unbounded Common Crawl, social-media scraping, and uncontrolled generated-text sweeps are rejected. The plan also excludes private conversations, leaked material, pirated books, benchmark test sets, machine-translated filler, duplicate filling, and sources with unclear rights.

## Acquisition gates

1. Resolve license and redistribution terms before network access.
2. Pin a dated snapshot, release, or Git commit and record official URLs.
3. Predeclare source and category caps; no source may exceed 15% of the tranche.
4. Download only the smallest approved subset needed for the next inventory gate.
5. Hash raw inputs, preserve attribution incrementally, and reject files lacking provenance.
6. Normalize and deduplicate within source, against Corpus V3, and across V4 candidates.
7. Screen PII and benchmark contamination before token allocation.
8. Admit only measured unique tokens; never fill a quota with duplicates or synthetic text.

The target needs at least eight distinct source families and should avoid further Wikimedia concentration. Turkish remains primary, while English, technical documentation, code/structured text, and controlled bilingual material receive materially larger representation than in Corpus V3.
