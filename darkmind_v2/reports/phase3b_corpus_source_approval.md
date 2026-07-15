# Phase 3B Corpus Source Approval

Status: **PLANNING APPROVAL ONLY; DOWNLOAD NOT AUTHORIZED**

All 20 Phase 3A candidates were reclassified under the stricter production policy. The machine-readable registry contains the official URL, dated snapshot status, license and redistribution evidence, attribution, checksum source, raw-byte estimate, usable-token estimate, language/category, cap, quality tier, deduplication, contamination, PII, and extraction-readiness fields for every source.

| Status | Sources | Expected usable tokens |
|---|---:|---:|
| approved | 8 | 365,000,000 |
| conditional | 5 | 95,000,000 |
| deferred | 4 | 140,000,000 |
| rejected | 3 | 0 |

Approved sources contribute an estimated **365,000,000 unique tokens**, leaving **135,000,000** to 500M and **635,000,000** to 1B. Approved plus conditional sources reach 460M, leaving 40M and 540M respectively.

## First 100M Tranche Proposal

| Source | Tokens | Language |
|---|---:|---|
| Turkish Wikipedia 2026-07-01 | 55,000,000 | tr |
| English Wikipedia 2026-07-01 | 30,000,000 | en |
| Python 3.14.6 Turkish docs | 5,000,000 | tr |
| Python 3.14.6 English docs | 10,000,000 | en |

Total: 100M tokens, 60% Turkish and 40% English. Each tranche cap is below its registry maximum and requires exact snapshot/checksum verification before any acquisition.

## Conditions and Deferrals

### Conditional

- `wikimedia_trwikivoyage_20260701`: Requires aggressive address, phone, commercial-listing, and template filters.
- `tatoeba_tr_en_20260713`: Requires sentence and translation-chain attribution plus license completeness enforcement.
- `mdn_content_20260713`: Requires file-level prose/code license separation and pre-2010 code handling.
- `project_gutenberg_en_20260713`: Requires work-level public-domain checks for the training jurisdiction and header cleanup.
- `stack_exchange_20260701`: Requires revision-date license versioning, user attribution, and code/content separation.

### Deferred

- `wikimedia_trwikisource_20260701`: Page-level public-domain and contributor-license filtering is not implemented.
- `wikimedia_enwikisource_20260701`: Work-level copyright jurisdiction and page filtering are not implemented.
- `resmi_gazete_archive_20260701`: Redistribution clearance, stable bulk access, checksums, and extraction are unresolved.
- `mevzuat_archive_20260701`: Redistribution clearance, dated bulk snapshot, checksums, and extraction are unresolved.

### Rejected

- `common_crawl_bulk`: Bulk document rights, privacy, attribution, and quality remain unresolvable for this plan.
- `reddit_user_content`: No approved training and redistribution grant exists for user content.
- `oscar_web_corpus`: Document-level rights, attribution, privacy, and redistribution evidence are insufficient.

Approved status does not authorize download. Corpus acquisition remains stopped until the user approves an exact source tranche and its snapshot/checksum plan.
