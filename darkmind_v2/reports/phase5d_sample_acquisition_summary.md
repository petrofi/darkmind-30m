# Phase 5D Bounded Sample Acquisition Summary

## Scope and controls

- Authorization: `darkmind-v2-phase5d-bounded-sampling-20260723`
- Selection seed: `20260723`
- Official candidates sampled: 6 of 8 maximum
- Downloaded bytes: 424,284,460 of 10,000,000,000 maximum
- Per-source maximum: 2,000,000,000 bytes; every source passed
- Runtime root: `C:\DarkMindRuntime\phase5d`
- Samples committed to Git: no
- Downloaded content executed: no
- Training or production tokenization: no

| Source | Bytes | Selected | Extracted | Success | Accepted | Post-overlap tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `govuk_content_ogl3_20260722` | 114,854 | 200 | 200 | 100.0% | 105 | 5,620 | Conditional |
| `govinfo_federal_register_2025_xml` | 67,225 | 0 | 0 | 0.0% | 0 | 0 | Deferred |
| `plos_ccby_jats_allowlist` | 49,354 | 200 | 40 | 20.0% | 11 | 479 | Conditional |
| `go_1_26_5_source_docs` | 34,140,216 | 500 | 500 | 100.0% | 493 | 2,172,878 | Approved |
| `kubernetes_website_f2987ba` | 334,486,939 | 500 | 500 | 100.0% | 480 | 1,658,665 | Approved |
| `nodejs_24_18_0_source_docs` | 55,425,872 | 500 | 500 | 100.0% | 469 | 2,727,507 | Conditional |

Go and Node.js published SHA-256 checks passed. Kubernetes exact-commit identity and local archive SHA-256 passed.
Dynamic API responses and the GovInfo service response have local SHA-256 records but no published checksum.
No redirect left an authorized domain. No unofficial mirror was used.
