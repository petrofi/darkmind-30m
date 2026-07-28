# Phase 5D Sample Capacity Model

| Source | Coverage | Optimistic | Expected | Conservative | Extraction loss | Quality loss | PII/secret loss | Internal dedup loss | V3 overlap loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `govuk_content_ogl3_20260722` | 100.00% | 5,620 | 5,058 | 4,215 | 0.00% | 27.12% | 0.00% | 4.28% | 0.00% |
| `govinfo_federal_register_2025_xml` | 0.00% | 0 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `plos_ccby_jats_allowlist` | 100.00% | 479 | 431 | 359 | 80.00% | 66.31% | 0.00% | 0.00% | 0.00% |
| `go_1_26_5_source_docs` | 8.94% | 10,000,000 | 7,500,000 | 4,500,000 | 0.00% | 3.39% | 1.97% | 1.17% | 0.00% |
| `kubernetes_website_f2987ba` | 22.04% | 7,527,021 | 5,645,265 | 4,233,948 | 0.00% | 0.03% | 2.22% | 0.00% | 0.00% |
| `nodejs_24_18_0_source_docs` | 55.49% | 4,914,967 | 3,686,225 | 2,764,668 | 0.00% | 4.36% | 6.11% | 0.02% | 0.00% |

Portal-wide extrapolation is prohibited. GOV.UK and PLOS capacities cover only the exact bounded API responses. GovInfo has zero evidenced capacity because no XML artifact was obtained.
Only Approved source capacity enters the lock: expected 23,145,265; conservative 15,033,948.
The uncertainty bands remain in `corpus_v4_sample_inspection_results.json`.
