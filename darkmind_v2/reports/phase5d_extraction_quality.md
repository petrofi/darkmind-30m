# Phase 5D Extraction and Quality Inspection

| Source | Extraction success | Empty | Parse failure | Quality rejected | Boilerplate hits | Emails | Phone patterns | Credential patterns | Private keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `govuk_content_ogl3_20260722` | 100.0% | 0.0% | 0.0% | 91 | 0 | 0 | 0 | 0 | 0 |
| `govinfo_federal_register_2025_xml` | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| `plos_ccby_jats_allowlist` | 20.0% | 80.0% | 0.0% | 189 | 0 | 0 | 0 | 0 | 0 |
| `go_1_26_5_source_docs` | 100.0% | 0.0% | 0.0% | 4 | 599 | 19 | 857 | 3 | 0 |
| `kubernetes_website_f2987ba` | 100.0% | 0.0% | 0.0% | 15 | 1 | 27 | 2289 | 11 | 0 |
| `nodejs_24_18_0_source_docs` | 100.0% | 0.0% | 0.0% | 9 | 105 | 111 | 845 | 129 | 1 |

All inspected files were digitally extractable; OCR was not used. Archive adapters opened only the deterministic selected members and never extracted or executed packages.
Quality scoring covered sentence completeness, alphabetic and symbol balance, repetition, minimum length, and template density.
Code and prose were assigned mutually exclusively. Go was 99.9576% code by raw sample tokens; Kubernetes was 100% English technical documentation; Node.js was 52.7365% code and 47.2635% technical documentation.
GOV.UK supplied short English metadata/descriptions rather than full publication bodies. PLOS supplied sparse metadata/abstract fields, with 80% empty extraction and 189 quality rejections.
Telephone and credential-pattern counts are conservative pattern hits, including code-shaped false positives. Matching documents were excluded where policy required. No raw personal data or secret value is reproduced here.
