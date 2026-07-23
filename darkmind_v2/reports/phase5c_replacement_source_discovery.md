# Phase 5C Replacement Source Discovery

Official metadata was reviewed without acquiring any dataset or corpus file.

| Source | Category | Exact artifact/snapshot | Status | Expected / conservative tokens | Decision |
|---|---|---|---|---:|---|
| `mozilla_common_voice_tr_26` | ["turkish_general_educational"] | cv-corpus-26.0-2026-06-12; 2.78 GB; 413,915 sentences | rejected | 0 / 0 | Rejected; no next action. |
| `leipzig_turkish_corpora_official_route` | ["turkish_general_educational"] | official Turkish route returned no artifact on 2026-07-22 | rejected | 0 / 0 | Rejected; no next action. |
| `govinfo_federal_register_2025_xml` | ["english_general_educational", "technical_documentation"] | Federal Register 2025 XML bulk directory; exact file manifest unresolved | conditional | 30,000,000 / 18,000,000 | Pin exact 2025 file inventory, checksums, notices, and reusable text fields. |
| `bccampus_ccby_textbook_allowlist` | ["english_general_educational"] | exact edition allowlist unresolved | conditional | 25,000,000 / 15,000,000 | Pin exact CC BY editions and remove third-party media and exceptions. |
| `plos_ccby_jats_allowlist` | ["technical_documentation"] | exact article and licence allowlist unresolved | conditional | 35,000,000 / 20,000,000 | Pin exact CC BY article IDs, JATS checksums, corrections, and exclusions. |
| `go_1_26_5_source_docs` | ["technical_documentation", "code_structured"] | go1.26.5.src.tar.gz | conditional | 5,000,000 / 3,000,000 | Verify archive checksum, path licences, generated files, and final filtered capacity. |
| `kubernetes_website_f2987ba` | ["technical_documentation", "code_structured"] | git commit f2987ba1cceaa85fcd44cd1a221010d745d7335c | conditional | 10,000,000 / 5,000,000 | Pin archive hash and exclude translations, generated files, vendored assets, and benchmark-like tutorials. |
| `nodejs_24_18_0_source_docs` | ["technical_documentation", "code_structured"] | node-v24.18.0.tar.xz | conditional | 12,000,000 / 6,000,000 | Resolve bundled third-party licences and select only eligible documentation/source paths. |

Turkish general prose remains the largest unresolved gap. Common Voice Turkish was rejected because the official record combines no-rehosting terms with heavy Turkish-Wikipedia prompt provenance. The Leipzig route was rejected because no exact Turkish artifact was available and source-rights evidence was inadequate.

DGT-TM was corrected to rejected: its official 24-language coverage does not include Turkish.
