# Phase 3C Download Inventory

Status: **PASS_SELECTED_SOURCES**

Hard raw-download cap: 8,000,000,000 bytes
Frozen planned bytes: 1,496,788,787 bytes
Verified bytes in this acquisition run: 6,779,020 bytes
Full plan complete: False
Available disk at preflight: 290,023,751,680 bytes

| Source | Snapshot | File | Expected bytes | Checksum policy | Target tokens | Status |
|---|---:|---|---:|---|---:|---|
| wikimedia_trwiki_20260701 | 2026-07-01 | trwiki-20260701-pages-articles-multistream1.xml-p1p1500000.bz2 | 381,155,966 | official MD5 + SHA-1 | 55,000,000 | not_downloaded_source_quota_gate |
| wikimedia_trwiki_20260701 | 2026-07-01 | trwiki-20260701-pages-articles-multistream1.xml-p1500001p3000000.bz2 | 406,014,080 | official MD5 + SHA-1 | 55,000,000 | not_downloaded_source_quota_gate |
| wikimedia_enwiki_20260701 | 2026-07-01 | enwiki-20260701-pages-articles-multistream1.xml-p1p41242.bz2 | 298,405,218 | official MD5 + SHA-1 | 30,000,000 | not_downloaded_source_quota_gate |
| wikimedia_enwiki_20260701 | 2026-07-01 | enwiki-20260701-pages-articles-multistream2.xml-p41243p151573.bz2 | 404,434,503 | official MD5 + SHA-1 | 30,000,000 | not_downloaded_source_quota_gate |
| python_docs_tr_3_14_6 | 3.14.6-2026-07-09 | python-3.14-docs-text-tr-3.14.6.tar.bz2 | 3,372,576 | official checksum unavailable; byte + Last-Modified + ETag lock, local SHA-256 | 5,000,000 | reused_verified |
| python_docs_en_3_14_6 | 3.14.6-2026-07-09 | python-3.14-docs-text-en-3.14.6.tar.bz2 | 3,308,380 | official checksum unavailable; byte + Last-Modified + ETag lock, local SHA-256 | 10,000,000 | reused_verified |

The four official checksum manifests contribute 98,064 bytes to the verified acquisition total. Raw archives are preserved unchanged under the ignored Phase 3C runtime directory.

The diagnostic Turkish Wikipedia `.partial.single-connection` file is incomplete, remains ignored, was not accepted as an archive, and was not counted in verified bytes.

No unapproved source, undated `latest` endpoint, or silent checksum substitution is permitted.
