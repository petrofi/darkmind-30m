# Phase 3C.1 Training Readiness

Status: **READY FOR APPROVAL**

Final documents: 447,127
Final tokens: 100,002,089
Turkish / English tokens: 60,000,490 / 40,001,599
Prose / technical tokens: 85,002,064 / 15,000,025

Phase 1B seed: 13,050,214 tokens
New unique Python: 2,907,272 tokens
Turkish Wikibooks: 894,388 tokens
English Wikiversity: 3,632,011 tokens
English Wikibooks: 3,619,974 tokens
Turkish Wikipedia: 51,191,686 tokens
English Wikipedia: 24,706,544 tokens

Train: 438,255 documents / 98,082,120 tokens
Validation: 4,441 documents / 937,640 tokens
Eval: 4,431 documents / 982,329 tokens
Shards: 22 / 200,004,178 bytes
Rejected records retained: 182,828
Exact / near duplicate removals: 330 / 4,414
Evaluation contamination removals / accepted: 6 / 0
Missing licenses / attribution: 0 / 0
Two-pass compared files / mismatches: 62 / 0
Remaining gap to 500M: 399,997,911 tokens
Remaining gap to 1B: 899,997,911 tokens

## Core provenance hashes

Corpus manifest: `e75c4aa4f39cc7a3cb4fe754e2a0e85268ced300f8504a86d443540eb609e1c5`
Documents: `f4055eb351578937563623d3e13271417800bc7226d92f291c76befdac51cd96`
Attribution manifest: `e1effb74dccf1e121eba134b1f5dfedc5133f1d78c80b45422a28b98042d4e6d`
Split manifest: `c7448c3c17ab4b56840216363b3e8268f7557a9f3129d56cb5d1b2190c56ea39`
Tokenized manifest: `1296caacf09d49b1c48c0fee7d5f5a523a0019e8e7e0e70132fbf68d8f023c82`
Boundaries: `3daae663c766575cd7526487baf713d6cfd83fdce5767cfea04722848418cdd8`
Shard checksums: `997ec0fe50a398e7cde90169e0c3ac94b55a8107b305444b08dbe0086148ee76`
Tokenizer model: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
Tokenizer vocab: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`
Tokenizer freeze manifest: `8e452c049f05ef1c6a94cb5fb42b6accdd1c18b76edebdb9d68bd85fbdfe538e`
Base V1 config: `8e9775721b0173a92e88de15c2195428932b3aa5beec57d568674c25887c5e39`
Seed documents: `4873305d396293a5cc846922c290b1878f635683ba32b735043813a5fff3bef2`
New Python documents: `8f317d353723c5c0e827cff5182bbe0d3bb23cb5f4235ae248f94fd47e3df76c`
Technical selection: `f6761e7cb06b0a2cea9aaae1eb0582bf520d978f550c6cf416c03c109d22f622`

## Verified archives

| Source | File | Bytes | SHA-256 |
|---|---|---:|---|
| wikimedia_enwiki_20260701 | enwiki-20260701-pages-articles-multistream1.xml-p1p41242.bz2 | 298,405,218 | `a7a10f37f1dda60f146606d86ed7cc8c9ca0cc57a274e93a4a90e973b3109124` |
| wikimedia_enwiki_20260701 | enwiki-20260701-pages-articles-multistream2.xml-p41243p151573.bz2 | 404,434,503 | `25af61677f94d390449f9f931213a661afa59fee9728b1b2eae3754fc4eee181` |
| wikimedia_enwikibooks_20260701 | enwikibooks-20260701-pages-articles.xml.bz2 | 202,864,457 | `09102ff8b21ef594a0a4046e33384df7d07310212da80ccc5f7a69ee9aaf7041` |
| wikimedia_enwikiversity_20260201 | enwikiversity-20260201-pages-articles.xml.bz2 | 112,905,404 | `ec89f3fe79aaf8f7622bfbced1f2ed2b4e6f2cd0b6dfde33338dec825a7646d8` |
| wikimedia_trwiki_20260701 | trwiki-20260701-pages-articles-multistream1.xml-p1500001p3000000.bz2 | 406,014,080 | `add0751147ac97ec20cbd29fd4b7c76c39ecf381d17d1c45b910cc36b537c1b8` |
| wikimedia_trwiki_20260701 | trwiki-20260701-pages-articles-multistream1.xml-p1p1500000.bz2 | 381,155,966 | `cb3cb6125c16a37f888d3ca0cbc6362d77f80052bc1d2be25cc43cd6254cf53b` |
| wikimedia_trwikibooks_20260701 | trwikibooks-20260701-pages-articles.xml.bz2 | 3,029,122 | `c126dcb88f10f36bfad6b498cd224d679c3f529b7a2d8a708170112a4c9575fe` |

## Hard gates

| Gate | Violations |
|---|---:|
| accepted_evaluation_contamination | 0 |
| aggregate_quota_violation | 0 |
| cross_split_duplicate_clusters | 0 |
| invalid_utf8_or_unicode_accepted | 0 |
| material_pii_accepted | 0 |
| missing_attribution | 0 |
| missing_licenses | 0 |
| official_checksum_failures | 0 |
| raw_cap_violation | 0 |
| source_cap_violation | 0 |
| tokenizer_mismatch | 0 |
| unapproved_source_count | 0 |
| unexplained_document_loss | 0 |
| unresolved_exact_duplicates | 0 |
| unresolved_near_duplicate_clusters | 0 |
| wrong_language_accepted | 0 |

Residual legal risk: Wikimedia and seed-source attribution/share-alike obligations remain attached to every reused record.
Residual quality risk: corpus gates establish data readiness, not downstream model coherence or public-release quality.

DARKMIND V2 CORPUS V3 AGGREGATE 100M IS READY FOR BASE V1 TRAINING APPROVAL
