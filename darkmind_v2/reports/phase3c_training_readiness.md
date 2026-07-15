# Phase 3C Training Readiness

Status: **NOT READY**

Raw archive integrity: PASS
Verified raw archive bytes: 6,680,956
Streamed documents: 11,337
Accepted documents: 5,396
Available validated unique tokens: 2,907,272
Exact duplicate removals: 18
Near-duplicate removals: 3
Phase 1B paragraph overlap removals: 25,210
Evaluation contamination removals: 0
Source quota gate: FAIL
Frozen tokenizer provenance: PASS
Tokenizer model SHA-256: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
Tokenizer vocabulary SHA-256: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`
Tokenizer freeze-manifest SHA-256: `8e452c049f05ef1c6a94cb5fb42b6accdd1c18b76edebdb9d68bd85fbdfe538e`
Final tokenization: NOT RUN
Two-pass determinism: NOT RUN

| Source | Raw documents | Accepted documents | Unique tokens | Target | Shortfall to minimum |
|---|---:|---:|---:|---:|---:|
| python_docs_tr_3_14_6 | 5,675 | 191 | 61,172 | 5,000,000 | 4,838,828 |
| python_docs_en_3_14_6 | 5,662 | 5,205 | 2,846,100 | 10,000,000 | 6,953,900 |

No Corpus V3 tranche was accepted. The remaining gap is therefore 500,000,000 tokens to 500M and 1,000,000,000 tokens to 1B.

The 100M-token corpus was not built, and production training did not start.

Python documentation publishes no official archive checksum manifest. The downloaded archives passed the frozen byte, Last-Modified, ETag, and local SHA-256 controls.

The immutable source plan requires correction. No approved source can silently replace these technical quotas; a revised config and new user approval are required.

Approved correction candidates in the frozen registry are `wikimedia_trwikibooks_20260701` (10M cap), `wikimedia_enwikiversity_20260701` (25M cap), and `wikimedia_enwikibooks_20260701` (15M cap). `wikimedia_simplewiki_20260701` is an approved 25M English educational option but does not solve the Turkish gap.

Wikimedia source archives were not downloaded after the independent Python source gate failed. Their official checksum manifests were verified, but archive checksums were not claimed as completed downloads.

The 5M production Stage-1 experiment is not ready for approval.

DARKMIND V2 CORPUS V3 FIRST TRANCHE REQUIRES CORRECTION BEFORE TRAINING
