# Phase 5B Corpus V3 overlap plan

The approved source set is small enough to run strict, source-specific overlap controls before allocation. No overlap computation or source acquisition occurred in Phase 5B.

| Check | MDN pinned prose | Rust 1.90 docs/code | PostgreSQL 18.0 docs |
| --- | --- | --- | --- |
| Document ID | Pinned repository path plus MDN slug | Commit, path and item anchor | Version, SGML/XML path and section ID |
| URL | Canonical MDN URL against Corpus V3 provenance URLs | Canonical doc.rust-lang.org URL set | Canonical PostgreSQL 18 URL set |
| Exact hash | SHA-256 extracted page body | SHA-256 documentation/code unit | SHA-256 extracted section |
| Normalized fingerprint | BLAKE2b normalized prose block | BLAKE2b normalized prose/code | BLAKE2b normalized section |
| Near dedup | MinHash over normalized 5-grams | MinHash over normalized 5-grams | MinHash over normalized 5-grams |
| N-gram sample | Deterministic 13-gram sample against V3 shards | Deterministic 13-gram sample | Deterministic 13-gram sample |
| Wikimedia expectation | Low; remove any high-overlap excerpts | Low | Low |
| Python-doc expectation | Low-medium for Python/web examples | Medium for generic programming explanations; cap reduced | Low for generic SQL/DB-API text |
| Benchmark control | Deny tests and known benchmark phrase hashes | Exclude compiler tests, challenge solutions and benchmark hashes | Exclude regression inputs/outputs and benchmark examples |

## Exact execution order

1. Materialize source document IDs and canonical URLs from the checksum-verified snapshot.
2. Reject duplicate IDs and URLs within the source.
3. Hash normalized documents and reject exact duplicates within source, against previous V4 sources and against Corpus V3 fingerprints.
4. Compute deterministic MinHash signatures and cluster near duplicates across all three scopes.
5. Sample fixed 13-grams by document hash and compare with Corpus V3 token/text indexes.
6. Apply source-specific Python-doc and Wikimedia thresholds.
7. Screen benchmark manifests and known evaluation phrase hashes.
8. Preserve rejection reason, matching source/document ID and similarity score.

High-overlap documents are removed rather than reassigned. Python documentation remains deferred because its current snapshot is already in Corpus V3. DGT-Acquis is deferred partly because CELEX-level overlap with EUR-Lex is expected to dominate its small unique contribution.

The overlap artifact schema for every approved source is enforced by validate_source_registry_v4.py: document ID, URL, exact hash, normalized fingerprint, near dedup, n-gram sampling, Wikimedia, Python-doc and benchmark strategies must all be nonempty.
