# DarkMind v2 Corpus Validation

This directory contains corpus hygiene tools for Phase 0.

The tools validate local text only. They do not download corpora and do not delete source data.

Core checks:

- strict UTF-8 decoding by default,
- Unicode NFC normalization,
- null byte and unsafe control-character removal,
- mojibake and replacement-character detection,
- Turkish/English language checks,
- exact and near-duplicate detection,
- source and license metadata completeness,
- deterministic SHA-256 manifests.

Automatic encoding repair is disabled by default. When repair mode is explicitly enabled, every change must be reported with a line number and reason.

