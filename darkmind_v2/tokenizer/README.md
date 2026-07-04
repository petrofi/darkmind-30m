# DarkMind v2 Tokenizer Validation

This directory contains tokenizer validation tools only. It must not train a tokenizer.

Use `build_tokenizer_manifest.py` on an already-existing tokenizer directory to record file hashes, special tokens, tokenizer type, vocabulary size, and immutable version metadata.

Use `audit_tokenizer.py` with a tokenizer directory and reviewed corpus sample to check round trips, unknown-token ratio, malformed vocabulary tokens, mojibake tokens, replacement characters, script distribution, Turkish coverage, English coverage, and sequence length percentiles.

Use `test_roundtrip.py` for exact encode/decode checks on fixed samples.

