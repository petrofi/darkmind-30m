# DarkMind v2 Tokenizer Experiments

This directory contains tokenizer validation tools and the Phase 1B candidate experiment workflow.

Use `train_tokenizer_candidates.py` only with the approved processed Phase 1B corpus. It trains the configured SentencePiece candidates under ignored data paths, verifies special-token IDs and round trips, and does not freeze a final tokenizer.

Use `compare_tokenizer_candidates.py` to audit the completed candidates against the fixed samples plus validation/eval splits, apply hard gates, calculate parameter costs, and generate the weighted comparison reports.

Use `build_tokenizer_manifest.py` on an already-existing tokenizer directory to record file hashes, special tokens, tokenizer type, vocabulary size, and immutable version metadata.

Use `audit_tokenizer.py` with a tokenizer directory and reviewed corpus sample to check round trips, unknown-token ratio, malformed vocabulary tokens, mojibake tokens, replacement characters, script distribution, Turkish coverage, English coverage, and sequence length percentiles.

Use `test_roundtrip.py` for exact encode/decode checks on fixed samples.

Candidate comparison does not authorize model pretraining, instruction tuning, final-tokenizer freezing, or teacher-data generation.

