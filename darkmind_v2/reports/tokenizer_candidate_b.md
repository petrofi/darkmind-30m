# Tokenizer Candidate B

Hard gates: **PASS**

- Algorithm: sentencepiece_bpe
- Vocabulary size: 16000
- Byte fallback pieces: 256 / 256
- Unknown tokens: 0
- Round-trip failures: 0
- Mojibake / replacement / malformed tokens: 0 / 0 / 0

## Efficiency

- Turkish tokens/character: 0.273507
- English tokens/character: 0.264554
- Technical/code tokens/character: 0.341584
- Tokens/word: 1.790331
- Turkish suffix fragmentation: 4.150000
- English word fragmentation: 1.896000
- Code/operator fragmentation: 1.024390

## Sequence Lengths

- p50 / p90 / p95 / p99 / max: 15 / 47 / 86 / 523 / 5968

## Parameter Cost

| Dimension | Tied | Parameters | % 45M | % 60M |
| ---: | --- | ---: | ---: | ---: |
| 384 | True | 6144000 | 13.6533 | 10.24 |
| 384 | False | 12288000 | 27.3067 | 20.48 |
| 512 | True | 8192000 | 18.2044 | 13.6533 |
| 512 | False | 16384000 | 36.4089 | 27.3067 |

## Hard Gates

- processed_corpus_gates: PASS
- special_token_mismatch: PASS
- roundtrip_failure: PASS
- hostile_fixture_leak: PASS
- manifest_nondeterminism: PASS
- manifest_hash_mismatch: PASS
- byte_fallback: PASS
- unknown_token_ratio: PASS
- turkish_efficiency: PASS
- english_efficiency: PASS
- vocabulary_cleanliness: PASS

## Hashes

- `tokenizer.model`: `abd86cbbf884b06ed9ed612f660f27e7d57c49165199c756dd0a26927afa0430`
- `tokenizer.vocab`: `87348015aac5bad490dbc36a075edaae343e0608b6f515c2484a2e9454937c8f`
- `training_config.json`: `02ee617e6bff0d235dd798b79d4a202437070430c7982f09f5b59882e1863c8a`
- `training_log.txt`: `bebd513f40a430fe33754205fef6c8768b52f0926eaffae32dc1926e33a9925e`
- `tokenizer_manifest.json`: `2ea4f3fe22f20bb60137a61e27d8d2d1ab030e3e7b34a20965bb4dc1734bef76`
