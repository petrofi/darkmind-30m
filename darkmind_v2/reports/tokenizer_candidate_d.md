# Tokenizer Candidate D

Hard gates: **PASS**

- Algorithm: sentencepiece_bpe
- Vocabulary size: 24000
- Byte fallback pieces: 256 / 256
- Unknown tokens: 0
- Round-trip failures: 0
- Mojibake / replacement / malformed tokens: 0 / 0 / 0

## Efficiency

- Turkish tokens/character: 0.255729
- English tokens/character: 0.251691
- Technical/code tokens/character: 0.318688
- Tokens/word: 1.685302
- Turkish suffix fragmentation: 3.850000
- English word fragmentation: 1.781333
- Code/operator fragmentation: 1.024390

## Sequence Lengths

- p50 / p90 / p95 / p99 / max: 14 / 44 / 81 / 484 / 5737

## Parameter Cost

| Dimension | Tied | Parameters | % 45M | % 60M |
| ---: | --- | ---: | ---: | ---: |
| 384 | True | 9216000 | 20.48 | 15.36 |
| 384 | False | 18432000 | 40.96 | 30.72 |
| 512 | True | 12288000 | 27.3067 | 20.48 |
| 512 | False | 24576000 | 54.6133 | 40.96 |

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

- `tokenizer.model`: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
- `tokenizer.vocab`: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`
- `training_config.json`: `a050dfe83ef0bf3a4cafeddf3287762539b43521d3b2430a162104c79b75ba55`
- `training_log.txt`: `146e4ecac7f135b72b61b03763005b842bdf68898e3928e0ad1d3dacf1119a46`
- `tokenizer_manifest.json`: `ee09b79e988fe3f14748873f874348574e2229ea92af178e2de63c46d86978d2`
