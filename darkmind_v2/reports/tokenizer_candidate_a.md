# Tokenizer Candidate A

Hard gates: **PASS**

- Algorithm: sentencepiece_bpe
- Vocabulary size: 12000
- Byte fallback pieces: 256 / 256
- Unknown tokens: 0
- Round-trip failures: 0
- Mojibake / replacement / malformed tokens: 0 / 0 / 0

## Efficiency

- Turkish tokens/character: 0.288833
- English tokens/character: 0.277066
- Technical/code tokens/character: 0.361386
- Tokens/word: 1.884602
- Turkish suffix fragmentation: 4.350000
- English word fragmentation: 1.972000
- Code/operator fragmentation: 1.024390

## Sequence Lengths

- p50 / p90 / p95 / p99 / max: 15 / 49 / 91 / 552 / 6182

## Parameter Cost

| Dimension | Tied | Parameters | % 45M | % 60M |
| ---: | --- | ---: | ---: | ---: |
| 384 | True | 4608000 | 10.24 | 7.68 |
| 384 | False | 9216000 | 20.48 | 15.36 |
| 512 | True | 6144000 | 13.6533 | 10.24 |
| 512 | False | 12288000 | 27.3067 | 20.48 |

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

- `tokenizer.model`: `d769a7950794e582303990bbacd3d684c179c923dc700975d76b2164db27cb69`
- `tokenizer.vocab`: `621d17c073f75a8d1abb6f84152752979a32112b49ed20c022a72000b50e9fe3`
- `training_config.json`: `37c21301e28afeba1a11c900204892986ae1533a4be878f43a39f085079c620c`
- `training_log.txt`: `2c90cbf30f31fa378305a6b9f0ac833bd78c43780bbc2d7e8ba94e55c4551ddb`
- `tokenizer_manifest.json`: `cb7de723b1ee54a2485e7d79f143df04ad9edb90deb2eee118d565e82773c58a`
