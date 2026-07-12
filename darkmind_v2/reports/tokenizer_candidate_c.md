# Tokenizer Candidate C

Hard gates: **PASS**

- Algorithm: sentencepiece_unigram
- Vocabulary size: 16000
- Byte fallback pieces: 256 / 256
- Unknown tokens: 0
- Round-trip failures: 0
- Mojibake / replacement / malformed tokens: 0 / 0 / 0

## Efficiency

- Turkish tokens/character: 0.270272
- English tokens/character: 0.267679
- Technical/code tokens/character: 0.354889
- Tokens/word: 1.785604
- Turkish suffix fragmentation: 3.900000
- English word fragmentation: 1.973333
- Code/operator fragmentation: 1.024390

## Sequence Lengths

- p50 / p90 / p95 / p99 / max: 15 / 47 / 87 / 514 / 5932

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

- `tokenizer.model`: `1e45678a9e63bb45f8847a05a376fdea5918dd78f2f8c737c142ee7560f7e931`
- `tokenizer.vocab`: `5f37d63e9033514c7606327313dd1959863d6c9fec7c9e255256727a59c3073b`
- `training_config.json`: `75a54b9a3c814810623fb30558c33ee616a44cea76f0b198fce88d2109820705`
- `training_log.txt`: `6e1df6c5ea33dcfd0d8d8df842422e8345153a37b096d2b880ae901d0aacacd6`
- `tokenizer_manifest.json`: `47e1e00e6a7929feaedf00b27db22fc8b17ce4fe408d324b8f09330c2df34138`
