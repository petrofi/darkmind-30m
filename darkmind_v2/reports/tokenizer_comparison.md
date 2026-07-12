# DarkMind v2 Phase 1B Tokenizer Comparison

| Candidate | Algorithm | Vocab | Gates | TR t/c | EN t/c | Tech t/c | p95 | p99 | Score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D | sentencepiece_bpe | 24000 | PASS | 0.255729 | 0.251691 | 0.318688 | 81 | 484 | 90.0000 |
| B | sentencepiece_bpe | 16000 | PASS | 0.273507 | 0.264554 | 0.341584 | 86 | 523 | 53.9126 |
| C | sentencepiece_unigram | 16000 | PASS | 0.270272 | 0.267679 | 0.354889 | 87 | 514 | 51.2455 |
| A | sentencepiece_bpe | 12000 | PASS | 0.288833 | 0.277066 | 0.361386 | 91 | 552 | 20.0000 |

## Recommendation

- Recommended candidate: D
- Recommendation strength: strong
- Reason: The leading eligible candidate was more than 2 points ahead.
- This recommendation does not freeze a final tokenizer.

## 24k Cost Discussion

Candidate D uses 9.216M parameters at 384-dim tied (20.48% of 45M) and 18.432M untied (40.96% of 45M). At 512-dim it uses 12.288M tied (27.31% of 45M) or 24.576M untied (54.61% of 45M), so compression gains must be substantial to justify the model-capacity cost.
