# Phase 5A learning-curve review

## Scope

This review compares initialization through the exact no-wrap Corpus V3 endpoint. The final checkpoint is `step_011972_tokens_098074624`; no second epoch or continuation was run.

| Training tokens | Validation loss | Eval loss | Eval perplexity |
| ---: | ---: | ---: | ---: |
| 0 | 10.246471 | 10.243818 | 28,108.240 |
| 4,997,120 | 6.356269 | 6.306323 | 548.026 |
| 24,993,792 | 5.526496 | 5.473539 | 238.302 |
| 49,995,776 | 5.204454 | 5.156582 | 173.570 |
| 74,997,760 | 5.091754 | 5.048094 | 155.725 |
| 84,992,000 | 5.071706 | 5.028681 | 152.731 |
| 89,997,312 | 5.064458 | 5.022116 | 151.732 |
| 94,994,432 | 5.059540 | 5.016205 | 150.838 |
| 98,074,624 | 5.057351 | 5.013796 | 150.475 |

## Marginal improvement

Positive values below are loss reductions normalized to 10M training tokens.

| Interval | Validation reduction / 10M | Eval reduction / 10M |
| --- | ---: | ---: |
| initialization to 5M | 7.7849 | 7.8795 |
| 5M to 25M | 0.4150 | 0.4165 |
| 25M to 50M | 0.1288 | 0.1268 |
| 50M to 75M | 0.0451 | 0.0434 |
| 75M to 85M | 0.0201 | 0.0194 |
| 85M to 90M | 0.0145 | 0.0131 |
| 90M to 95M | 0.0098 | 0.0118 |
| 95M to 98.074M | 0.0071 | 0.0078 |

Loss continued to improve, with no validation or eval rebound, but marginal gains fell sharply after 75M. This is evidence of slowing on Corpus V3, not proof that the 118M-parameter architecture is capacity-saturated.

## Category probes

From 75M to the final checkpoint, loss improved by approximately 0.763% for English prose, 0.090% for English technical, 1.123% for Turkish prose, and 0.355% for Turkish technical. Several source-specific probes were flat or slightly negative. English technical behavior is the clearest weak-learning region; Turkish prose retained the strongest late relative gain.

The Phase 5A prompt-perplexity probe also showed a large language gap: English mean loss 6.0697 and perplexity 432.5, versus Turkish mean loss 7.0480 and perplexity 1,150.6. These prompts are original and not a standardized benchmark, so the result is directional. Factual-context and Turkish-technical categories had the highest prompt perplexities, while code-form prompts had low token loss but only a 65-67.5% automatic structure proxy. Low code-form perplexity does not establish semantic code usefulness.

## Generation trend

The directly comparable Phase 4D-to-final audit improved greedy repetition from 50.0% to 45.5%, sampling repetition from 19.8% to 17.0%, sampling loops from 16.8% to 15.8%, and EOS from 11.0% to 16.6% for sampling. Greedy exact loops worsened from 29.0% to 36.0%, although greedy EOS improved from 4.0% to 8.0%.

The new Phase 5A suite uses different prompts and a 48-token generation limit, so its 60.0% greedy repetition, 56.8% greedy loops, 2.0% greedy EOS, 33.2% sampling repetition, 33.2% sampling loops, and 8.4% sampling EOS must not be treated as a continuation of the older series. It independently confirms that repetition, loops, and weak completion remain material.

## Optimization health

- Pre-clip gradient norm: p50 `1.9453`, p95 `2.2188`, maximum `3.1094`.
- Clipped-step fraction: `1.0`; clipping remained effectively continuous.
- Clipping coefficient: p50 `0.5141`, p95 `0.5590`, minimum `0.3216`.
- Update-to-weight ratio: p50 `0.0001776`, p95 `0.0002312`, maximum `0.0002662`.
- Applied LR moved from about `4.041e-5` at 75M to `3.388e-5` at 85M, `3.174e-5` at 90M, `3.044e-5` at 95M, and `3.0065e-5` at the final step.

The run remained numerically stable and productive, but near-universal clipping and the decayed LR justify a controlled continuation-policy comparison on new data. They do not justify an automatic LR reset.

## Conclusion

An identical second Corpus V3 epoch is not recommended. It would repeat the same narrow source mixture while late marginal gains are already small, and it would increase duplication and memorization pressure. New, legally cleared, unique data is more valuable.

The largest deficits are semantic continuation, factual and technical continuity, Turkish technical prose, repetition control, loop avoidance, EOS behavior, punctuation/paragraph closure, and source diversity. Base V1 has not shown sufficient evidence of architecture saturation. The next experiment should keep the architecture frozen, expand Corpus V4, and compare P1/P2 on one identical small new-data slice before authorizing a longer continuation.
