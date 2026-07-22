# Phase 5A Corpus V4 gap allocation

## Measured rationale

Base V1 still has high repetition and exact-loop rates, low EOS completion, weak punctuation closure, and poor factual/technical continuation. English prompt fit is better than Turkish in the new suite, but English remains underrepresented in Corpus V3 and English technical probes improved only 0.09% from 75M to the endpoint. The allocation therefore increases English, technical, structured/code, bilingual, and non-Wikimedia source diversity without displacing Turkish as the primary natural language.

## Current and proposed tokens

Corpus V3 contains 100,002,089 unique tokens: 60,000,490 Turkish and 40,001,599 English. Its broad category accounting is 85,002,064 general prose and 15,000,025 technical/educational tokens.

| Corpus V4 Tranche 2 category | New unique tokens | Tranche share | Allowed range | Gap addressed |
| --- | ---: | ---: | ---: | --- |
| Turkish general and educational | 72,000,000 | 36% | 30-40% | Turkish grammar, educational continuity, non-Wikimedia prose diversity |
| English general and educational | 50,000,000 | 25% | 20-30% | English representation and broader prose continuation |
| Turkish and English technical documentation | 42,000,000 | 21% | 15-25% | weakest technical probes and semantic structure |
| Open-licensed code and structured text | 26,000,000 | 13% | 8-15% | code/structured validity and delimiter completion |
| Controlled bilingual/dialogue material | 10,000,000 | 5% | 3-7% | language-switch resistance and aligned bilingual context |
| **Total** | **200,000,000** | **100%** | | |

Categories are exclusive at allocation time. A document receives one primary category, so these totals do not double count multilingual or technical records.

The language projection for the new tranche is 99M Turkish-attributable tokens, 75M English-attributable tokens, and 26M language-neutral code tokens. Projected cumulative tokens are 300,002,089. Exact cumulative language/category totals will be recomputed after deduplication because code-neutral and bilingual records cannot be honestly assigned twice.

## Capacity and remaining gaps

- Gap to the approximately 300M cumulative stage: 200,000,000 validated unique tokens.
- Gap from that stage to 500M: 199,997,911 additional unique tokens.
- Unconditionally planning-approved candidate capacity: 52,000,000 tokens.
- Conditional candidate capacity: 286,000,000 tokens.
- Approved plus conditional planning capacity: 338,000,000 tokens.
- Deferred capacity: 22,000,000 tokens; not counted toward feasibility.

The 200M plan is feasible only if at least 148M conditional tokens pass item-level legal, attribution, quality, PII, contamination, and deduplication gates. If they do not, the tranche must be reduced or new clearly licensed candidates must be proposed. Quotas must not override rights or quality evidence.

## Source diversity

The plan requires at least eight source families, no single source above 30M tokens or 15% of the new tranche, and staged admission from government/institutional prose, open education, official technical documentation, licensed books, code repositories, scientific literature, and controlled bilingual records. Wikimedia additions are disabled by default.

## Storage projection

The full approved-plus-conditional registry advertises 899.5GB of possible raw inputs, so blind acquisition would be wasteful. Acquisition must be staged and source-capped. For the accepted 200M-token tranche, uint16 token IDs alone require about 400MB; normalized text is likely several GB, while raw archives, extracted copies, deduplication indexes, deterministic rebuilds, and evidence can multiply this substantially. Reserve at least 1TB for candidate-source work or use a 2TB external SSD if raw archives and deterministic rebuild copies must coexist. This is a planning estimate, not a download authorization.
