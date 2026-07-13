# Phase 3A Corpus Source Gap

Status: **SOURCE RESEARCH ONLY; PROPOSED CONTRIBUTIONS ARE NOT DOWNLOADED TOKENS**

## Current Validated Baseline

The existing Phase 1B corpus contains 13,050,214 total tokens and approximately 11.74M train tokens. The measured tokenizer plan places lexical tokens at approximately 60.38% Turkish and 39.62% English, implying about 7.88M Turkish and 5.17M English tokens in the current total. These language values are estimates; the immutable Phase 2 tokenized manifest remains authoritative.

| Planning level | Existing tokens | Gap |
|---:|---:|---:|
| 100M | 13,050,214 | 86,949,786 |
| 250M | 13,050,214 | 236,949,786 |
| 500M | 13,050,214 | 486,949,786 |

The existing corpus has about 9.26M technical-documentation characters, approximately 2.4M tokens at the corpus-wide ratio, and no dedicated code quota. Against the default allocation, the conservative minimum gaps are therefore roughly 292.1M Turkish tokens, 119.8M English tokens, 37.6M technical tokens, and 20M code tokens. Existing categories must be reclassified before any exact quota credit is claimed.

## Candidate Registry

| Status | Sources | Expected usable tokens | Meaning |
|---|---:|---:|---|
| Approved proposal | 10 | 390M | License/source posture passed planning review; still not downloaded |
| Deferred | 7 | 210M potential | Legal, attribution, snapshot, privacy, or extraction blocker remains |
| Rejected | 3 | 0 | Not eligible for Phase 3 |
| Total | 20 | 600M proposed/potential | Potential values are not additive guarantees |

Approved-source download estimates total 31.119 GB decimal. Their nominal caps provide about 210M Turkish/controlled bilingual tokens and 180M English/controlled bilingual tokens. By exclusive content category they provide approximately 200M Turkish prose, 165M English prose, 15M technical documentation, 10M bilingual text, and 0 dedicated code tokens.

After applying the exact 500M composition rather than summing every nominal source cap, the approved proposal can credit at most about 350M tokens: 200M Turkish prose, 125M English prose, 15M technical, 0 code, and 10M bilingual. The composition-compatible gap is therefore about 150M tokens even though the raw approved cap sum is 390M.

## Blocking Gaps After Approved Proposals

- Turkish prose: about 100M tokens short.
- Technical/educational: about 25M tokens short.
- Code/structured text: 20M tokens short.
- Controlled bilingual/dialogue: about 5M tokens short.
- English prose: no volume gap, but source concentration and attribution diversity remain concerns.

Deferred Turkish Wikisource, Resmi Gazete, and Mevzuat candidates could nominally add 120M Turkish tokens, but none may be counted until work-level or official bulk-reuse rights are resolved. Deferred MDN and Stack Exchange could add 40M technical/code tokens, but mixed licenses, revision-level attribution, privacy filtering, and code/prose separation are unresolved. At least one additional license-clear Turkish technical or educational source is likely required.

## Storage and Yield Risk

- Expected raw-download budget: 35-60 GB for a targeted approved build; the current approved candidate list alone estimates 31.119 GB.
- Expected processed text/metadata: 3-5 GiB at 500M tokens.
- Expected uint16 shard storage: 0.931 GiB raw, approximately 1.0 GiB with indexes and manifests.
- Likely quality/dedup loss: 20-35% from raw extracted token estimates.
- Concentration risk: Turkish Wikipedia reaches the 35% hard source cap at 175M tokens and cannot close the Turkish gap alone.
- English overcapacity must not compensate for a Turkish or category shortfall.

## Decision

The 500M target is feasible only conditionally. The architecture plan can proceed to user review, but corpus acquisition and production-base training must remain blocked until at least 100M additional Turkish prose tokens, 25M technical tokens, 20M code tokens, and 5M controlled bilingual tokens have approved, attributable source plans with realistic post-dedup yield.
