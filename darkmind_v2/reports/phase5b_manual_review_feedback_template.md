# Phase 5B manual-review feedback template

Reference packet: C:\DarkMindRuntime\phase5a\manual_review\phase5a_manual_review_packet_v2.md.

Human scores were not invented and were not available when the Phase 5B category caps were recorded. Source due diligence may continue without them, but final category caps must record the completed review identity and resulting allocation changes before a fully locked plan.

## Minimum useful subset

| Stratum | Required completed reviews | Selection rule |
| --- | ---: | --- |
| Turkish outputs | 20 | Balanced across greedy/sampling and Turkish probe categories |
| English outputs | 15 | Balanced across greedy/sampling and general/educational categories |
| Technical outputs | 10 | Include factual, explanatory and structured technical prompts |
| Code or structured outputs | 5 | Include syntax, delimiter and explanatory-code probes |
| **Total** | **50** | No output counted twice |

## Completion record

| Field | Value |
| --- | --- |
| Reviewer identifier | |
| Review date | |
| Packet content hash | |
| Prompt/output IDs reviewed | |
| Missing or skipped items | |
| Inter-rater process, if any | |

## Aggregate findings

| Stratum | Coherence | Relevance | Fluency | Completion | Repetition | Safety | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turkish | | | | | | | |
| English | | | | | | | |
| Technical | | | | | | | |
| Code/structured | | | | | | | |

## Decision integration

- Weak Turkish coherence increases priority for diverse, high-quality Turkish educational prose; it does not justify lower license standards.
- Weak English relevance raises English general/educational priority within the fixed 200M target.
- Weak technical factual structure raises official technical-documentation priority and lowers low-quality forum caps.
- Weak code structure raises carefully licensed code/tutorial priority while preserving the 5% single-ecosystem limit.
- Strong repetition or poor completion keeps the continuation success gates strict.
- Human evidence informs SFT readiness but cannot make SFT appropriate before the base model and new-corpus continuation are evaluated.

Record each changed category target, old value, new value, human finding, automatic metric and reviewer approval. A completed score must never be inferred from an empty cell.
