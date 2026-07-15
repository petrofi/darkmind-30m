# Phase 3B Scaling Target Rationale

The frozen base has 118,056,960 parameters. The minimum serious unique clean corpus target is 500M tokens; 1B is a preferred stretch target only if licensing, attribution, quality, and acquisition remain practical.

| Seen tokens | Tokens / parameter | Active training | Conservative wall clock |
|---:|---:|---:|---:|
| 500M | 4.235 | 9.55 h | 13.18 h |
| 1B | 8.470 | 19.10 h | 26.36 h |
| 2B | 16.941 | 38.20 h | 52.72 h |

The projection uses Candidate D's 1,000-step measured 14,542.1 token/s and a 1.38 wall-clock multiplier for evaluation, checkpoint, interruption, and thermal allowance. It is a budget estimate, not a runtime guarantee.

Stages are 5M, 25M, 100M, 250M, 500M, optional 1B, and optional 2B. Every transition requires validation/eval and generation-health review, checkpoint integrity, memorization/source-exposure checks, and explicit user approval.

At 500M, continuation is evidence-driven. Additional unique clean data is preferred over repetition. Repeated epochs require deterministic reshuffling plus source-overexposure and near-verbatim memorization checks. General scaling-law ratios cannot be treated as an exact prescription for a 118M-parameter Turkish-English model, and 2B tokens are neither mandatory nor sufficient by definition.
