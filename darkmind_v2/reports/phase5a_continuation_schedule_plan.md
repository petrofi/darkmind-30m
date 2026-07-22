# Phase 5A continuation-schedule plan

## Boundary

The source checkpoint is `step_011972_tokens_098074624`, ending at applied LR `0.000030065052779045817`. Phase 5A authorizes no training, optimizer modification, LR reset, or second Corpus V3 epoch.

Any future comparison begins only after a legally cleared, deduplicated Corpus V4 slice exists. All arms use the same frozen model weights, 4,997,120 new unique tokens, 610 optimizer steps, 8,192 tokens per step, sequence order, evaluation probes, and random controls. Only the declared policy factor may vary.

## P1: continue at low LR

- Preserve the final Adam moments and resume state.
- Start at the current applied LR, approximately `3.0065e-5`.
- Do not rewarm.
- Extend the scheduler horizon for new unique data, with a planning minimum of `2e-5`.
- Treat P1 as the conservative baseline.

## P2: mild controlled rewarm

- Preserve the final Adam moments.
- Start at the current applied LR.
- Rewarm gradually for 64 steps to no more than `5e-5`.
- Follow with cosine decay over the approved new-data horizon, with a planning minimum of `2e-5`.
- No abrupt LR jump is permitted.

## P3: optimizer-state review

P3 is not approved by default. It is activated only if P1/P2 evidence on the same slice suggests stale Adam moments materially impair adaptation. The future experiment must compare a preserved-moment control with a carefully reset-moment arm while keeping model weights, LR path, data, sequence order, budget, and probes matched. An optimizer reset cannot be selected from intuition alone.

## Decision measurements

- Validation and eval loss on unchanged held-out data.
- Fixed-probe loss by source family and language/category.
- Gradient norm, clipped-step fraction, and update-to-weight distributions.
- Repetition, exact-loop, EOS, Unicode, and special-token health.
- Evidence of catastrophic regression or source over-adaptation.

P1/P2 should first be compared without extending either arm beyond the declared slice. Prefer the policy with stable optimization and broad held-out improvement; a small training-loss advantage alone is insufficient. Any longer continuation requires a new explicit authorization and a no-wrap data budget derived from Corpus V4.
