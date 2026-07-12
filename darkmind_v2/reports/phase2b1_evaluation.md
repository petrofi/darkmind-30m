# Phase 2B.1 Evaluation And Acceptance

## Decision

Pipeline-integrity acceptance: **PASS**. Model-quality/public-release acceptance: **NOT CLAIMED**.

## Hard Gates

| Gate | Result |
| --- | --- |
| Exactly 256 optimizer steps | PASS |
| Exactly 1,048,576 consumed tokens | PASS |
| Finite losses and gradients | PASS |
| Full validation improved from 10.123761 to 7.096028 | PASS |
| Initial/midpoint/final checkpoint reload | PASS |
| Actual midpoint resume state | PASS |
| Frozen tokenizer hashes unchanged | PASS |
| Tokenized manifest and all eight shard hashes unchanged | PASS |
| Token IDs within `[0, 23999]` | PASS |
| Normal vocabulary U+FFFD/mojibake | PASS: none found |
| Tokenizer artifact corruption | PASS: none found |
| Decoder defect | PASS: none found |
| Checkpoint corruption/config/provenance mismatch | PASS: none found |
| Generation hard failures across five checkpoints | PASS: zero |
| Local Hugging Face offline reload | PASS |

The replacement characters observed at initialization were traced exclusively to sampled token `155` (`<0x93>`), an invalid one-byte UTF-8 sequence. Under the Stage-1 policy this is an explicit quality warning. Under public-release policy it remains a hard blocker.

## Remaining Risks

- Greedy generation remains highly repetitive, including 26 of 48 final-checkpoint prompts.
- Text remains incoherent and is not conversationally useful.
- Unexpected/mixed scripts occurred at initialization.
- Initial seeded sampling produced invalid byte sequences in all 48 prompts under the recorded seed.
- Only about 8.9% of one train epoch was consumed.
- The checkpoint is not instruction-tuned and is not production-ready.

Authoritative machine-readable evaluation: `darkmind_v2/data/phase2a/runs/tiny_stage1_seed20260712_r2/evaluations/byte_trace_policy_v2/stage1_evaluation.json`.
