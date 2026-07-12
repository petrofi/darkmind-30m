# Phase 2B.1 Checkpoint Comparison

## Evaluation Contract

All checkpoints used the complete validation and eval token shards, the same 48 fixed Turkish/English prompts, 16 continuation tokens, seed `20260712`, and identical greedy and seeded-sampling settings. `byte_trace_policy_v2` is the authoritative comparison directory. The earlier `byte_trace_policy_v1` directory was preserved after an external tool timeout and contains the diagnosis, integrity audit, and completed initial outputs.

| Checkpoint | Step | Tokens | Validation loss | Eval loss | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial | 0 | 0 | 10.123761 | 10.123414 | 24,928.340 |
| step 64 | 64 | 262,144 | 7.843530 | 7.836454 | 2,549.187 |
| midpoint | 128 | 524,288 | 7.391306 | 7.377266 | 1,621.823 |
| step 192 | 192 | 786,432 | 7.169022 | 7.154002 | 1,298.574 |
| final | 256 | 1,048,576 | **7.096028** | **7.081271** | **1,207.163** |

## Generation Findings

| Checkpoint | Greedy warnings | Seeded-sampling warnings | Hard failures |
| --- | --- | --- | ---: |
| initial | repetition 41; unexpected script 2; mixed script 2 | invalid UTF-8 bytes 48; byte-origin U+FFFD 48 | 0 |
| step 64 | repetition 45 | none | 0 |
| midpoint | repetition 44 | none | 0 |
| step 192 | repetition 47 | none | 0 |
| final | repetition 26 | none | 0 |

The initial sampling warning is preserved with raw token IDs and byte runs. It is not tokenizer corruption and was not sanitized. It counts against generation quality and remains public-release blocking even though it does not block this controlled Stage-1 pipeline-integrity decision.

## Selection

The deterministic selection rule is minimum full validation-shard loss, with the checkpoint label used only as a deterministic tie-breaker. The selected checkpoint is `step_000256_tokens_001048576`; it is also the final checkpoint. Its source model hash is `5e2fd69d4775940629926a7bf659e36beafe4c1cd544feb8d97beeae6537b097`.

Training loss was not used as the sole selection criterion.
