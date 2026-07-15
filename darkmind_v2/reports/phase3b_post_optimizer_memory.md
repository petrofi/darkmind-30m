# Phase 3B Post-Optimizer Memory Audit

Both finalists passed 10 real AdamW optimizer steps in clean isolated processes. Optimizer state was materialized before measurement.

| Candidate | Model weights | Gradients | Optimizer states | Activation/temp peak | Peak allocated | Peak reserved | Reserved headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 207,762,432 | 207,762,432 | 415,525,456 | 531,079,600 | 1,362,129,920 | 1,440,743,424 | 83.22% |
| D | 236,113,920 | 236,113,920 | 472,228,528 | 568,419,664 | 1,512,876,032 | 1,589,641,216 | 81.48% |

Profile: BF16, sequence length 512, micro-batch 2, SDPA, gradient checkpointing disabled. Both exceed the 15% safe-headroom gate; the architecture decision uses these post-warmup figures rather than initialization-only memory.
