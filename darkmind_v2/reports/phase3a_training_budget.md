# Phase 3A Training Budget

These projections use each candidate's measured safe MB2, SDPA, checkpointing-off throughput. For conservatism, sustained active throughput is 80% of the microbenchmark result, evaluation/checkpoint overhead is 15% of active time, and wall time adds a further 20% interruption/thermal contingency. These are planning numbers, not training authorization.

| Candidate | Measured tok/s | Budget tok/s | BF16 model checkpoint | Measured optimizer state | Full resume checkpoint estimate |
|---|---:|---:|---:|---:|---:|
| A | 17,915.5 | 14,332.4 | 124.16 MiB | 240.29 MiB | 364.45 MiB |
| B | 19,165.3 | 15,332.3 | 155.27 MiB | 304.01 MiB | 459.28 MiB |
| C | 17,162.1 | 13,729.7 | 201.15 MiB | 396.28 MiB | 597.43 MiB |
| D | 14,906.3 | 11,925.1 | 228.69 MiB | 450.35 MiB | 679.05 MiB |

## Stage Durations

| Candidate | Tokens | Active GPU | Eval/checkpoint overhead | Estimated wall | Checkpoint frequency | Evaluation frequency |
|---|---:|---:|---:|---:|---|---|
| A | 5M | 0.10h | 0.01h | 0.13h | 1M tokens | 1M tokens and stage boundary |
| A | 25M | 0.48h | 0.07h | 0.67h | 2.5M tokens | 2.5M tokens and stage boundary |
| A | 100M | 1.94h | 0.29h | 2.67h | 5M tokens | 10M tokens and stage boundary |
| A | 250M | 4.85h | 0.73h | 6.69h | 10M tokens | 25M tokens and stage boundary |
| A | 500M | 9.69h | 1.45h | 13.37h | 20M tokens | 25M tokens and final audit |
| B | 5M | 0.09h | 0.01h | 0.13h | 1M tokens | 1M tokens and stage boundary |
| B | 25M | 0.45h | 0.07h | 0.63h | 2.5M tokens | 2.5M tokens and stage boundary |
| B | 100M | 1.81h | 0.27h | 2.50h | 5M tokens | 10M tokens and stage boundary |
| B | 250M | 4.53h | 0.68h | 6.25h | 10M tokens | 25M tokens and stage boundary |
| B | 500M | 9.06h | 1.36h | 12.50h | 20M tokens | 25M tokens and final audit |
| C | 5M | 0.10h | 0.02h | 0.14h | 1M tokens | 1M tokens and stage boundary |
| C | 25M | 0.51h | 0.08h | 0.70h | 2.5M tokens | 2.5M tokens and stage boundary |
| C | 100M | 2.02h | 0.30h | 2.79h | 5M tokens | 10M tokens and stage boundary |
| C | 250M | 5.06h | 0.76h | 6.98h | 10M tokens | 25M tokens and stage boundary |
| C | 500M | 10.12h | 1.52h | 13.96h | 20M tokens | 25M tokens and final audit |
| D | 5M | 0.12h | 0.02h | 0.16h | 1M tokens | 1M tokens and stage boundary |
| D | 25M | 0.58h | 0.09h | 0.80h | 2.5M tokens | 2.5M tokens and stage boundary |
| D | 100M | 2.33h | 0.35h | 3.21h | 5M tokens | 10M tokens and stage boundary |
| D | 250M | 5.82h | 0.87h | 8.04h | 10M tokens | 25M tokens and stage boundary |
| D | 500M | 11.65h | 1.75h | 16.07h | 20M tokens | 25M tokens and final audit |

## Corpus Storage

| Tokens | Raw uint16 | Planned shard/index budget |
|---:|---:|---:|
| 5M | 0.009 GiB | 0.010 GiB |
| 25M | 0.047 GiB | 0.049 GiB |
| 100M | 0.186 GiB | 0.196 GiB |
| 250M | 0.466 GiB | 0.489 GiB |
| 500M | 0.931 GiB | 0.978 GiB |

## Recommended Operating Plan

Candidate C is projected at 10.12h active GPU and 13.96h conservative wall time for 500M tokens. A complete model-plus-optimizer resume point is approximately 597.43 MiB before scheduler/RNG metadata.

Use atomic checkpoint directories, retain the latest two valid checkpoints plus every stage boundary, fsync manifests before publishing the completion marker, and validate a real process resume at Stage 0 and Stage 1. A laptop power loss must never leave the last known-good checkpoint replaceable by a partial write.

The microbenchmark omits real shard I/O, periodic full validation, generation audits, and thermal behavior over many hours. The 20% wall contingency does not replace a Stage 1 calibration run on the approved corpus.
