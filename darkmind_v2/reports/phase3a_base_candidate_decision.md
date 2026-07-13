# Phase 3A Base Candidate Decision

Status: **RECOMMENDED PENDING CORPUS APPROVAL; NOT FROZEN**

Recommended architecture: **Candidate C (103,881,216 parameters)**

Recommendation strength: **MODERATE**. Candidate C has safe measured headroom and practical throughput, while one isolated checkpointing-on MB2 SDPA worker exited without a result. The matching checkpointing-off SDPA and fallback profiles passed, so this is a profile-level stability concern rather than an architecture-wide hard rejection.

## Weighted Decision

| Rank | Candidate | Weighted score | Params | Body params | Vocab share | BF16 tok/s | VRAM headroom | Stable profiles |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | D | 64.29 | 118,056,960 | 99,624,960 | 15.6128% | 14,906.3 | 82.12% | 8/8 |
| 2 | C | 62.71 | 103,881,216 | 85,449,216 | 17.7433% | 17,162.1 | 83.85% | 7/8 |
| 3 | B | 55.42 | 79,694,720 | 64,334,720 | 19.2735% | 19,165.3 | 85.42% | 8/8 |
| 4 | A | 44.35 | 62,989,312 | 50,701,312 | 19.5081% | 17,915.5 | 87.32% | 8/8 |

The raw weighted score places D ahead of C by less than 3 percentage points. The required tie-break therefore selects the smaller and faster C. D is not selected merely for being larger.

## Why C

- C has 85,449,216 transformer-body parameters, 32.8% more than B, while its preferred-profile throughput is only 10.5% lower than B.
- C leaves 83.85% of reserved VRAM free in the measured MB2 SDPA profile, far above the 15% safety gate.
- C uses the shallowest candidate depth (12 layers), standard 64-dimensional heads, and the same tied-tokenizer contract as every candidate.
- C checkpoint save/reload passed; its BF16 package is practical at about 201 MiB.

## Rejections and Deferrals

- A is not recommended because its 50,701,312-parameter body leaves materially less capacity than C; the speed advantage over C is only about 4.4%.
- B is the fastest candidate and remains the fallback recommendation if future long-run C stability fails, but its body is 21.1M parameters smaller than C.
- D is not recommended because it is about 13.1% slower than C, has a larger/slower checkpoint, and is within the 3-point score tie band where the smaller/faster rule applies.
- No candidate was rejected for vocabulary share, head dimensions, tied embeddings, OOM, non-finite values, or checkpoint reload.

## Gradient Checkpointing

| Candidate | MB1 SDPA throughput change | Reserved-memory change |
|---|---:|---:|
| A | -28.1% | -11.6% |
| B | -36.5% | -0.2% |
| C | -24.5% | -9.6% |
| D | -27.9% | -1.5% |

Checkpointing is implemented and valid, but it is not recommended for the measured MB1/MB2 profiles because headroom is already large and recomputation costs 24-37% throughput. It remains available for longer contexts or future larger batches.

## Hard Failure Record

Candidate C MB2 + gradient checkpointing + SDPA exited with Windows code 3221226505 before writing a result. No OOM was reported. The failure remains in the benchmark JSON and lowers C's stability score to 87.5%; it must be rerun during architecture-freeze review.
