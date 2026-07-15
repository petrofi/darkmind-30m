# Phase 3B Candidate C Checkpointing Diagnosis

Classification: **intermittent process/backend instability**

The checkpointing diagnosis environment was Windows, PyTorch 2.4.1+cu121, CUDA 12.1, RTX 4060 Laptop GPU, BF16, sequence length 512, micro-batch 2, SDPA, and non-reentrant gradient checkpointing.

Phase 3A preserved one worker exit with code 3221226505 (`STATUS_STACK_BUFFER_OVERRUN` / fail-fast, `0xC0000409`) before a result file was written. The five identical Phase 3B attempts below did not reproduce that exit.

## Identical Candidate C Attempts

| Attempt | Exit | Windows exception | Last completed operation | Result |
|---:|---:|---|---|---|
| 1 | 0 | - | worker_complete | PASS |
| 2 | 0 | - | worker_complete | PASS |
| 3 | 0 | - | worker_complete | PASS |
| 4 | 0 | - | worker_complete | PASS |
| 5 | 0 | - | worker_complete | PASS |

## Controls

| Candidate | MB | Attention | Checkpointing | Exit | Result |
|---|---:|---|---|---:|---|
| C | 2 | sdpa | off | 0 | PASS |
| C | 1 | sdpa | on | 0 | PASS |
| C | 2 | fallback | on | 0 | PASS |
| C | 1 | fallback | on | 0 | PASS |
| D | 2 | sdpa | on | 0 | PASS |
| D | 2 | sdpa | off | 0 | PASS |
| D | 1 | sdpa | on | 0 | PASS |
| D | 2 | fallback | on | 0 | PASS |
| D | 1 | fallback | on | 0 | PASS |

## Subsequent Checkpointing-Off Incidents

1. The first C LR=0.0001 calibration worker exited with `3221226505` and wrote no result; its exact retry passed.
2. The first C Segment B child ended before a new metric was written, left the midpoint checkpoint unchanged, and emitted no diagnostic stdout/stderr. Its native exception code is unavailable; the instrumented exact retry passed.

These two events occurred in the recommended checkpointing-off profile. They satisfy the predeclared repeated-process-crash hard gate even though the successful retries, memory audit, and soak show that the failure is intermittent.

## Policy

Gradient checkpointing is unnecessary at the measured memory levels and remains disabled. The exact C checkpointing-on combination is rejected before training. The broader C checkpointing-off signature emits a warning on the affected Windows/PyTorch environment. Candidate D is the frozen production-base architecture.

No deterministic source-code defect, OOM, or exact native cause was established. The evidence supports intermittent process/backend instability; all failed and successful evidence remains in ignored runtime JSON.
