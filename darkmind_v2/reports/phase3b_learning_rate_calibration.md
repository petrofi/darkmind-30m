# Phase 3B Learning-Rate Calibration

Each run used the same seed, first 524,288 ordered train tokens, validation subset, optimizer, effective batch, warmup, and cosine horizon.

| Candidate | Peak LR | Initial val | Final val | Improvement | Slope/step | Grad p95 | tok/s | VRAM GiB | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C | 0.0001 | 10.270182 | 8.076579 | 2.193603 | -0.026749 | 3.4375 | 24,077.0 | 1.56 | PASS |
| C | 0.0002 | 10.270182 | 7.688493 | 2.581689 | -0.027532 | 2.6094 | 24,097.7 | 1.56 | PASS |
| C | 0.0003 | 10.270182 | 7.615174 | 2.655008 | -0.024248 | 2.4219 | 24,244.7 | 1.56 | PASS |
| D | 0.0001 | 10.283493 | 8.078627 | 2.204867 | -0.026654 | 3.4219 | 21,317.1 | 1.77 | PASS |
| D | 0.0002 | 10.283493 | 7.700303 | 2.583190 | -0.027257 | 2.5938 | 21,459.6 | 1.77 | PASS |
| D | 0.0003 | 10.283493 | 7.631346 | 2.652147 | -0.023889 | 2.3750 | 21,436.6 | 1.77 | PASS |

Selected Candidate C peak LR: `0.0003`

Selected Candidate D peak LR: `0.0003`

## Process Reliability Note

The first Candidate C LR=0.0001 child exited with `3221226505` before writing a result. The identical retry passed and produced the table entry above. The failed attempt remains part of the architecture reliability gate.
