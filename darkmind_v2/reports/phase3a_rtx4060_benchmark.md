# Phase 3A RTX 4060 Laptop Benchmark

GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Each profile ran in an isolated process with deterministic synthetic token IDs, 10 warmup microsteps, and 30 measured optimizer microsteps. Checkpoint operations are excluded from active-step throughput.

Seeds, inputs, ordering, and protocol were fixed. The measured Windows PyTorch backend warned that CuBLAS and memory-efficient attention may not be bitwise deterministic.

PyTorch SDPA was measured, but this build reported that Flash Attention was unavailable.

| Candidate | MB | GC | Attention | tok/s | avg ms | p95 ms | Peak alloc | Peak reserved | Headroom | Result |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| A | 1 | on | sdpa | 5,562.1 | 92.1 | 113.5 | 0.57 GiB | 0.64 GiB | 92.0% | PASS |
| A | 1 | on | fallback | 4,279.2 | 119.6 | 144.7 | 0.57 GiB | 0.64 GiB | 92.0% | PASS |
| A | 1 | off | sdpa | 7,739.9 | 66.2 | 78.3 | 0.66 GiB | 0.73 GiB | 90.9% | PASS |
| A | 1 | off | fallback | 8,303.9 | 61.7 | 70.3 | 0.86 GiB | 0.89 GiB | 88.9% | PASS |
| A | 2 | on | sdpa | 12,204.7 | 83.9 | 103.1 | 0.67 GiB | 0.86 GiB | 89.3% | PASS |
| A | 2 | on | fallback | 10,187.7 | 100.5 | 112.5 | 0.67 GiB | 0.90 GiB | 88.8% | PASS |
| A | 2 | off | sdpa | 17,915.5 | 57.2 | 64.6 | 0.95 GiB | 1.01 GiB | 87.3% | PASS |
| A | 2 | off | fallback | 13,876.3 | 73.8 | 84.9 | 1.33 GiB | 1.44 GiB | 82.0% | PASS |
| B | 1 | on | sdpa | 7,927.3 | 64.6 | 69.3 | 0.70 GiB | 0.85 GiB | 89.3% | PASS |
| B | 1 | on | fallback | 6,670.9 | 76.8 | 80.6 | 0.70 GiB | 0.84 GiB | 89.5% | PASS |
| B | 1 | off | sdpa | 12,490.9 | 41.0 | 46.9 | 0.76 GiB | 0.85 GiB | 89.3% | PASS |
| B | 1 | off | fallback | 10,010.5 | 51.1 | 56.3 | 0.95 GiB | 1.02 GiB | 87.3% | PASS |
| B | 2 | on | sdpa | 15,802.6 | 64.8 | 70.3 | 0.77 GiB | 1.01 GiB | 87.3% | PASS |
| B | 2 | on | fallback | 11,278.3 | 90.8 | 98.3 | 0.77 GiB | 1.06 GiB | 86.7% | PASS |
| B | 2 | off | sdpa | 19,165.3 | 53.4 | 60.0 | 1.04 GiB | 1.17 GiB | 85.4% | PASS |
| B | 2 | off | fallback | 13,139.1 | 77.9 | 85.5 | 1.44 GiB | 1.55 GiB | 80.7% | PASS |
| C | 1 | on | sdpa | 8,667.7 | 59.1 | 64.9 | 0.92 GiB | 0.99 GiB | 87.6% | PASS |
| C | 1 | on | fallback | 7,903.9 | 64.8 | 70.0 | 0.91 GiB | 1.06 GiB | 86.8% | PASS |
| C | 1 | off | sdpa | 11,479.6 | 44.6 | 48.7 | 0.92 GiB | 1.10 GiB | 86.2% | PASS |
| C | 1 | off | fallback | 10,025.0 | 51.1 | 58.2 | 1.14 GiB | 1.21 GiB | 84.9% | PASS |
| C | 2 | on | sdpa | 0.0 | 0.0 | 0.0 | 0.00 GiB | 0.00 GiB | 0.0% | worker did not produce a result |
| C | 2 | on | fallback | 9,599.9 | 106.7 | 114.6 | 0.97 GiB | 1.19 GiB | 85.1% | PASS |
| C | 2 | off | sdpa | 17,162.1 | 59.7 | 67.4 | 1.22 GiB | 1.29 GiB | 83.9% | PASS |
| C | 2 | off | fallback | 11,784.3 | 86.9 | 99.4 | 1.64 GiB | 1.72 GiB | 78.5% | PASS |
| D | 1 | on | sdpa | 6,606.2 | 77.5 | 124.6 | 1.02 GiB | 1.16 GiB | 85.5% | PASS |
| D | 1 | on | fallback | 6,302.1 | 81.2 | 108.1 | 1.03 GiB | 1.17 GiB | 85.4% | PASS |
| D | 1 | off | sdpa | 9,165.0 | 55.9 | 73.5 | 1.03 GiB | 1.18 GiB | 85.3% | PASS |
| D | 1 | off | fallback | 7,807.1 | 65.6 | 87.2 | 1.28 GiB | 1.35 GiB | 83.2% | PASS |
| D | 2 | on | sdpa | 11,085.2 | 92.4 | 130.4 | 1.04 GiB | 1.33 GiB | 83.4% | PASS |
| D | 2 | on | fallback | 8,564.7 | 119.6 | 135.6 | 1.07 GiB | 1.36 GiB | 83.0% | PASS |
| D | 2 | off | sdpa | 14,906.3 | 68.7 | 74.4 | 1.36 GiB | 1.43 GiB | 82.1% | PASS |
| D | 2 | off | fallback | 10,407.8 | 98.4 | 102.5 | 1.84 GiB | 2.02 GiB | 74.8% | PASS |

## Candidate Checkpoints

- Candidate A: 124.16 MiB, save 0.264s, reload 0.662s
- Candidate B: 155.27 MiB, save 0.307s, reload 0.777s
- Candidate C: 201.15 MiB, save 0.392s, reload 0.962s
- Candidate D: 228.69 MiB, save 0.557s, reload 1.565s

## Protocol Notes

Peak reserved VRAM is the safety basis because it captures allocator reservation. Micro-batch 2 was attempted only for matching micro-batch-1 profiles with at least 25% reserved-memory headroom. OOM and CUDA failures remain visible in the table and do not remove a candidate silently.
