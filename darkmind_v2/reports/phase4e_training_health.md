# DarkMind v2 Phase 4E Training Health

Recorded Stage-3 optimizer steps: `6,104`.
Pre-clipping gradient p50/p95/max: `1.726562` / `2.031250` / `2.734375`.
Clipped-step fraction: `100.000%`.
Update-to-weight p50/p95/max: `0.000496683` / `0.000756581` / `0.000881642`.
Non-finite events: `0`.

Clipping remained effectively 100%. Losses improved and gradient p95 stayed below twice the Phase 4D baseline, so the hard health stop did not fire; the frequency remains a material concern in the Conditional PASS decision.

| Step | Logit std | Prediction entropy | Embedding norm | Final/early residual RMS |
|---:|---:|---:|---:|---:|
| 3,051 | 1.851104 | 5.793063 | 0.646361 | 95.388398 |
| 4,096 | 1.990565 | 5.760346 | 0.653538 | 87.254251 |
| 5,120 | 1.937818 | 5.577746 | 0.657285 | 79.171456 |
| 6,103 | 1.946782 | 5.655141 | 0.659089 | 73.431893 |
| 7,168 | 1.956795 | 5.500879 | 0.659764 | 70.436298 |
| 8,192 | 1.916301 | 5.483047 | 0.660052 | 66.843295 |
| 9,155 | 1.941936 | 5.499532 | 0.660196 | 65.840187 |
