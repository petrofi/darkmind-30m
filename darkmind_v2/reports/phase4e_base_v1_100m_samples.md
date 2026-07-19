# DarkMind v2 Phase 4E Controlled Samples

Raw outputs are retained only in the external Phase 4E runtime. This source-controlled report contains aggregate diagnostics and no copyrighted passages.

| Checkpoint | Mode | Count | Repetition | Exact loops | EOS | Empty | Unique ratio | Hard failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25M authoritative | greedy | 200 | 50.0% | 29.0% | 4.0% | 5 | 0.290 | 0 |
| 25M authoritative | sampling | 500 | 19.8% | 16.8% | 11.0% | 18 | 0.601 | 0 |
| 50M subset | greedy | 16 | 68.8% | 62.5% | 0.0% | 1 | 0.236 | 0 |
| 50M subset | sampling | 16 | 12.5% | 6.2% | 18.8% | 1 | 0.596 | 0 |
| 75M subset | greedy | 16 | 50.0% | 50.0% | 12.5% | 2 | 0.290 | 0 |
| 75M subset | sampling | 16 | 37.5% | 37.5% | 18.8% | 1 | 0.635 | 0 |

The 50M and 75M rows use controlled 16-prompt subsets and are directional, not authoritative quality estimates. The final 200/500 audit was not run because the final segment was not authorized.
