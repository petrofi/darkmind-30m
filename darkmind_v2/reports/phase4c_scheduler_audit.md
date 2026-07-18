# DarkMind v2 Phase 4C Scheduler Audit

Optimizer stepping order is `optimizer.step()` followed by `scheduler.step()`. The LR recorded before the optimizer call is the LR actually applied. LambdaLR epoch zero maps to optimizer step one, and restored scheduler state advances to the exact next LR without replay.

| Step | Phase 3B local 3e-4 | Global 1e-4 | Global 7.5e-5 | Staged 1e-4 -> 5e-5 |
|---:|---:|---:|---:|---:|
| 1 | 1.4999999999999999e-05 | 0.000001000 | 0.000000750 | 0.000001000 |
| 64 | 0.00029631177534073004 | 0.000064000 | 0.000048000 | 0.000064000 |
| 100 | 0.0002879356943354133 | 0.000100000 | 0.000075000 | 0.000100000 |
| 128 | 0.00027828573337538535 | 0.000099999 | 0.000074999 | 0.000099629 |
| 192 | 0.0002472303092672385 | 0.000099990 | 0.000074994 | 0.000096092 |
| 256 | 0.00020671729424061788 | 0.000099971 | 0.000074982 | 0.000089318 |
| 384 | 0.00011650849687453078 | 0.000099905 | 0.000074939 | 0.000070558 |
| 512 | 4.796694820919524e-05 | 0.000099800 | 0.000074872 | 0.000054419 |
| 610 | 2.9999999999999997e-05 | 0.000099694 | 0.000074803 | 0.000050000 |
| 3051 | n/a | 0.000090230 | 0.000068720 | 0.000047892 |
| 12207 | n/a | 0.000030000 | 0.000030000 | 0.000030000 |

The staged candidate warms to 1e-4, decays continuously to 5e-5 at the 5M gate, then continues from that exact LR toward 3e-5 at the 100M horizon. It has no hidden reset at step 610.

Staged LR at 25M step 3051: `0.000047892`.
Staged LR at 100M step 12207: `0.000030000`.
