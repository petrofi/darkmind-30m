# DarkMind v2 Phase 4B Sequence-Order Audit

Legacy order classification: **severely clustered**

Material clustering was predeclared as any major window exceeding 10 percentage points in language/category, 0.10 bits source-family Jensen-Shannon divergence, or a 64-sequence contiguous source-family run.

| Window | Tokens | TR % | EN % | Prose % | Technical % | Source JSD | Longest source run | Longest language run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steps_1_128 | 1,048,576 | 59.550 | 40.450 | 68.895 | 31.105 | 0.708904 | 19 | 21 |
| steps_129_305 | 1,449,984 | 60.061 | 39.939 | 69.851 | 30.149 | 0.708855 | 19 | 19 |
| steps_306_458 | 1,253,376 | 60.728 | 39.272 | 70.194 | 29.806 | 0.708832 | 20 | 20 |
| steps_459_610 | 1,245,184 | 61.573 | 38.427 | 70.854 | 29.146 | 0.708822 | 21 | 23 |

Detailed source-family percentages, document statistics, EOS density, validation/eval comparisons, and percentage-point differences are retained in the JSON report.
