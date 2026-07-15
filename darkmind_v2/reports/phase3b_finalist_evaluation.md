# Phase 3B Finalist Evaluation

Each unique milestone used 200 deterministic greedy generations and 500 fixed seeded-sampling generations. The best-validation checkpoint equaled the final checkpoint for both finalists, so its byte-identical audit is reused rather than regenerated.

The meaningful-continuation count is a structural proxy: at least four generated tokens, no exact n-gram loop, no hard failure, and no empty/repetition/script/special-token/invalid-byte warning. It is not a human quality judgment.

| Candidate | Step | Tokens | Val loss | Eval loss | Greedy rep warn | Greedy loops | Greedy unique | Sampling rep warn | Sampling loops | Sampling unique | EOS greedy/sample | Meaningful proxy greedy/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| C | 0 | 0 | 10.272607 | 10.275949 | 200 | 200 | 0.169 | 7 | 7 | 0.836 | 0.000/0.000 | 0/135 |
| C | 152 | 1,245,184 | 6.766369 | 6.748732 | 196 | 191 | 0.170 | 231 | 204 | 0.561 | 0.090/0.408 | 4/267 |
| C | 305 | 2,498,560 | 6.340935 | 6.322310 | 148 | 123 | 0.471 | 98 | 77 | 0.800 | 0.455/0.678 | 49/375 |
| C | 458 | 3,751,936 | 6.181056 | 6.163216 | 125 | 103 | 0.541 | 123 | 117 | 0.767 | 0.485/0.606 | 72/338 |
| C | 610 | 4,997,120 | 6.148458 | 6.130813 | 142 | 112 | 0.517 | 136 | 129 | 0.747 | 0.440/0.600 | 53/343 |
| D | 0 | 0 | 10.283829 | 10.285346 | 198 | 198 | 0.205 | 2 | 2 | 0.837 | 0.000/0.000 | 1/174 |
| D | 152 | 1,245,184 | 6.802695 | 6.784964 | 200 | 195 | 0.128 | 210 | 198 | 0.547 | 0.050/0.334 | 0/290 |
| D | 305 | 2,498,560 | 6.371558 | 6.352909 | 100 | 53 | 0.767 | 72 | 56 | 0.816 | 0.765/0.722 | 95/413 |
| D | 458 | 3,751,936 | 6.209152 | 6.190742 | 130 | 99 | 0.552 | 86 | 67 | 0.793 | 0.520/0.616 | 65/402 |
| D | 610 | 4,997,120 | 6.175414 | 6.157462 | 176 | 144 | 0.408 | 90 | 71 | 0.776 | 0.295/0.572 | 22/397 |

All raw generations remain in ignored runtime manifests. Invalid UTF-8 byte sequences, U+FFFD, mojibake, unexpected/mixed script, special-token leakage, output lengths, language/category counts, and finite-logit status are retained in the machine-readable audit summaries and manifests.
