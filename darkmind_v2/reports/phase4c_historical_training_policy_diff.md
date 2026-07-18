# DarkMind v2 Phase 4C Historical Training Policy Diff

Absolute validation losses are not treated as directly comparable because the Phase 3B and Phase 4 corpora/splits differ. Curve shape, LR trajectory, gradients, and implementation are compared.

| Property | Phase 3B Candidate D | Phase 4A | Phase 4B best arm |
|---|---|---|---|
| Model dimensions | Base V1 / 118,056,960 | Same | Same |
| Seed | 20260712 | 20260712 | 20260712 |
| Corpus | Phase 2A full_v1 | Corpus V3 | Corpus V3 |
| Order | Contiguous | Contiguous | deterministic_stratified_v1 |
| Batch | 512 x 2 x 8 = 8,192 tokens | Same | Same |
| AdamW | beta 0.9/0.95, wd 0.1, one group | Same | Same |
| Warmup | 20 steps | 100 steps | 100 steps |
| Cosine horizon | 610 steps / local 5M | 12,207 / global 100M | 12,207 / global 100M |
| Peak/min LR | 0.0003 / 0.00003 | 0.0003 / 0.00003 | 0.00015 / 0.00003 |
| LR at step 610 | 0.000030000 | 0.000298820 | 0.000149475 |
| Precision / attention | BF16 / SDPA | BF16 / SDPA | BF16 / SDPA |
| Loss/shift | Same model implementation | Same | Same |

## Finding

The Phase 3B finalist pilot used a 5M-local cosine schedule and reached its minimum LR at step 610. Phase 4A/4B used a 100M-global schedule, leaving LR close to peak throughout the 5M gate. This is the largest controlled policy difference aligned with the observed curve-shape difference.

Re-derived BF16 tensor initialization identity: **PASS**.
