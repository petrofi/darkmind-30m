# DarkMind v2 Phase 4C Corrected Training Policy

Final classification: **PASS**
Selected exploratory arm: `arm3_global_lr1e4_corrected_groups`
V2 config SHA-256: `9358b8b33a87729ef2f19cfad76acbe370ed44ada911fa21512e4085eccf52ea`

## Diagnosis

The Phase 3B pilot used a 5M-local cosine schedule, while Phase 4A/4B used a 100M-global schedule that stayed near peak LR through step 610. Phase 4B peak LR 1.5e-4 remained too high. At 1e-4, the Base V1 loss trajectory is stable without rebound.

The prior optimizer placed bias, LayerNorm, token/position embeddings, and the tied LM-head weight in the decay group. The tied tensor appeared once, so there was no duplicate optimizer state. V2 uses decay only for approved matrix weights and no decay for bias, normalization, and embeddings.

The staged first-5M decay remained stable but underfit the global 1e-4 schedule. Depth-scaled residual initialization improved loss but failed the diagnostic-health gate because residual growth reached 8.30x its initialization ratio and 96.7% of steps clipped. Standard Base V1 initialization is retained.

## Exploratory arms

| Arm | Final val | Final eval | Val/eval rebound | Clip fraction | Residual growth | Stability |
|---|---:|---:|---:|---:|---:|---|
| arm1_global_lr1e4_current_groups | 6.356273 | 6.306373 | 0.000% / 0.000% | 0.707 | 2.704x | stable |
| arm2_global_lr75e6_current_groups | 6.570066 | 6.524260 | 0.000% / 0.000% | 0.833 | 2.339x | stable |
| arm3_global_lr1e4_corrected_groups | 6.356242 | 6.306306 | 0.000% / 0.000% | 0.707 | 2.704x | stable |
| arm4_staged_decay_corrected_groups | 6.517767 | 6.471126 | 0.000% / 0.000% | 0.631 | 2.703x | stable |
| arm5_depth_scaled_init_staged | 6.358198 | 6.311177 | 0.000% / 0.000% | 0.967 | 8.303x | partial stability |

## Frozen V2 policy

- Initialization: `base_v1_standard_v1`, seed `20260712`.
- Optimizer: AdamW, corrected decay/no-decay groups, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.1, gradient clip 1.0.
- Schedule: 100M-global warmup cosine, peak `0.0001`, minimum `0.00003`, warmup 100 steps, no restart.
- Applied LR at 5M/25M/100M: `0.000099693965` / `0.000090230387` / `0.000030000000`.
- Sequence order: `deterministic_stratified_v1`, no replacement and no wrap.

## Immutable confirmation

- Initialization hash: `f1da070885650b70dc999f22b6ef8a438bb47fe7479020dc46ebdf68ae3d9c6b`.
- Steps/tokens: `610` / `4997120`.
- Validation: `10.246471` to `6.356269` (37.966% improvement), rebound `0.000%`.
- Eval: `10.243818` to `6.306323` (38.438% improvement), rebound `0.000%`.
- Fresh process restart: `PASS`, step 305 to 306 continuity; PIDs `[25844, 24016, 24244]`.
- Authoritative generation: greedy repetition 200/200, loops 200/200; sampling repetition 296/500, loops 293/500.
- Generation quality remains diagnostic at 5M and is not a chatbot-quality claim.

The exact step-610 confirmation checkpoint is resume-capable. A 25M continuation is recommended only after explicit user approval; it was not started here.
