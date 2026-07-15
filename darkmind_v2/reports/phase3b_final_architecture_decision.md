# Phase 3B Final Architecture Decision

Status: **FROZEN**

Selected architecture: **Candidate D / darkmind-v2-base-v1**

Recommendation strength: **STRONG for the production-base architecture selection**. The 5M-token checkpoints remain early research pilots and are not finished or conversational models.

## Predeclared Rules

Before comparing final outcomes, a material D learning advantage was defined as at least 2% lower final validation loss than C, or a clearly better loss slope without a stability penalty. Hard eligibility gates are applied before score or the within-2% Candidate C tie-break.

## Equal-Token Learning

Both candidates consumed exactly **4,997,120 tokens** in 610 optimizer steps, with identical ordered sequences and a fresh-process restart at step 305.

| Candidate | Step | Validation loss | Eval loss |
|---|---:|---:|---:|
| C | 0 | 10.272607 | 10.275949 |
| C | 152 | 6.766369 | 6.748732 |
| C | 305 | 6.340935 | 6.322310 |
| C | 458 | 6.181056 | 6.163216 |
| C | 610 | 6.148458 | 6.130813 |
| D | 0 | 10.283829 | 10.285346 |
| D | 152 | 6.802695 | 6.784964 |
| D | 305 | 6.371558 | 6.352909 |
| D | 458 | 6.209152 | 6.190742 |
| D | 610 | 6.175414 | 6.157462 |

| Candidate | Validation reduction | Eval reduction | Validation reduction / 1M tokens | Eval reduction / 1M tokens | Mean loss reduction / wall hour |
|---|---:|---:|---:|---:|---:|
| C | 4.124148 | 4.145136 | 0.825305 | 0.829505 | 53.530 |
| D | 4.108415 | 4.127885 | 0.822156 | 0.826053 | 47.388 |

C finished with 0.436% lower validation loss and 0.433% lower eval loss. This is below the predeclared 2% materiality threshold.

## Generation-Health Trend

The composite averages no-repetition, no-loop, unique-token ratio, EOS completion, and meaningful-continuation proxy rates across greedy and seeded sampling. It is supporting evidence, not a conversational-quality score.

| Candidate | Init composite | Final composite | Trend | Final greedy repetition | Final sampling repetition |
|---|---:|---:|---:|---:|---:|
| C | 0.3247 | 0.5456 | +0.2209 | 142/200 | 136/500 |
| D | 0.3407 | 0.5033 | +0.1626 | 176/200 | 90/500 |

Both finalists eliminated invalid UTF-8 and replacement-character events by the final milestone. C had the stronger greedy trend; D had healthier final sampling repetition, loop, unique-token, and meaningful-proxy results. Outputs remain largely incoherent at this token budget.

## Weighted Score

| Candidate | Weighted score | Eligible | Validation 25% | Eval 15% | Loss/hour 10% | Body 10% | Generation 10% | Stability 10% | VRAM 10% | Resume 5% | Backend 5% |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 93.58 | FAIL | 100.00 | 100.00 | 100.00 | 85.77 | 100.00 | 100.00 | 100.00 | 50.00 | 50.00 |
| D | 95.85 | PASS | 99.62 | 99.58 | 88.52 | 100.00 | 73.63 | 100.00 | 97.92 | 100.00 | 100.00 |

## Hard-Gate Decision

C is ineligible because the recommended checkpointing-off Windows/PyTorch profile had two unexpected process terminations: one LR-calibration child exited with 3221226505 and one first Segment B child ended silently before writing a new metric. Both exact retries passed, which classifies the issue as intermittent process/backend instability rather than a deterministic model-code defect.

D passed post-warmup memory, 1,000-step soak, all calibration runs, safetensors integrity, and the forced midpoint checkpoint/resume continuity gates without a process crash. D is selected even though C's final losses are about 0.44% better, because hard eligibility precedes the score and tie-break.

The Phase 3A score-only recommendation for C is superseded. Candidate D is frozen as `darkmind_v2/config/model_base_v1.json`.
