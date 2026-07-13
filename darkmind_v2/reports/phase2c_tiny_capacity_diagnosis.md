# DarkMind v2 Phase 2C Tiny Capacity Diagnosis

## Experiment

This is one deterministic epoch-equivalent with 99.9915% train-token coverage. It restarted from the same seed and did not reuse Stage-1 weights.
The run consumed 11,743,232 of 11,744,226 train tokens in 2,867 optimizer steps; the final 994-token tail was excluded without wraparound.

## Loss Progression

| Step | Tokens | Validation loss | Eval loss | Perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 10.123761 | 10.123414 | 24928.340 |
| 256 | 1,048,576 | 6.854168 | 6.839016 | 947.824 |
| 717 | 2,936,832 | 6.162291 | 6.155099 | 474.514 |
| 1,434 | 5,873,664 | 5.761349 | 5.750333 | 317.777 |
| 2,150 | 8,806,400 | 5.561690 | 5.552910 | 260.262 |
| 2,867 | 11,743,232 | 5.483633 | 5.476849 | 240.720 |

Validation and eval loss improve throughout the run. There is no measured train/validation divergence, so classic overfitting is not the dominant failure.

## Stage-1 Comparison

| Metric | Stage-1 | Full epoch-equivalent | Change |
|---|---:|---:|---:|
| Validation loss | 7.096028 | 5.483633 | -1.612396 |
| Eval loss | 7.081271 | 5.476849 | -1.604422 |
| Greedy repetition warning | 89.5% | 75.5% | -14.0% points |
| Greedy exact n-gram loop | 65.5% | 74.5% | 9.0% points |
| Greedy EOS completion | 41.0% | 25.0% | -16.0% points |
| Sampling repetition warning | 31.6% | 14.8% | -16.8% points |
| Sampling exact n-gram loop | 13.4% | 14.0% | 0.6% points |
| Sampling EOS completion | 64.6% | 58.0% | -6.6% points |

## Human Quality Review

- Meaningful Turkish ordinary continuations: 2/60 (3.3%), both short and generic.
- Meaningful English ordinary continuations: 0/50 under a strict topical-continuation criterion.
- Factual success: 0%; outputs are unreliable and often numeric loops.
- Technical/code success: 0%; repeated symbols, fragments, or irrelevant continuations dominate.
- Greedy language consistency improved mechanically, but this did not produce useful semantics.

## Diagnosis

**Outcome B - capacity-limited, with data scale/composition as a secondary constraint.**

The model has 9,369,088 parameters, but 6,144,000 (65.58%) belong to the tied 24k embedding table. Only 3,225,088 parameters remain for attention, MLP, normalization, and positions.
Loss improvement proves the pipeline learns, but generation remains largely repetitive and incoherent. Greedy repetition falls by 14 points, while exact loops worsen by 9 points and EOS completion falls by 16 points.
The current corpus also imprints numeric and Python/C-symbol patterns on weak generations, so a larger model should not be trained on this 13.05M-token corpus alone.

A second tiny epoch is not technically justified from this evidence. It could lower loss further, but the dominant representational bottleneck and unstable generation metrics are unlikely to be solved by repeating the same data.

## Release Decision

Public research-preview eligibility remains **FAIL**. Pipeline hard failures are zero, but quality is inadequate and the model-weight distribution license remains unresolved.
