# Phase 5A next-phase decision

## Dominant decision

**Decision A: expand the unique corpus.** Build Corpus V4 Tranche 2 toward approximately 200M new validated unique tokens and approximately 300M cumulative unique tokens. Do not begin an identical second Corpus V3 epoch.

The decision follows four observations:

1. Validation and eval loss remained positive through 98,074,624 tokens with zero rebound.
2. Marginal learning slowed after 75M, but did not stop; there is no sufficient evidence that Base V1 capacity is saturated.
3. Weak technical/factual continuation, repetition, loops, EOS, punctuation closure, English representation, and source diversity align with data and mixture deficits.
4. The registry has 338M approved-plus-conditional candidate capacity, although only 52M is currently unconditional and every conditional source still needs item-level clearance.

Architecture dimensions, frozen tokenizer, Corpus V3, the final checkpoint, and V2 training policy remain unchanged. Capacity review becomes the dominant alternative only if category probes stop improving on well-diversified new data or the controlled P1/P2 experiment shows broad saturation.

## Required next gates

- Resolve enough conditional source rights to support the tranche without quota filling.
- Inventory, hash, attribute, filter, and deduplicate new sources against Corpus V3.
- Freeze a small 4,997,120-token unique-data comparison slice.
- Compare P1 low-LR continuation with P2 mild rewarm under identical controls.
- Activate P3 optimizer-state review only with evidence of stale-moment adaptation failure.
- Require a new explicit authorization before any longer pretraining run.
- Repeat automatic and blinded manual quality review before reconsidering SFT or release.

SFT is not yet justified. Public upload is technically possible as packaging, but publicly inadvisable because generation quality is weak, EOS is low, loops remain high, human review is incomplete, and the model-weight license is unresolved.

No training was run, no second epoch began, no Corpus V4 data was acquired, and no SFT, Qwen generation, or upload occurred during Phase 5A.

**DARKMIND V2 BASE V1 REQUIRES CORPUS V4 EXPANSION BEFORE INSTRUCTION TUNING OR PUBLIC RELEASE**
