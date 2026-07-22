# Phase 5A SFT readiness

## Decision

**SFT not yet justified.**

Base V1 is a genuine from-scratch checkpoint, but the current evidence does not show sufficiently stable grammatical continuation, topical consistency, technical structure, non-repetitive generation, or EOS behavior for instruction tuning.

## Evidence

- Greedy repetition is 60.0% and exact loops are 56.8% on the new 440-prompt suite.
- Seeded-sampling repetition and exact loops are both 33.2%.
- EOS completion is 2.0% greedy and 8.4% sampled.
- Punctuation completion proxies are 5.5% greedy and 16.4% sampled.
- Code-structure proxies are 65.0% and 67.5%, but they do not establish semantic correctness.
- Unicode health, special-token containment, and non-empty output behavior are strong.
- The automatic language heuristic reported no switch errors, but this is not a human language-quality judgment.
- The balanced 150-output manual packet remains unscored, so grammatical, topical, factual, and technical usefulness have not been human-validated.

SFT can teach response format, role behavior, and instruction-following conventions. It cannot reliably repair a weak base distribution, persistent loops, poor semantic continuation, or missing technical knowledge. Starting SFT now risks teaching a polished response shape over unstable language behavior.

Instruction-data policy, provenance, and evaluation planning may proceed as documentation only. Dataset generation, Qwen teacher generation, instruction-data acquisition, and SFT training remain outside Phase 5A. Reconsider SFT after Corpus V4 continuation demonstrates lower repetition/loops, materially better EOS and completion behavior, stable language quality, and a completed blinded human review.
