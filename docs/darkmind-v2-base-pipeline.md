# DarkMind v2 Base Pipeline

This document defines the next DarkMind base-model pipeline after the Pilot500 TR/EN v2 failure analysis. It is a design document only. It does not start tokenizer training, corpus downloads, pretraining, SFT, or new teacher-data generation.

## Scope

- Turkish and English only.
- From-scratch base model.
- No instruction tuning until base generation quality passes defined gates.
- Target hardware: consumer NVIDIA RTX 4060 Laptop GPU constraints.
- Prioritize reproducibility, tokenizer/data hygiene, and deterministic evaluation over rushing to a larger checkpoint.

## Tokenizer

Recommended initial target:

- 24,000-token vocabulary.
- SentencePiece BPE or Unigram.
- Byte fallback or equivalent unknown-character protection.
- Unicode NFC normalization before tokenizer training.
- Encoding repair before tokenizer training.
- Explicit mojibake rejection before tokenizer training.
- Turkish character preservation, including `ç`, `ğ`, `ı`, `İ`, `ö`, `ş`, and `ü`.
- Tokenizer version manifest.
- SHA-256 hashes for tokenizer files.
- Immutable tokenizer once pretraining begins.

Required tokenizer manifest fields:

- tokenizer name and version.
- training corpus manifest hash.
- normalization policy.
- vocabulary size.
- tokenizer algorithm.
- byte fallback setting.
- special token list and IDs.
- SHA-256 for all tokenizer files.
- creation date and script version.

## Corpus Hygiene

Every document entering tokenizer training or base pretraining must pass corpus checks:

- UTF-8 validation.
- Unicode normalization.
- mojibake detection.
- duplicate and near-duplicate removal.
- language identification for Turkish and English.
- quality scoring.
- source and license metadata.
- PII filtering.
- train/validation/test split before tokenization.

Rejected text should be counted and sampled in a report. The pipeline should fail closed when encoding corruption is detected above a reviewed threshold.

## Validation Before Full Training

Build a tiny tokenizer and data-pipeline smoke test first. Required gates:

- exact round-trip tests.
- no mojibake tokens.
- no replacement characters.
- acceptable Turkish and English tokens-per-character.
- fixed prompt encode/decode tests.
- reproducible tokenizer hashes.
- dataset checksums.
- deterministic split hashes.

The smoke test should include clean Turkish, clean English, mixed punctuation, short code snippets, and common Turkish software-assistant prompts.

## Model Progression

Do not immediately start the final model.

### Stage A: Tiny Base

- Tiny 5M-10M model.
- Short reviewed corpus.
- Verify that loss decreases.
- Verify that fixed prompts produce readable Turkish and English fragments.
- Verify checkpoint loading and tokenizer compatibility.

Stage A exits only when deterministic generation is readable enough to justify scaling.

### Stage B: Mid-Scale Base Candidate

- Approximately 45M-60M base model.
- Recommended starting architecture:
  - 12 layers.
  - 8 attention heads.
  - embedding size 512.
  - context size 512 initially.
  - 24k tokenizer vocabulary.
- Calculate the exact parameter count before implementation.

The exact parameter count must be recorded in the config and experiment report before training starts.

### Stage C: Base Pretraining Only

- Base pretraining only.
- Periodic deterministic generation tests.
- Checkpoint every fixed token interval.
- Stop if outputs become mixed-script or corrupted.

SFT, identity tuning, and teacher-student distillation remain forbidden in Stage C.

## Base Quality Gates Before SFT

Instruction tuning is forbidden until the base model can:

- produce readable Turkish for simple Turkish prompts.
- produce readable English for simple English prompts.
- complete short coherent sentences.
- avoid mixed-script corruption.
- preserve tokenizer round trips.
- outperform random or gibberish generation on a fixed prompt suite.
- pass at least 80% of basic language-format checks.

Failing these gates means the correct response is to revisit tokenizer, corpus, architecture, schedule, or training stability, not to add SFT.

## Evaluation

Create an immutable fixed-prompt suite covering:

- Turkish completion.
- English completion.
- basic factual continuation.
- simple code continuation.
- repetition.
- mixed-script detection.
- encoding corruption.
- empty output.
- checkpoint regression.

Evaluation should run deterministically with greedy decoding first. Sampling-based demos can be added only after deterministic outputs are sane.

Each evaluation report should include:

- checkpoint path and hash.
- tokenizer manifest hash.
- prompt-suite version.
- device.
- decoding settings.
- pass/fail counts.
- representative failures.
- recommendation to continue, pause, or stop.

## Pilot500 Reuse

The existing clean Pilot500 TR/EN dataset may be reused only after the new v2 base checkpoint passes its base-quality gates.

Pilot500 must not be used to compensate for a weak or unreadable base model. It is a later instruction-tuning resource, not a fix for base pretraining failure.

## Stop Conditions

Stop and audit before continuing if any of the following appears:

- mixed-script corruption in deterministic outputs.
- replacement characters or mojibake in generated text.
- tokenizer/checkpoint vocabulary mismatch.
- sudden validation loss divergence.
- memorized identity text in unrelated prompts.
- high duplicate rate in corpus stages.
- unstable checkpoint load or incompatible tokenizer hash.

## Resulting Path

The Pilot500 failure led to a base-pipeline redesign:

1. clean TR/EN corpus and tokenizer pipeline.
2. immutable tokenizer manifest and hashes.
3. tiny model smoke test.
4. base-only pretraining with fixed prompt gates.
5. SFT only after readable deterministic base generation.
