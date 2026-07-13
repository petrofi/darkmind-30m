# Phase 3A Release Strategy

## Research Base

DarkMind v2 is first a from-scratch Turkish-English decoder-only base model for research, language-model development, reproducibility, and tokenizer/corpus/training experiments.

A base checkpoint is not a chat assistant. Its release materials must state its training stage, corpus scope, limitations, known repetition or Unicode failures, evaluation coverage, intended research use, and unresolved risks. No checkpoint may be described as conversational, helpful, production-ready, or safe merely because loss decreased.

The base release path requires:

- all required stage gates through the intended release stage;
- frozen-tokenizer and corpus-manifest compatibility;
- deterministic checkpoint reload and real process resume;
- Turkish, English, technical, code, repetition, EOS, and Unicode audits;
- contamination and attribution review;
- resolved model-weight license and local Hugging Face package validation;
- explicit human approval.

## User-Facing Instruct Model

The model users would actually chat with is a separate instruct model. It requires a sufficiently capable and explicitly approved base checkpoint, controlled instruction data, supervised fine-tuning, conversation evaluation, safety evaluation, Turkish human review, English review, technical/code review, and strict repetition and Unicode gates.

SFT must not begin before the approved base checkpoint passes its required quality stage. Qwen or other teacher data must not be generated before instruction-data policy, provenance, license, quality, and contamination controls are approved. An instruct checkpoint must not be published until responses are consistently usable across Turkish and English prompts and hard safety/repetition gates pass.

## Release Sequence

1. Approve corpus sources and the recommended architecture; architecture C remains a recommendation, not a freeze.
2. Run Stage 0 initialization and checkpoint/resume validation.
3. Train only to the next approved token gate and stop for evaluation.
4. Continue only when the gate passes and the user approves the next budget.
5. At 500M, complete strict base audit, contamination review, weight-license decision, and offline release-package validation.
6. Release a research base only with an accurate model card and no chat-assistant claims.
7. Begin a separately approved instruct phase; publish only after conversation and safety gates pass.

No automatic Hugging Face upload is part of this plan.
