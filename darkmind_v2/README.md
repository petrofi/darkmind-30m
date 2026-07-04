# DarkMind v2 Phase 0

DarkMind v2 starts from the Pilot500 TR/EN v2 failure analysis. DarkMind v1 was stopped because the base checkpoint already failed deterministic generation, the tokenizer vocabulary contained mojibake artifacts, and instruction tuning amplified corrupted output despite decreasing loss.

Phase 0 is validation infrastructure only. It does not train a tokenizer, download a corpus, pretrain a model, run instruction tuning, or generate teacher data.

## Why Base Quality Comes First

Instruction tuning cannot fix an unreadable base model. Before any SFT work, the base model must produce readable Turkish and English continuations, avoid mixed-script corruption, preserve tokenizer round trips, and pass fixed prompt quality gates.

## Phase 0 Pipeline

The Phase 0 tools provide:

- UTF-8 and Unicode NFC validation.
- mojibake and replacement-character detection.
- Turkish/English language checks.
- exact and near-duplicate detection.
- deterministic corpus manifests with SHA-256 hashes.
- tokenizer manifests for already-existing tokenizer directories.
- tokenizer round-trip and vocabulary audits.
- immutable fixed base-prompt validation.

## Immutable Tokenizer Rule

Once DarkMind v2 pretraining begins, the tokenizer is immutable. Any tokenizer file change must create a new tokenizer version and a new manifest. Checkpoints must always be tied to the exact tokenizer manifest hash they were trained with.

## Deterministic Corpus Manifest Rule

The corpus manifest records input hashes, normalized output hashes, language counts, document counts, filtering statistics, deduplication settings, split seed, and deterministic train/validation/test hashes. Two runs with the same input and config must produce identical content hashes.

## No-Training Status

This directory is safe validation infrastructure. It intentionally contains no tokenizer training script, no corpus downloader, and no model training launcher.

## Next Gate

The next gate before tokenizer training is a reviewed corpus smoke test:

1. run corpus validation on a small local sample,
2. confirm zero mojibake and zero replacement characters,
3. confirm only Turkish and English documents,
4. confirm complete source and license metadata,
5. produce deterministic manifest hashes,
6. only then consider a tiny tokenizer experiment.

