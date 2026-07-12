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

## Phase 0 No-Training Status

Phase 0 originally contained validation infrastructure only. Phase 1B later added controlled corpus and tokenizer-candidate tooling, but no model training launcher has been authorized or run.

## Phase 1A Source And Tokenizer Planning

Phase 1A extends the validation-first design with corpus-source approval and tokenizer experiment planning. It still does not download a corpus, train a tokenizer, pretrain a model, run instruction tuning, or generate teacher data.

New Phase 1A artifacts:

- `corpus/source_registry.schema.json` defines required source, license, attribution, redistribution, commercial-use, modification, version, checksum, cap, approval, and risk fields.
- `corpus/source_registry.example.json` records currently reviewable official-source candidates.
- `corpus/validate_source_registry.py` fails closed on missing license evidence, unofficial URLs, ambiguous licenses, unapproved sources, unsupported languages, missing version/snapshot data, missing retrieval caps, Common Crawl auto-approval, and social/private/leaked/personal datasets.
- `config/tokenizer_pilot_corpus.json` defines a 30M-60M character pilot target, with the current plan set to 50M normalized characters, Turkish 60%, English 40%, train/validation/test 90/5/5, source cap <= 40%, and hard download cap 1GB.
- `config/tokenizer_candidates.json` defines four tokenizer candidates: SentencePiece BPE 12k, BPE 16k, Unigram 16k, and BPE 24k with shared special tokens.
- `tokenizer/estimate_vocab_parameter_cost.py` estimates embedding/output vocabulary parameter cost for 384 and 512 dimensions, tied and untied output heads, FP32/FP16 storage, and 45M/60M model targets.
- `config/tokenizer_acceptance_gates.json` defines hard failures and weighted scoring for future tokenizer experiments.
- `tokenizer/tokenizer_eval_samples.jsonl` provides 200 controlled eval samples covering Turkish, English, technical/code-adjacent text, source-code snippets, and hostile encoding fixtures.

The next gate is explicit source approval. A future Phase 1B run must validate the approved registry before retrieval, then validate the prepared local corpus before any tokenizer training is considered.

## Next Gate

The next gate before tokenizer training is a reviewed corpus smoke test:

1. run corpus validation on a small local sample,
2. confirm zero mojibake and zero replacement characters,
3. confirm only Turkish and English documents,
4. confirm complete source and license metadata,
5. produce deterministic manifest hashes,
6. only then consider a tiny tokenizer experiment.

## Phase 1B Result

- The final tokenizer pilot corpus contains 49,999,936 normalized characters with the approved Turkish/English balance and passed every corpus gate.
- Four SentencePiece candidates were trained and audited against fixed samples plus the complete validation/eval splits.
- Candidate D, SentencePiece BPE with a 24,000-token vocabulary, passed all hard gates and was selected with a weighted score of 90.00.
- The tokenizer is frozen under `tokenizer/frozen/darkmind_v2_sp_bpe24k_v1/` with deterministic hashes and tied-embedding constraints.
- Model training and SFT have not started.

The next phase is tiny base smoke planning for pipeline compatibility, not full pretraining. Any real base configuration must calculate its exact parameter count first and use tied input/output embeddings by default.

## Phase 2A Pipeline Preparation

Phase 1B is complete and the tokenizer under
`tokenizer/frozen/darkmind_v2_sp_bpe24k_v1/` is immutable. Phase 2A prepares
the tiny decoder-only model, frozen-tokenizer adapter, deterministic uint16
fixture shards, checkpoint/resume path, base evaluation health checks, hardware
probe, and local Hugging Face package validation.

The tiny smoke architecture has 9,369,088 parameters with tied input/output
embeddings. Fixture-only forward/backward and overfit checks validate plumbing,
not language quality.

## Phase 2B First Checkpoint Result

The first real Stage-1 checkpoint completed 256 optimizer steps over 1,048,576
training tokens. Full validation loss improved from 10.123761 to 7.096028 and
eval loss reached 7.081271. Checkpoint reload, real process restart/resume,
tokenizer/corpus provenance, and the local Hugging Face package all passed
their mechanical integrity gates.

The checkpoint is not publicly released. Its public research-preview audit
failed: 89.5% of greedy outputs received repetition warnings, 65.5% contained
exact repeated n-gram loops, and representative Turkish, English, technical,
and code continuations remained largely incoherent. Model weights are not part
of the source repository, and no Hugging Face upload occurred.

Additional base pretraining and model-quality work are required. No SFT,
teacher-data generation, or public release is allowed from this checkpoint.

