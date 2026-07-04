# DarkMind v2 Phase 1B Execution Plan

This is a future execution plan. Phase 1A does not run these commands, does not download a corpus, does not train a tokenizer, and does not start model training.

## Entry Conditions

Before Phase 1B begins:

- `darkmind_v2/corpus/source_registry.example.json` must be replaced or copied into an approved run registry with exact source versions.
- Every approved source must pass `validate_source_registry.py`.
- The final approved source list must fit the 1GB hard download cap.
- The corpus plan must keep Turkish at 60%, English at 40%, and every single source at or below 40% of normalized characters.
- Hostile encoding samples must remain eval-only and must never enter tokenizer training data.

## Planned Commands

Validate the source registry:

```powershell
python darkmind_v2/corpus/validate_source_registry.py `
  --registry darkmind_v2/corpus/source_registry.example.json
```

Estimate tokenizer vocabulary parameter costs:

```powershell
python darkmind_v2/tokenizer/estimate_vocab_parameter_cost.py
```

Validate fixed base prompts:

```powershell
python darkmind_v2/eval/validate_fixed_prompts.py
```

Validate the local fixture corpus:

```powershell
python darkmind_v2/corpus/validate_corpus.py `
  --input darkmind_v2/tests/fixtures/sample_corpus.jsonl
```

Run tests:

```powershell
python -m pytest darkmind_v2/tests
```

## Future Phase 1B Corpus Retrieval Outline

These commands are placeholders until the approved source registry contains final pinned URLs and checksums. Do not run them in Phase 1A.

```powershell
# Future only: create an isolated data directory outside git-tracked paths.
New-Item -ItemType Directory -Force .\local_phase1b_raw | Out-Null
```

```powershell
# Future only: download exactly one approved official source at a time.
# The implementation must verify max_download_bytes before retrieval and checksum after retrieval.
python darkmind_v2/corpus/fetch_approved_source.py `
  --registry darkmind_v2/corpus/source_registry.approved.json `
  --source-id wikimedia_trwiki_articles_split `
  --out .\local_phase1b_raw
```

```powershell
# Future only: normalize and filter a local sample.
python darkmind_v2/corpus/prepare_tokenizer_pilot_sample.py `
  --registry darkmind_v2/corpus/source_registry.approved.json `
  --plan darkmind_v2/config/tokenizer_pilot_corpus.json `
  --raw-dir .\local_phase1b_raw `
  --out .\local_phase1b_processed\tokenizer_pilot.jsonl `
  --manifest-out .\local_phase1b_processed\tokenizer_pilot_manifest.json
```

```powershell
# Future only: validate the prepared local sample before any tokenizer experiment.
python darkmind_v2/corpus/validate_corpus.py `
  --input .\local_phase1b_processed\tokenizer_pilot.jsonl `
  --report-out .\local_phase1b_processed\tokenizer_pilot_validation.json
```

## Future Tokenizer Experiment Outline

Tokenizer training is forbidden until corpus validation passes with zero mojibake, zero replacement characters, zero UTF-8 failures, approved source metadata, and deterministic manifest hashes.

```powershell
# Future only: train candidate A after corpus validation passes.
python darkmind_v2/tokenizer/train_tokenizer_candidate.py `
  --candidate-id A `
  --candidate-config darkmind_v2/config/tokenizer_candidates.json `
  --corpus .\local_phase1b_processed\tokenizer_pilot.jsonl `
  --manifest .\local_phase1b_processed\tokenizer_pilot_manifest.json `
  --out .\local_phase1b_tokenizers\candidate_A
```

```powershell
# Future only: audit a trained tokenizer candidate.
python darkmind_v2/tokenizer/audit_tokenizer.py `
  --tokenizer-dir .\local_phase1b_tokenizers\candidate_A `
  --sample .\local_phase1b_processed\tokenizer_pilot.jsonl `
  --report-out .\local_phase1b_tokenizers\candidate_A_audit.json
```

```powershell
# Future only: build immutable tokenizer manifest.
python darkmind_v2/tokenizer/build_tokenizer_manifest.py `
  --tokenizer-dir .\local_phase1b_tokenizers\candidate_A `
  --training-corpus-manifest-hash <manifest_hash> `
  --tokenizer-version darkmind-v2-candidate-A `
  --special-token "<pad>" `
  --special-token "<unk>" `
  --special-token "<s>" `
  --special-token "</s>" `
  --special-token "<|system|>" `
  --special-token "<|user|>" `
  --special-token "<|assistant|>" `
  --special-token "<|end|>"
```

## Required Phase 1B Stop Points

Stop and report before tokenizer training if any source-registry validation fails.

Stop and report before tokenizer training if the prepared corpus has:

- mojibake detections,
- replacement characters,
- UTF-8 failures,
- language outside Turkish or English,
- missing source/license metadata,
- nondeterministic manifest hashes,
- source share above 40%,
- any hostile encoding eval sample in training data.

Stop and report after each tokenizer candidate audit. Do not continue into base-model training in Phase 1B.
