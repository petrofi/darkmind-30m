# Phase 2A Tokenization Plan

## Fixture Validation

The controlled eight-document fixture produced 97 tokens in three deterministic
`uint16-le` shards:

- Turkish: 161 characters, 56 tokens
- English: 171 characters, 41 tokens
- Accepted/rejected documents: 8 / 0
- EOS document boundaries: 8
- Stored shard bytes: 194
- Shard validation: PASS

The fixture is deliberately tiny, so its token ratio is not used as the primary
full-corpus estimate.

## Pilot-Corpus Estimate

The estimate uses the approved Candidate D evaluation rates and the actual
Phase 1B corpus counts:

- Turkish: 29,999,974 characters x 0.25572938 tokens/character = 7,671,875 tokens
- English: 19,999,962 characters x 0.25169088 tokens/character = 5,033,808 tokens
- One EOS per 348,859 documents = 348,859 boundary tokens
- Estimated total tokens per complete corpus epoch: **13,054,542**
- Approximate train tokens per 90% epoch: **11.75M**
- Turkish/English lexical-token shares: approximately 60.38% / 39.62%
- Raw `uint16` storage: **26,109,084 bytes** (24.90 MiB)
- Planning allowance with manifests and shard overhead: 30-40 MiB
- Complete 256-token sequence equivalents: **50,994**
- Approximate 90% train sequence equivalents: **45,895**
- EOS boundary overhead: **2.6723%**

These are planning estimates, not final counts. Domain mix and long technical
documents can move the result; full tokenization must report actual split and
language totals before training approval.

## Deterministic Full Build

The tokenization input must be a split-manifest-ordered JSONL stream containing
`id`, `split`, `language`, and normalized `text`. Phase 1B currently
stores split text and attribution metadata separately, so an approved full run
must first produce that joined stream without changing document order.

Expected command after that approval:

```powershell
python -m darkmind_v2.data_pipeline.tokenize_corpus `
  darkmind_v2/data/phase2a/input/phase1b_documents_ordered.jsonl `
  --output-dir darkmind_v2/data/phase2a/tokenized
```

No BOS token is added by default. Every document receives EOS, documents are
never split across shards, split contamination is rejected, and output records
the Phase 1B source hashes plus immutable tokenizer hashes. Phase 2A did not
create the joined full-corpus input and did not tokenize the 50M-character
corpus.
